# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 11:49:58 2016

"""


from __future__ import division
import numpy as np
#from sklearn.gaussian_process import GaussianProcess
from scipy.optimize import minimize
from acquisition_functions import AcquisitionFunction, unique_rows
#from visualization import Visualization
from prada_gaussian_process import PradaGaussianProcess
from prada_gaussian_process import PradaMultipleGaussianProcess

from acquisition_maximization import acq_max_nlopt
from acquisition_maximization import acq_max_direct
from acquisition_maximization import acq_max
from sklearn.metrics.pairwise import euclidean_distances
import time
#import nlopt

#@author: Vu

#======================================================================================================
#======================================================================================================
#======================================================================================================
#======================================================================================================

class PradaBayOptFn(object):

    def __init__(self, gp_params, f, init_bounds, pbounds, acq, verbose=1, opt_toolbox='nlopt'):
        """      
        Input parameters
        ----------
        f:              function to optimize:        
        pbounds:        bounds on parameters        
        acq:            acquisition function, acq['name']=['ei','ucb','poi','lei']
                            ,acq['kappa'] for ucb, acq['k'] for lei
        opt:            optimization toolbox, 'nlopt','direct','scipy'
        
        Returns
        -------
        dim:            dimension
        bounds:         bounds on original scale
        scalebounds:    bounds on normalized scale of 0-1
        time_opt:       will record the time spent on optimization
        gp:             Gaussian Process object
        """

        # Find number of parameters
        self.dim = len(pbounds)

        # Create an array with parameters bounds
        if isinstance(pbounds,dict):
            # Get the name of the parameters
            self.keys = list(pbounds.keys())
        
            self.bounds = []
            for key in pbounds.keys():
                self.bounds.append(pbounds[key])
            self.bounds = np.asarray(self.bounds)
        else:
            self.bounds=np.asarray(pbounds)

        if len(init_bounds)==0:
            self.init_bounds=self.bounds.copy()
        else:
            self.init_bounds=init_bounds
            
        if isinstance(init_bounds,dict):
            # Get the name of the parameters
            self.keys = list(init_bounds.keys())
        
            self.init_bounds = []
            for key in init_bounds.keys():
                self.init_bounds.append(init_bounds[key])
            self.init_bounds = np.asarray(self.init_bounds)
        else:
            self.init_bounds=np.asarray(init_bounds)            
            
        # create a scalebounds 0-1
        scalebounds=np.array([np.zeros(self.dim), np.ones(self.dim)])
        self.scalebounds=scalebounds.T
        
        self.max_min_gap=self.bounds[:,1]-self.bounds[:,0]
        
        # Some function to be optimized
        self.f = f
        # optimization toolbox
        self.opt_toolbox=opt_toolbox
        # acquisition function type
        
        self.acq=acq
        
        # store X in original scale
        self.X_original= None

        # store X in 0-1 scale
        self.X = None
        
        # store y=f(x)
        # (y - mean)/(max-min)
        self.Y = None
               
        # y original scale
        self.Y_original = None
        
        self.time_opt=0

        self.k_Neighbor=2
        
        # Lipschitz constant
        self.L=0
        
        # Gaussian Process class
        self.gp=PradaGaussianProcess(gp_params)

        # acquisition function
        self.acq_func = None
    
        # stop condition
        self.stop_flag=0

    # will be later used for visualization
    def posterior(self, Xnew):
        self.gp.fit(self.X, self.Y)
        mu, sigma2 = self.gp.predict(Xnew, eval_MSE=True)
        return mu, np.sqrt(sigma2)
    
    
    def init(self, gp_params, n_init_points=3):
        """      
        Input parameters
        ----------
        gp_params:            Gaussian Process structure      
        n_init_points:        # init points
        """

        # Generate random points
        l = [np.random.uniform(x[0], x[1], size=n_init_points) for x in self.init_bounds]

        # Concatenate new random points to possible existing
        # points from self.explore method.
        temp=np.asarray(l)
        temp=temp.T
        init_X=list(temp.reshape((n_init_points,-1)))
        
        self.X_original = np.asarray(init_X)
        
        # Evaluate target function at all initialization           
        y_init=self.f(init_X)
        y_init=np.reshape(y_init,(n_init_points,1))

        self.Y_original = np.asarray(y_init)        
        self.Y=(self.Y_original-np.mean(self.Y_original))/(np.max(self.Y_original)-np.min(self.Y_original))

        # convert it to scaleX
        temp_init_point=np.divide((init_X-self.bounds[:,0]),self.max_min_gap)
        
        self.X = np.asarray(temp_init_point)
           
    def estimate_L(self,bounds):
        '''
        Estimate the Lipschitz constant of f by taking maximizing the norm of the expectation of the gradient of *f*.
        '''
        def df(x,model,x0):
            mean_derivative=gp_model.predictive_gradient(self.X,self.Y,x)
            
            temp=mean_derivative*mean_derivative
            if len(temp.shape)<=1:
                res = np.sqrt( temp)
            else:
                res = np.sqrt(np.sum(temp,axis=1)) # simply take the norm of the expectation of the gradient        

            return -res

        gp_model=self.gp
                
        dim = len(bounds)
        num_data=1000*dim
        samples = np.zeros(shape=(num_data,dim))
        for k in range(0,dim): samples[:,k] = np.random.uniform(low=bounds[k][0],high=bounds[k][1],size=num_data)

        #samples = np.vstack([samples,gp_model.X])
        pred_samples = df(samples,gp_model,0)
        x0 = samples[np.argmin(pred_samples)]

        res = minimize(df,x0, method='L-BFGS-B',bounds=bounds, args = (gp_model,x0), options = {'maxiter': 100})
        
        
        try:
            minusL = res.fun[0][0]
        except:
            if len(res.fun.shape)==1:
                minusL = res.fun[0]
            else:
                minusL = res.fun
                
        L=-minusL
        if L<1e-6: L=0.0001  ## to avoid problems in cases in which the model is flat.
        
        return L    


        
    def maximize(self,gp_params,kappa=2):
        """
        Main optimization method.

        Input parameters
        ----------

        kappa: parameter for UCB acquisition only.

        gp_params: parameter for Gaussian Process

        Returns
        -------
        x: recommented point for evaluation
        """

        if self.acq['name']=='random':
            x_max = [np.random.uniform(x[0], x[1], size=1) for x in self.bounds]
            x_max=np.asarray(x_max)
            x_max=x_max.T
            self.X_original=np.vstack((self.X_original, x_max))
            # evaluate Y using original X
            
            #self.Y = np.append(self.Y, self.f(temp_X_new_original))
            self.Y_original = np.append(self.Y_original, self.f(x_max))
            
            # update Y after change Y_original
            self.Y=(self.Y_original-np.mean(self.Y_original))/(np.max(self.Y_original)-np.min(self.Y_original))
            
            self.time_opt=np.hstack((self.time_opt,0))
            return         

            
        # init a new Gaussian Process
        self.gp=PradaGaussianProcess(gp_params)
        
        # Find unique rows of X to avoid GP from breaking
        ur = unique_rows(self.X)
        self.gp.fit(self.X[ur], self.Y[ur])
        

        # Set acquisition function
        start_opt=time.time()

        acq=self.acq
        y_max = self.Y.max()

        #self.L=self.estimate_L(self.scalebounds)
        # select the acquisition function
        if acq['name']=='nei':
            self.L=self.estimate_L(self.scalebounds)
            self.acq_func = AcquisitionFunction(kind=self.acq, L=self.L)
        else:
            self.acq_func = AcquisitionFunction(self.acq)

            if acq['name']=="ei_mu":
                #find the maximum in the predictive mean
                mu_acq={}
                mu_acq['name']='mu'
                mu_acq['dim']=self.dim
                acq_mu=AcquisitionFunction(mu_acq)
                x_mu_max = acq_max(ac=acq_mu.acq_kind,gp=self.gp,y_max=y_max,bounds=self.scalebounds,opt_toolbox=self.opt_toolbox)
                # set y_max = mu_max
                y_max=acq_mu.acq_kind(x_mu_max,gp=self.gp, y_max=y_max)

        
        x_max = acq_max(ac=self.acq_func.acq_kind,gp=self.gp,y_max=y_max,bounds=self.scalebounds,opt_toolbox=self.opt_toolbox)


        val_acq=self.acq_func.acq_kind(x_max,self.gp,y_max)
        #print "alpha[x_max]={:.5f}".format(np.ravel(val_acq)[0])
        # check the value alpha(x_max)==0
        #if val_acq<0.0001:
            #self.stop_flag=1
            #return


        # select the optimization toolbox  
        """      
        if self.opt=='nlopt':
            x_max,f_max = acq_max_nlopt(ac=self.acq_func.acq_kind,gp=self.gp,y_max=y_max,bounds=self.scalebounds)
        if self.opt=='scipy':
            
        if self.opt=='direct':
            x_max = acq_max_direct(ac=self.acq_func.acq_kind,gp=self.gp,y_max=y_max,bounds=self.scalebounds)
        """
        
        # record the optimization time
        finished_opt=time.time()
        elapse_opt=finished_opt-start_opt
        self.time_opt=np.hstack((self.time_opt,elapse_opt))
        
        # Test if x_max is repeated, if it is, draw another one at random
        if np.any((self.X - x_max).sum(axis=1) == 0):

            x_max = np.random.uniform(self.scalebounds[:, 0],
                                      self.scalebounds[:, 1],
                                      size=self.scalebounds.shape[0])
                                     
        # store X                                     
        self.X = np.vstack((self.X, x_max.reshape((1, -1))))

        # compute X in original scale
        temp_X_new_original=x_max*self.max_min_gap+self.bounds[:,0]
        self.X_original=np.vstack((self.X_original, temp_X_new_original))
        # evaluate Y using original X
        
        #self.Y = np.append(self.Y, self.f(temp_X_new_original))
        self.Y_original = np.append(self.Y_original, self.f(temp_X_new_original))
        
        # update Y after change Y_original
        self.Y=(self.Y_original-np.mean(self.Y_original))/(np.max(self.Y_original)-np.min(self.Y_original))
     
#======================================================================================
#======================================================================================================
#======================================================================================================
#======================================================================================================
