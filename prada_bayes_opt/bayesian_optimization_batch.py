# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 11:49:58 2016

"""

from __future__ import division
import numpy as np
#from sklearn.gaussian_process import GaussianProcess
from scipy.optimize import minimize
from acquisition_functions import AcquisitionFunction, unique_rows
#from prada_bayes_opt import visualization

#from visualization import Visualization
from prada_gaussian_process import PradaGaussianProcess
from prada_gaussian_process import *
from visualization import *
from prada_bayes_opt import visualization

from acquisition_maximization import *

from sklearn.metrics.pairwise import euclidean_distances
from sklearn import cluster
from sklearn import mixture
import matplotlib.pyplot as plt
from scipy.ndimage import filters
from sklearn import linear_model

import time
import copy


#import nlopt

#@author: Vu

#======================================================================================================
#======================================================================================================
#======================================================================================================
#======================================================================================================

class PradaBayOptBatch(object):

    def __init__(self,gp_params, f, pbounds, acq, verbose=1, opt_toolbox='scipy'):
        """      
        Input parameters
        ----------
        f:              function to optimize:        
        pbounds:        bounds on parameters        
        acq:            acquisition function, 'ei', 'ucb'        
        opt:            optimization toolbox, 'nlopt','direct','scipy'
        
        Returns
        -------
        dim:            dimension
        bounds:         bounds on original scale
        scalebounds:    bounds on normalized scale of 0-1
        time_opt:       will record the time spent on optimization
        gp:             Gaussian Process object
        """
        # Store the original dictionary
        self.pbounds = pbounds

        # Find number of parameters
        self.dim = len(pbounds)

        # Create an array with parameters bounds

        if isinstance(pbounds,dict):
            # Get the name of the parameters
            self.keys = list(pbounds.keys())
        
            self.bounds = []
            for key in self.pbounds.keys():
                self.bounds.append(self.pbounds[key])
            self.bounds = np.asarray(self.bounds)
        else:
            self.bounds=np.asarray(pbounds)

        scalebounds=np.array([np.zeros(self.dim), np.ones(self.dim)])
        self.scalebounds=scalebounds.T
        
        self.max_min_gap=self.bounds[:,1]-self.bounds[:,0]

        # Some function to be optimized
        self.f = f

        # optimization tool: direct, scipy, nlopt
        self.opt_toolbox=opt_toolbox
        # acquisition function type
        self.acq=acq
        
        # store the batch size for each iteration
        self.NumPoints=[]
        # Numpy array place holders
        self.X_original= None
        
        # scale the data to 0-1 fit GP better
        self.X = None # X=( X_original - min(bounds) / (max(bounds) - min(bounds))
        
        self.Y = None # Y=( Y_original - mean(bounds) / (max(bounds) - min(bounds))
        self.Y_original = None
        self.opt_time=0
        
        self.L=0 # lipschitz

        self.gp=PradaGaussianProcess(gp_params)

        # Acquisition Function
        #self.acq_func = None
        self.acq_func = AcquisitionFunction(acq=self.acq)

        
    def posterior(self, Xnew):
        #xmin, xmax = -2, 10
        ur = unique_rows(self.X)

        self.gp.fit(self.X[ur], self.Y[ur])
        mu, sigma2 = self.gp.predict(Xnew, eval_MSE=True)
        return mu, np.sqrt(sigma2)
    
    
    def init(self, n_init_points):
        """      
        Input parameters
        ----------
        gp_params:            Gaussian Process structure      
        n_init_points:        # init points
        """

        # Generate random points
        l = [np.random.uniform(x[0], x[1], size=n_init_points) for x in self.bounds]

        # Concatenate new random points to possible existing
        # points from self.explore method.
        #self.init_points += list(map(list, zip(*l)))
        temp=np.asarray(l)
        temp=temp.T
        init_X=list(temp.reshape((n_init_points,-1)))

        # Evaluate target function at all initialization           
        y_init=self.f(init_X)

        # Turn it into np array and store.
        self.X_original=np.asarray(init_X)
        temp_init_point=np.divide((init_X-self.bounds[:,0]),self.max_min_gap)
        
        self.X_original = np.asarray(init_X)
        self.X = np.asarray(temp_init_point)
        y_init=np.reshape(y_init,(n_init_points,1))
        
        self.Y_original = np.asarray(y_init)
        self.Y=(self.Y_original-np.mean(self.Y_original))/(np.max(self.Y_original)-np.min(self.Y_original))
        
        self.NumPoints=np.append(self.NumPoints,n_init_points)
        
        # Set parameters if any was passed
        #self.gp=PradaGaussianProcess(gp_params)
        
        # Find unique rows of X to avoid GP from breaking
        #ur = unique_rows(self.X)
        #self.gp.fit(self.X[ur], self.Y[ur])
        
        #print "#Batch={:d} f_max={:.4f}".format(n_init_points,self.Y.max())

    def init_with_data(self, init_X,init_Y):
        """      
        Input parameters
        ----------
        gp_params:            Gaussian Process structure      
        x,y:        # init data observations (in original scale)
        """


        # Turn it into np array and store.
        self.X_original=np.asarray(init_X)
        temp_init_point=np.divide((init_X-self.bounds[:,0]),self.max_min_gap)
        
        self.X_original = np.asarray(init_X)
        self.X = np.asarray(temp_init_point)
        
        self.Y_original = np.asarray(init_Y)
        self.Y=(self.Y_original-np.mean(self.Y_original))/(np.max(self.Y_original)-np.min(self.Y_original))
        
        self.NumPoints=np.append(self.NumPoints,len(init_Y))
        
        # Set acquisition function
        self.acq_func = AcquisitionFunction(self.acq)

        
        # Find unique rows of X to avoid GP from breaking
        ur = unique_rows(self.X)
        self.gp.fit(self.X[ur], self.Y[ur])
   
        
    def smooth_the_peak(self,my_peak):
        
        # define the local bound around the estimated point
        local_bound=np.zeros((self.dim,2))
        for dd in range(self.dim):
            try:
                local_bound[dd,0]=my_peak[-1][dd]-0.005
                local_bound[dd,1]=my_peak[-1][dd]+0.005
            except:
                local_bound[dd,0]=my_peak[dd]-0.005
                local_bound[dd,1]=my_peak[dd]+0.005
                
        local_bound=np.clip(local_bound,self.scalebounds[:,0],self.scalebounds[:,1])
                 
        dim = len(local_bound)
        num_data=1000*dim
        samples = np.zeros(shape=(num_data,dim))
        #for k in range(0,dim): samples[:,k] = np.random.uniform(low=local_bound[k][0],high=local_bound[k][1],size=num_data)
        for dd in range(0,dim): samples[:,dd] = np.linspace(local_bound[dd][0],local_bound[dd][1],num_data)

        # smooth the peak
        """
        n_bins =  100*np.ones(self.dim)
        mygrid = np.mgrid[[slice(row[0], row[1], n*1j) for row, n in zip(local_bound, n_bins)]]
        mygrid=mygrid.reshape(100**self.dim, self.dim)
        utility_grid=self.acq_func.acq_kind(mygrid,self.gp,self.Y.max())        
        
        mysamples=np.vstack((mygrid,utility_grid))
        samples_smooth=filters.uniform_filter(mysamples, size=[2,2], output=None, mode='reflect', cval=0.0, origin=0)
        """

        # get the utility after smoothing
        samples_smooth=samples
        utility_smooth=self.acq_func.acq_kind(samples_smooth,self.gp,self.Y.max()) 
        
        # get the peak value y
        #peak_y=np.max(utility_smooth)
        
        # get the peak location x
        #peak_x=samples_smooth[np.argmax(utility_smooth)]        
        
        peak_x=my_peak
        # linear regression
        regr = linear_model.LinearRegression()

        regr.fit(samples_smooth, utility_smooth)
        #residual_ss=np.mean((regr.predict(samples_smooth) - utility_smooth) ** 2)
        mystd=np.std(utility_smooth)

        return peak_x,mystd

    def check_real_peak(self,my_peak,threshold=0.1):
        
        # define the local bound around the estimated point
        local_bound=np.zeros((self.dim,2))
        for dd in range(self.dim):
            try:
                local_bound[dd,0]=my_peak[-1][dd]-0.01
                local_bound[dd,1]=my_peak[-1][dd]+0.01
            except:
                local_bound[dd,0]=my_peak[dd]-0.01
                local_bound[dd,1]=my_peak[dd]+0.01
                
        #local_bound=np.clip(local_bound,self.scalebounds[:,0],self.scalebounds[:,1])
        local_bound[:,0]=local_bound[:,0].clip(self.scalebounds[:,0],self.scalebounds[:,1])
        local_bound[:,1]=local_bound[:,1].clip(self.scalebounds[:,0],self.scalebounds[:,1])
                 
        dim = len(local_bound)
        num_data=100*dim
        samples = np.zeros(shape=(num_data,dim))
        for dd in range(0,dim): samples[:,dd] = np.linspace(local_bound[dd][0],local_bound[dd][1],num_data)

        # get the utility after smoothing
        myutility=self.acq_func.acq_kind(samples,self.gp,self.Y.max()) 
        
        # linear regression
        #regr = linear_model.LinearRegression()
        #regr.fit(samples, myutility)
        #residual_ss=np.mean((regr.predict(samples_smooth) - utility_smooth) ** 2)
        
        #mystd=np.std(myutility)
        mystd=np.mean(myutility)

        IsPeak=0
        if mystd>threshold/(self.dim**2):
            IsPeak=1
        return IsPeak,mystd
        
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
        
    def maximize_batch_PS(self,gp_params,B=5, kappa=2):
        """
        Finding a batch of points using Peak Suppression approach
        
        Input Parameters
        ----------

        gp_params:          Parameters to be passed to the Gaussian Process class
        
        kappa:              constant value in UCB
              
        Returns
        -------
        X: a batch of [x_1..x_Nt]
        """
                
        const_liar=self.Y_original.min()
        
        # Set acquisition function
        #self.acq_func = AcquisitionFunction(kind=self.acq, kappa=kappa)
        
        y_max = self.Y.max()
        
        # Set parameters if any was passed
        self.gp=PradaGaussianProcess(gp_params)
        
        # Find unique rows of X to avoid GP from breaking
        ur = unique_rows(self.X)
        self.gp.fit(self.X[ur], self.Y[ur])
        
        start_opt=time.time()        

        # copy GP, X and Y
        temp_gp=self.gp
        temp_X=self.X
        temp_Y=self.Y
        
        #store new_x
        new_X=[]
        stdPeak=[0]*B
        IsPeak=[0]*B
        for ii in range(B):
        
            # Finding argmax of the acquisition function.
            x_max = acq_max(ac=self.acq_func.acq_kind,gp=temp_gp, y_max=y_max, bounds=self.scalebounds,opt_toolbox=self.opt_toolbox)

            # Test if x_max is repeated, if it is, draw another one at random
            if np.any((np.abs(temp_X - x_max)).sum(axis=1) <0.002*self.dim) | np.isnan(x_max.sum()):
                #x_max = np.random.uniform(self.scalebounds[:, 0], self.scalebounds[:, 1],size=self.scalebounds.shape[0])
                                          
                IsPeak[ii]=0
                stdPeak[ii]=0
                print "reject"
            else:
                IsPeak[ii],stdPeak[ii]=self.check_real_peak(x_max)               
           
            print "IsPeak={:d} std={:.5f}".format(IsPeak[ii],stdPeak[ii])
                                                                 
            if ii==0:
                new_X=x_max
            else:
                new_X= np.vstack((new_X, x_max.reshape((1, -1))))
                
            temp_X = np.vstack((temp_X, x_max.reshape((1, -1))))
            temp_Y = np.append(temp_Y, const_liar )
            
            #temp_gp.fit(temp_X,temp_Y)
            temp_gp.fit_incremental(x_max, np.asarray([const_liar]))

            """
            toplot_bo=copy.deepcopy(self)
            toplot_bo.gp=copy.deepcopy(temp_gp)
            toplot_bo.X=temp_X
            toplot_bo.X_original=[val*self.max_min_gap+self.bounds[:,0] for idx, val in enumerate(temp_X)]
            toplot_bo.X_original=np.asarray(toplot_bo.X_original)
            toplot_bo.Y=temp_Y
            toplot_bo.Y_original=temp_Y*(np.max(self.Y_original)-np.min(self.Y_original))+np.mean(self.Y_original)
            visualization.plot_bo(toplot_bo)
            """

            
        IsPeak=np.asarray(IsPeak)

        # check if there is no real peak, then pick up the top peak (highest std)

        # rank the peak
        idx=np.sort(stdPeak)


        if np.sum(IsPeak)==0:
            top_peak=np.argmax(stdPeak)
            new_X=new_X[top_peak]
        else:
            new_X=new_X[IsPeak==1]
            
        print new_X

        finished_opt=time.time()
        elapse_opt=finished_opt-start_opt
        self.opt_time=np.hstack((self.opt_time,elapse_opt))

        # Updating the GP.
        #new_X=new_X.reshape((-1, self.dim))

        # Test if x_max is repeated, if it is, draw another one at random
        temp_new_X=[]
        for idx,val in enumerate(new_X):
            if np.all(np.any(np.abs(self.X-val)>0.02,axis=1)): # check if a data point is already taken
                temp_new_X=np.append(temp_new_X,val)
                
        if len(temp_new_X)==0:
            temp_new_X=np.zeros((1,self.dim))
            for idx in range(0,self.dim):
                temp_new_X[0,idx]=np.random.uniform(self.scalebounds[idx,0],self.scalebounds[idx,1],1)
        else:
            temp_new_X=temp_new_X.reshape((-1,self.dim))
         

        self.X=np.vstack((self.X, temp_new_X))
        
        # convert back to original scale
        temp_X_new_original=[val*self.max_min_gap+self.bounds[:,0] for idx, val in enumerate(temp_new_X)]
        temp_X_new_original=np.asarray(temp_X_new_original)
        self.X_original=np.vstack((self.X_original, temp_X_new_original))
        
        for idx,val in enumerate(temp_X_new_original):
            self.Y_original = np.append(self.Y_original, self.f(val))

        # update Y after change Y_original
        self.Y=(self.Y_original-np.mean(self.Y_original))/(np.max(self.Y_original)-np.min(self.Y_original))
        self.NumPoints=np.append(self.NumPoints,temp_X_new_original.shape[0])

        
        print "#Batch={:d} f_max={:.4f}".format(temp_X_new_original.shape[0],self.Y_original.max())
        
    def maximize_batch_CL(self,gp_params,B=5):
        """
        Finding a batch of points using Constant Liar approach
        
        Input Parameters
        ----------

        gp_params:          Parameters to be passed to the Gaussian Process class
        
        kappa:              constant value in UCB
              
        Returns
        -------
        X: a batch of [x_1..x_Nt]
        """
        
        self.NumPoints=np.append(self.NumPoints,B)
        
        if self.acq['name']=='random':
            x_max = [np.random.uniform(x[0], x[1], size=B) for x in self.bounds]
            x_max=np.asarray(x_max)
            x_max=x_max.T
            self.X_original=np.vstack((self.X_original, x_max))
            # evaluate Y using original X
            
            #self.Y = np.append(self.Y, self.f(temp_X_new_original))
            self.Y_original = np.append(self.Y_original, self.f(x_max))
            
            # update Y after change Y_original
            self.Y=(self.Y_original-np.mean(self.Y_original))/(np.max(self.Y_original)-np.min(self.Y_original))
            
            self.opt_time=np.hstack((self.opt_time,0))
            return     
            
        #const_liar=self.Y.mean()
        #const_liar=self.Y_original.mean()
        #const_liar=self.Y.max()
        
        # Set acquisition function
        self.acq_func = AcquisitionFunction(self.acq)
        
        y_max = self.Y.max()
        
        # Set parameters if any was passed
        self.gp=PradaGaussianProcess(gp_params)
        
        # Find unique rows of X to avoid GP from breaking
        ur = unique_rows(self.X)
        self.gp.fit(self.X[ur], self.Y[ur])
        
        start_opt=time.time()        


        # copy GP, X and Y
        temp_gp=self.gp
        temp_X=self.X
        temp_Y=self.Y
        #temp_Y_original=self.Y_original
        
        #store new_x
        new_X=[]
        for ii in range(B):
        
            # Finding argmax of the acquisition function.
            x_max = acq_max(ac=self.acq_func.acq_kind,gp=temp_gp, y_max=y_max, bounds=self.scalebounds)
            val_acq=self.acq_func.acq_kind(x_max,temp_gp,y_max)
            print "CL alpha[x_max]={:.5f}".format(np.ravel(val_acq)[0])

            # Test if x_max is repeated, if it is, draw another one at random
            # If it is repeated, print a warning
            #if np.any((self.X - x_max).sum(axis=1) == 0) | np.isnan(x_max.sum()):
                #x_max = np.random.uniform(self.scalebounds[:, 0], self.scalebounds[:, 1],size=self.scalebounds.shape[0])
                   
                                  
            if ii==0:
                new_X=x_max
            else:
                new_X= np.vstack((new_X, x_max.reshape((1, -1))))
            
            temp_X = np.vstack((temp_X, x_max.reshape((1, -1))))
            
            const_liar,const_liar_variance=temp_gp.predict(x_max,eval_MSE=1)
            temp_Y = np.append(temp_Y, const_liar )
            
            temp_gp.fit(temp_X,temp_Y)
            
         
        # Updating the GP.
        new_X=new_X.reshape((B, -1))

        finished_opt=time.time()
        elapse_opt=finished_opt-start_opt
        self.opt_time=np.hstack((self.opt_time,elapse_opt))
        
        #print new_X
        
        self.X=np.vstack((self.X, new_X))
        
        # convert back to original scale
        temp_X_new_original=[val*self.max_min_gap+self.bounds[:,0] for idx, val in enumerate(new_X)]
        temp_X_new_original=np.asarray(temp_X_new_original)
        self.X_original=np.vstack((self.X_original, temp_X_new_original))
        
        for idx,val in enumerate(temp_X_new_original):
            self.Y_original = np.append(self.Y_original, self.f(val))

        # update Y after change Y_original
        self.Y=(self.Y_original-np.mean(self.Y_original))/(np.max(self.Y_original)-np.min(self.Y_original))
        #print "#Batch={:d} f_max={:.4f}".format(B,self.Y_original.max())
        

        return new_X,temp_X_new_original
    def maximize_batch_CL_incremental(self,gp_params,B=5):
        """
        Finding a batch of points using Constant Liar approach
        
        Input Parameters
        ----------

        gp_params:          Parameters to be passed to the Gaussian Process class
        
        kappa:              constant value in UCB
              
        Returns
        -------
        X: a batch of [x_1..x_Nt]
        """
        
        self.NumPoints=np.append(self.NumPoints,B)
        
        if self.acq['name']=='random':
            x_max = [np.random.uniform(x[0], x[1], size=B) for x in self.bounds]
            x_max=np.asarray(x_max)
            x_max=x_max.T
            self.X_original=np.vstack((self.X_original, x_max))
            # evaluate Y using original X
            
            #self.Y = np.append(self.Y, self.f(temp_X_new_original))
            self.Y_original = np.append(self.Y_original, self.f(x_max))
            
            # update Y after change Y_original
            self.Y=(self.Y_original-np.mean(self.Y_original))/(np.max(self.Y_original)-np.min(self.Y_original))
            
            self.opt_time=np.hstack((self.opt_time,0))
            return     
            
        #const_liar=self.Y.mean()
        #const_liar=self.Y_original.min()
        #const_liar=self.Y.max()
        
        # Set acquisition function
        self.acq_func = AcquisitionFunction(self.acq)
        
        y_max = self.Y.max()
        
        # Set parameters if any was passed
        self.gp=PradaGaussianProcess(gp_params)
        
        # Find unique rows of X to avoid GP from breaking
        ur = unique_rows(self.X)
        self.gp.fit(self.X[ur], self.Y[ur])
        
        start_opt=time.time()        


        # copy GP, X and Y
        temp_gp=copy.deepcopy(self.gp)
        temp_X=self.X
        temp_Y=self.Y
        #temp_Y_original=self.Y_original
                
        #store new_x
        new_X=[]
        for ii in range(B):
        
            # Finding argmax of the acquisition function.
            x_max = acq_max(ac=self.acq_func.acq_kind,gp=temp_gp, y_max=y_max, bounds=self.scalebounds)
            
            # Test if x_max is repeated, if it is, draw another one at random
            if np.any(np.any(np.abs(self.X-x_max)<0.02,axis=1)): # check if a data point is already taken
                x_max = np.random.uniform(self.scalebounds[:, 0], self.scalebounds[:, 1],
                                          size=self.scalebounds.shape[0])  
            
            if ii==0:
                new_X=x_max
            else:
                new_X= np.vstack((new_X, x_max.reshape((1, -1))))  
            
            const_liar=temp_gp.predict(x_max,eval_MSE=true)

            #temp_X= np.vstack((temp_X, x_max.reshape((1, -1))))
            #temp_Y = np.append(temp_Y, const_liar )
            
            #temp_gp.fit(temp_X,temp_Y)
            
            # update the Gaussian Process and thus the acquisition function                         
            #temp_gp.compute_incremental_var(temp_X,x_max)
            temp_gp.fit_incremental(x_max,np.asarray([const_liar]))
            
         
        # Updating the GP.
        new_X=new_X.reshape((B, -1))

        finished_opt=time.time()
        elapse_opt=finished_opt-start_opt
        self.opt_time=np.hstack((self.opt_time,elapse_opt))
        
        #print new_X
        
        self.X=np.vstack((self.X, new_X))
        
        
        # convert back to original scale
        temp_X_new_original=[val*self.max_min_gap+self.bounds[:,0] for idx, val in enumerate(new_X)]
        temp_X_new_original=np.asarray(temp_X_new_original)
        self.X_original=np.vstack((self.X_original, temp_X_new_original))
        
        for idx,val in enumerate(temp_X_new_original):
            self.Y_original = np.append(self.Y_original, self.f(val))

        # update Y after change Y_original
        self.Y=(self.Y_original-np.mean(self.Y_original))/(np.max(self.Y_original)-np.min(self.Y_original))
        #print "#Batch={:d} f_max={:.4f}".format(B,self.Y_original.max())
        
    def fitIGMM(self,obs,IsPlot=0):
        """
        Fitting the Infinite Gaussian Mixture Model and GMM where applicable
        Input Parameters
        ----------
        
        obs:        samples  generated under the acqusition function by BGSS
        
        IsPlot:     flag variable for visualization    
        
        
        Returns
        -------
        mean vector: mu_1,...mu_K
        """

        if self.dim<=2:
            n_init_components=3
        else:
            n_init_components=np.int(self.dim*1.1)
            
        dpgmm = mixture.DPGMM(n_components=n_init_components,covariance_type="full",min_covar=10)
        dpgmm.fit(obs) 

        # check if DPGMM fail, then use GMM.
        mydist=euclidean_distances(dpgmm.means_,dpgmm.means_) 
        np.fill_diagonal(mydist,99)
        if dpgmm.converged_ is False or np.min(mydist)<(0.01*self.dim):
            dpgmm = mixture.GMM(n_components=n_init_components,covariance_type="full",min_covar=1e-3)
            dpgmm.fit(obs)  

        if self.dim>=5:
            # since kmeans does not provide weight and means, we will manually compute it
            try:
                dpgmm.weights_=np.histogram(dpgmm.labels_,np.int(self.dim*1.2))
                dpgmm.weights_=np.true_divide(dpgmm.weights_[0],np.sum(dpgmm.weights_[0]))
                dpgmm.means_=dpgmm.cluster_centers_
            except:
                pass

        # truncated for variational inference
        weight=dpgmm.weights_
        weight_sorted=np.sort(weight)
        weight_sorted=weight_sorted[::-1]
        temp_cumsum=np.cumsum(weight_sorted)
        
        cutpoint=0
        for idx,val in enumerate(temp_cumsum):
            if val>0.73:
                cutpoint=weight_sorted[idx]
                break
        
        ClusterIndex=[idx for idx,val in enumerate(dpgmm.weights_) if val>=cutpoint]        
                
        myMeans=dpgmm.means_[ClusterIndex]
        #dpgmm.means_=dpgmm.means_[ClusterIndex]
        dpgmm.truncated_means_=dpgmm.means_[ClusterIndex]

        #myCov=dpgmm.covars_[ClusterIndex]
        
        if IsPlot==1 and self.dim<=2:
            visualization.plot_histogram(self,obs)
            visualization.plot_mixturemodel(dpgmm,self,obs)

        new_X=myMeans.reshape((len(ClusterIndex), -1))
        new_X=new_X.tolist()
        
        return new_X
    
    def maximize_batch_B3O(self,gp_params, kappa=2,IsPlot=0):
        """
        Finding a batch of points using Budgeted Batch Bayesian Optimization approach
        
        Input Parameters
        ----------

        gp_params:          Parameters to be passed to the Gaussian Process class
        
        kappa:              constant value in UCB
        
        IsPlot:             flag variable for visualization    
        
        Returns
        -------
        X: a batch of [x_1..x_Nt]
        """
                
        # Set acquisition function
        self.acq_func = AcquisitionFunction(self.acq)
        
        # Step 2 in the Algorithm
        
        # Set parameters for Gaussian Process
        self.gp=PradaGaussianProcess(gp_params)
        
        if len(self.gp.KK_x_x_inv)==0: # check if empty
            self.gp.fit(self.X, self.Y)
        #else:
            #self.gp.fit_incremental(self.X[ur], self.Y[ur])
        
        # record optimization time
        start_gmm_opt=time.time()        
        
        if IsPlot==1 and self.dim<=2:#plot
            visualization.plot_bo(self)                
                
        # Step 4 in the Algorithm
        # generate samples from Acquisition function
        
        # check the bound 0-1 or original bound        
        obs=acq_batch_generalized_slice_sampling_generate(self.acq_func.acq_kind,self.gp,self.scalebounds,N=500,y_max=self.Y.max())
        
        # Step 5 and 6 in the Algorithm
        if len(obs)==0: # monotonous acquisition function
            print "Monotonous acquisition function"
            new_X=np.random.uniform(self.bounds[:, 0],self.bounds[:, 1],size=self.bounds.shape[0])
            new_X=new_X.reshape((1,-1))
            new_X=new_X.tolist()

        else:
            new_X=self.fitIGMM(obs,IsPlot)
            

        # Test if x_max is repeated, if it is, draw another one at random
        temp_new_X=[]
        for idx,val in enumerate(new_X):
            if np.all(np.any(np.abs(self.X-val)>0.02,axis=1)): # check if a data point is already taken
                temp_new_X=np.append(temp_new_X,val)
                
        
        if len(temp_new_X)==0:
            temp_new_X=np.zeros((1,self.dim))
            for idx in range(0,self.dim):
                temp_new_X[0,idx]=np.random.uniform(self.scalebounds[idx,0],self.scalebounds[idx,1],1)
        else:
            temp_new_X=temp_new_X.reshape((-1,self.dim))
            
        self.NumPoints=np.append(self.NumPoints,temp_new_X.shape[0])


        finished_gmm_opt=time.time()
        elapse_gmm_opt=finished_gmm_opt-start_gmm_opt
        
        self.opt_time=np.hstack((self.opt_time,elapse_gmm_opt))
        
       
        self.X=np.vstack((self.X, temp_new_X))
        
        temp_X_new_original=[val*self.max_min_gap+self.bounds[:,0] for idx, val in enumerate(temp_new_X)]
        temp_X_new_original=np.asarray(temp_X_new_original)        
        
        # Step 7 in the algorithm
        # Evaluate y=f(x)
        
        temp=self.f(temp_X_new_original)
        temp=np.reshape(temp,(-1,1))
        
        # Step 8 in the algorithm
        
        self.Y_original=np.append(self.Y_original,temp)
        
        self.Y=(self.Y_original-np.mean(self.Y_original))/(np.max(self.Y_original)-np.min(self.Y_original))


        self.X_original=np.vstack((self.X_original, temp_X_new_original))

        print "#Batch={:d} f_max={:.4f}".format(temp_new_X.shape[0],self.Y_original.max())
        
        
        #ur = unique_rows(self.X)
        #self.gp.fit(self.X[ur], self.Y[ur])
        #self.gp.fit_incremental(temp_new_X, temp_new_Y)
        
#======================================================================================
#======================================================================================================
#======================================================================================================
#======================================================================================================

    def maximize_batch_BUCB(self,gp_params, B=5):
        """
        Finding a batch of points using GP-BUCB approach
        
        Input Parameters
        ----------

        gp_params:          Parameters to be passed to the Gaussian Process class
        
        B:                  fixed batch size for all iteration
        
        kappa:              constant value in UCB
        
        IsPlot:             flag variable for visualization    
        
        
        Returns
        -------
        X: a batch of [x_1..x_B]
        """        
        self.B=B
                
        # Set acquisition function
        self.acq_func = AcquisitionFunction(self.acq)
               
        # Set parameters if any was passed
        self.gp=PradaGaussianProcess(gp_params)
        
        if len(self.gp.KK_x_x_inv)==0: # check if empty
            self.gp.fit(self.X, self.Y)
        #else:
            #self.gp.fit_incremental(self.X[ur], self.Y[ur])
        
        start_gmm_opt=time.time()
       
        y_max=self.gp.Y.max()
        # check the bound 0-1 or original bound        
        temp_X=self.X
        temp_gp=self.gp  
        temp_gp.X_bucb=temp_X
        temp_gp.KK_x_x_inv_bucb=self.gp.KK_x_x_inv
        
        # finding new X
        new_X=[]
        for ii in range(B):
            # Finding argmax of the acquisition function.
            x_max = acq_max(ac=self.acq_func.acq_kind, gp=temp_gp, y_max=y_max, bounds=self.scalebounds)
                     

            if np.any((temp_X - x_max).sum(axis=1) == 0) | np.isnan(x_max.sum()):
                x_max = np.random.uniform(self.scalebounds[:, 0],
                                          self.scalebounds[:, 1],
                                          size=self.scalebounds.shape[0])
                                          
            if ii==0:
                new_X=x_max
            else:
                new_X= np.vstack((new_X, x_max.reshape((1, -1))))
                            
            # update the Gaussian Process and thus the acquisition function                         
            temp_gp.compute_incremental_var(temp_X,x_max)

            temp_X = np.vstack((temp_X, x_max.reshape((1, -1))))
            temp_gp.X_bucb=temp_X
        
        
        # record the optimization time
        finished_gmm_opt=time.time()
        elapse_gmm_opt=finished_gmm_opt-start_gmm_opt
        
        self.time_gmm_opt=np.hstack((self.time_gmm_opt,elapse_gmm_opt))

        self.NumPoints=np.append(self.NumPoints,B)


        self.X=temp_X
                    
        # convert back to original scale
        temp_X_new_original=[val*self.max_min_gap+self.bounds[:,0] for idx, val in enumerate(new_X)]
        temp_X_new_original=np.asarray(temp_X_new_original)
        self.X_original=np.vstack((self.X_original, temp_X_new_original))
        
        # evaluate y=f(x)
        temp=self.f(temp_X_new_original)
        temp=np.reshape(temp,(-1,1))
        self.Y_original=np.append(self.Y_original,temp)
        self.Y=(self.Y_original-np.mean(self.Y_original))/(np.max(self.Y_original)-np.min(self.Y_original))
        print "#Batch={:d} f_max={:.4f}".format(new_X.shape[0],self.Y_original.max())
                
        
#======================================================================================================
        
class PradaBOFn_MulGP(object):

	def __init__(self, f, pbounds, acq='ei', verbose=1):
		"""
		:param f:
			Function to be maximized.

		:param pbounds:
			Dictionary with parameters names as keys and a tuple with minimum
			and maximum values.

		:param verbose:
			Whether or not to print progress.

		"""
		# Store the original dictionary
		self.pbounds = pbounds


		# Find number of parameters
		self.dim = len(pbounds)

		if isinstance(pbounds,dict):
            # Get the name of the parameters
			self.keys = list(pbounds.keys())
		
			self.bounds = []
			for key in self.pbounds.keys():
				self.bounds.append(self.pbounds[key])
			self.bounds = np.asarray(self.bounds)
		else:
			self.bounds=np.asarray(pbounds)

		# Some function to be optimized
		self.f = f

		# acquisition function type
		self.acq=acq

		# Initialization flag
		self.initialized = False

		# Initialization lists --- stores starting points before process begins
		self.init_points = []
		self.x_init = []
		self.y_init = []

		# Numpy array place holders
		self.X = None
		self.Y = None


		# Since scipy 0.16 passing lower and upper bound to theta seems to be
		# broken. However, there is a lot of development going on around GP
		# is scikit-learn. So I'll pick the easy route here and simple specify
		# only theta0.
		#self.gp = GaussianProcess(theta0=np.random.uniform(0.001, 0.05, self.dim),
								  #thetaL=1e-5 * np.ones(self.dim),
								  #thetaU=1e0 * np.ones(self.dim),random_start=30)

		self.gp=PradaMultipleGaussianProcess
		self.theta=[]
		# Utility Function placeholder
		self.acq_func = None


	def posterior(self, xmin=-2, xmax=10):
		#xmin, xmax = -2, 10
		self.gp.fit(self.X, self.Y)
		mu, sigma2 = self.gp.predict(np.linspace(xmin, xmax, 1000).reshape(-1, 1), eval_MSE=True)
		return mu, np.sqrt(sigma2)


	def init(self, init_points):
		# Generate random points
            l = [np.random.uniform(x[0], x[1], size=init_points) for x in self.bounds]

		# Concatenate new random points to possible existing
		# points from self.explore method.
            temp=np.asarray(l)
            self.init_points=list(temp.reshape((init_points,-1)))

        # Create empty list to store the new values of the function
            y_init = []

            # Evaluate target function at all initialization
            for x in self.init_points:
                #y_init.append(self.f(**dict(zip(self.keys, x))))
                y_init.append(self.f(x))
    
    
    		# Append any other points passed by the self.initialize method (these
    		# also have a corresponding target value passed by the user).
    		self.init_points += self.x_init
    
    		# Append the target value of self.initialize method.
    		y_init += self.y_init
    
    		# Turn it into np array and store.
    		self.X = np.asarray(self.init_points)
    		self.Y = np.asarray(y_init)
    
    		self.n_batch=1
    
    
    		self.initialized = True


	def maximize(self,
                 init_points=5,
                 n_iter=25,
                 acq='ucb',
                 kappa=2.576,
                 **gp_params):
		"""
		Main optimization method.

		Parameters
		----------
		:param init_points:
			Number of randomly chosen points to sample the
			target function before fitting the gp.

		:param n_iter:
			Total number of times the process is to repeated. Note that
			currently this methods does not have stopping criteria (due to a
			number of reasons), therefore the total number of points to be
			sampled must be specified.

		:param acq:
			Acquisition function to be used, defaults to Expected Improvement.

		:param gp_params:
			Parameters to be passed to the Scikit-learn Gaussian Process object

		Returns
		-------
		:return: Nothing
		"""
		# Reset timer
		#self.plog.reset_timer()

		# Set acquisition function
		self.acq_func = AcquisitionFunction(kind=self.acq, kappa=kappa)

		# Initialize x, y and find current y_max
		if not self.initialized:
			#if self.verbose:
				#self.plog.print_header()
			self.init(init_points)

		y_max = self.Y.max()

		self.theta=gp_params['theta']

		# Set parameters if any was passed
		#self.gp.set_params(**gp_params)
		self.gp=PradaMultipleGaussianProcess(**gp_params)

		# Find unique rows of X to avoid GP from breaking
		ur = unique_rows(self.X)
		self.gp.fit(self.X[ur], self.Y[ur])

		# Finding argmax of the acquisition function.
		x_max = acq_max(ac=self.acq_func.acq_kind,gp=self.gp,
						y_max=y_max,bounds=self.bounds)

		#print "start acq max nlopt"
		#x_max,f_max = acq_max_nlopt(f=self.acq_func.acq_kind,gp=self.gp,y_max=y_max,
							  #bounds=self.bounds)
		#print "end acq max nlopt"

		# Test if x_max is repeated, if it is, draw another one at random
		# If it is repeated, print a warning
		#pwarning = False
		if np.any((self.X - x_max).sum(axis=1) == 0):
			#print "x max uniform random"

			x_max = np.random.uniform(self.bounds[:, 0],
									  self.bounds[:, 1],
									  size=self.bounds.shape[0])
									
		#print "start append X,Y"
		self.X = np.vstack((self.X, x_max.reshape((1, -1))))
		#self.Y = np.append(self.Y, self.f(**dict(zip(self.keys, x_max))))
		self.Y = np.append(self.Y, self.f(x_max))


		#print "end append X,Y"
		#print 'x_max={:f}'.format(x_max[0])

		#print "start fitting GP"

		# Updating the GP.
		ur = unique_rows(self.X)
		self.gp.fit(self.X[ur], self.Y[ur])

		#print "end fitting GP"
		# Update maximum value to search for next probe point.
		if self.Y[-1] > y_max:
			y_max = self.Y[-1]