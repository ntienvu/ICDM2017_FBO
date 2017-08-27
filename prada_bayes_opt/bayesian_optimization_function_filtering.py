# -*- coding: utf-8 -*-
"""

"""
from __future__ import division
import numpy as np
from numpy import prod

#from sklearn.gaussian_process import GaussianProcess
from scipy.optimize import minimize
from acquisition_functions import AcquisitionFunction, unique_rows
#from visualization import Visualization
from prada_gaussian_process import PradaGaussianProcess

from acquisition_maximization import acq_max
import time
#import nlopt


#======================================================================================================
#======================================================================================================
#======================================================================================================
#======================================================================================================

class PradaBayOptFBO(object):

    def __init__(self, gp_params, f, b_init_lower, b_init_upper, b_limit_lower,b_limit_upper, acq, verbose=1, opt_toolbox='nlopt'):
        """      
        Input parameters
        ----------
        f:              function to optimize:        
        pbounds0:       bounds on parameters predefined
                
        acq:            acquisition function, acq['name']=['ei','ucb','poi','lei']
                            ,acq['kappa'] for ucb, acq['k'] for lei
        opt:            optimization toolbox, 'nlopt','direct','scipy'
        
        Returns
        -------
        dim:            dimension
        bounds0:        initial bounds on original scale
        bounds_limit:   limit bounds on original scale
        bounds:         bounds on parameters (current)
        bounds_list:    bounds at all iterations
        bounds_bk:      bounds backup for computational purpose
        scalebounds:    bounds on normalized scale of 0-1 # be careful with scaling
        scalebounds_bk: bounds on normalized scale of 0-1 backup for computation
        time_opt:       will record the time spent on optimization
        gp:             Gaussian Process object
        
        MaxIter:        Maximum number of iterations
        """

        # Find number of parameters
        self.dim = len(b_init_lower)


        self.b_init_lower=b_init_lower
        self.b_init_upper=b_init_upper
        
        self.bounds0=np.asarray([b_init_lower,b_init_upper]).T

        self.bounds = self.bounds0.copy()
        self.bounds_list = self.bounds0.copy()
        self.bounds_bk=self.bounds.copy() # keep track



        # create a scalebounds 0-1
        scalebounds=np.array([np.zeros(self.dim), np.ones(self.dim)])
        self.scalebounds=scalebounds.T
        
        self.max_min_gap=self.bounds[:,1]-self.bounds[:,0]
        self.max_min_gap_bk=self.max_min_gap.copy()
        
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
        
        # volume of initial box
        
        # compute in log space

        #self.vol0=prod(self.max_min_gap)
        self.l_radius0=np.exp(1.0*np.sum(np.log(self.max_min_gap))/self.dim)

        self.l_radius=self.l_radius0

        self.MaxIter=gp_params['MaxIter']
        
        self.b_limit_lower=b_limit_lower
        self.b_limit_upper=b_limit_upper

        # visualization purpose        
        self.X_invasion=[]

        
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
        l = [np.random.uniform(x[0], x[1], size=n_init_points) for x in self.bounds]

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
        
        self.X = self.X_original.copy()
           
    def max_volume(self,gp, max_bounds_scale,max_lcb):
        """
        A function to find the a data point that maximums a searching volume
    
        Input Parameters
        ----------
        ac: The acquisition function object that return its point-wise value.
        gp: A gaussian process fitted to the relevant data.
        y_max: The current maximum known value of the target function.
        bounds: The variables bounds to limit the search of the acq max.
        
        Returns
        -------
        x_max, The arg max of the acquisition function.
        """
    
        def compute_utility_score_for_maximizing_volume_wrapper(x_tries,gp,dim,max_lcb):
            if len(x_tries.shape)==1:
                return compute_utility_score_for_maximizing_volume(x_tries,gp,dim,max_lcb)
            return np.apply_along_axis(compute_utility_score_for_maximizing_volume,1,x_tries,gp,dim,max_lcb)
            
        def compute_utility_score_for_maximizing_volume(x_tries,gp,dim,max_lcb):
            new_bounds=self.scalebounds

            kappa=2
            mean, var = gp.predict(x_tries, eval_MSE=True)

                    
            var.flags['WRITEABLE']=True
            #var=var.copy()
            var[var<1e-10]=0

                    
            myucb=mean + kappa * np.sqrt(var)
            myucb=np.ravel(myucb)
            
            if np.asscalar(myucb)<np.asscalar(max_lcb):
                return myucb

            # store the points (outside the previous bound) that sastify the constraint   (original scale)

            # convert to original scale before adding to 
            x_tries_original=x_tries*self.max_min_gap+self.bounds_bk[:,0]
            
            # check if it is outside the old bound
            flagOutside=0
            for d in xrange(self.dim):
                if x_tries[d]> self.scalebounds_bk[d,1] or x_tries[d]<self.scalebounds_bk[d,0]: #outside the old bound
                    flagOutside=1                
                    break
            
            if flagOutside==1: # append to the invasion set
                if len(self.X_invasion)==0:
                    self.X_invasion=x_tries_original
                    self.Y_invasion=myucb
    
                else:
                    self.X_invasion=np.vstack((self.X_invasion, x_tries_original))
                    self.Y_invasion=np.vstack((self.Y_invasion, myucb))

            # expanse the bound
            for d in xrange(dim):
                # expand lower bound
                if x_tries[d]<new_bounds[d,0]:
                    new_bounds[d,0]=x_tries[d]
                    
                if x_tries[d]>new_bounds[d,1]:
                    new_bounds[d,1]=x_tries[d]  
                    
            self.scalebounds=new_bounds
            # update the utility score
            return myucb
                    
        
        dim=max_bounds_scale.shape[0]
        # Start with the lower bound as the argmax
        #x_max = max_bounds[:, 0]
        max_acq = None   
    
        myopts ={'maxiter':1000,'fatol':0.001,'xatol':0.001}
    
        # multi start
        for i in xrange(5*dim):
            # Find the minimum of minus the acquisition function
            
            x_tries = np.random.uniform(max_bounds_scale[:, 0], max_bounds_scale[:, 1],size=(100*dim, dim))
            
            # evaluate L(x)
            # estimate new L
            y_tries=compute_utility_score_for_maximizing_volume_wrapper(x_tries,gp,dim,max_lcb)

            #find x optimal for init
            idx_max=np.argmax(y_tries)
            x_init_max=x_tries[idx_max]
        
            res = minimize(lambda x: -compute_utility_score_for_maximizing_volume_wrapper(x,gp,dim,max_lcb),
               #x_init_max.reshape(1, -1),bounds=bounds,options=myopts,method="nelder-mead")#L-BFGS-B
                           x_init_max.reshape(1, -1),bounds=max_bounds_scale,options=myopts,method="L-BFGS-B")#L-BFGS-B
    
    
            # value at the estimated point
            val=compute_utility_score_for_maximizing_volume(res.x,gp,dim,max_lcb)
                
            # Store it if better than previous minimum(maximum).
            if max_acq is None or val >= max_acq:
                x_max = res.x
                max_acq = val
                #print max_acq
    
        # Clip output to make sure it lies within the bounds. Due to floating
        # point technicalities this is not always the case.


    def run_FBO(self,gp_params):
        """
        Main optimization method for filtering strategy for BO.

        Input parameters
        ----------

        gp_params: parameter for Gaussian Process

        Returns
        -------
        x: recommented point for evaluation
        """


        # for random approach
        if self.acq['name']=='random':
            x_max = [np.random.uniform(x[0], x[1], size=1) for x in self.bounds]
            x_max=np.asarray(x_max)
            x_max=x_max.T
            self.X_original=np.vstack((self.X_original, x_max))
            # evaluate Y using original X          
            self.Y_original = np.append(self.Y_original, self.f(x_max))
            
            # update Y after change Y_original
            self.Y=(self.Y_original-np.mean(self.Y_original))/(np.max(self.Y_original)-np.min(self.Y_original))
            
            self.time_opt=np.hstack((self.time_opt,0))
            return         
            
        # init a new Gaussian Process
        self.gp=PradaGaussianProcess(gp_params)
        
        # scale the data before updating the GP
        # convert it to scaleX
        self.max_min_gap=self.bounds[:,1]-self.bounds[:,0]
        temp=np.divide((self.X_original-self.bounds[:,0]),self.max_min_gap)
        self.X = np.asarray(temp)
        
        ur = unique_rows(self.X)
        self.gp.fit(self.X[ur], self.Y[ur])
        
        # Set acquisition function
        start_opt=time.time()


        # obtain the maximum on the observed set (for EI)
        y_max = self.Y.max()

        #self.L=self.estimate_L(self.scalebounds)
        # select the acquisition function

        self.acq_func = AcquisitionFunction(self.acq)

        # consider the expansion step       
        
        # finding the maximum over the lower bound
        # mu(x)-kappa x sigma(x)
        mu_acq={}
        mu_acq['name']='lcb'
        mu_acq['dim']=self.dim
        mu_acq['kappa']=2
        acq_mu=AcquisitionFunction(mu_acq)
        
        # obtain the argmax(lcb), make sure the scale bound vs original bound
        x_lcb_max = acq_max(ac=acq_mu.acq_kind,gp=self.gp,y_max=y_max,bounds=self.scalebounds,opt_toolbox=self.opt_toolbox)
        
        # obtain the max(lcb)
        max_lcb=acq_mu.acq_kind(x_lcb_max,gp=self.gp, y_max=y_max)
        max_lcb=np.ravel(max_lcb)
        
        # finding the region outside the box, that has the ucb > max_lcb        
        self.max_min_gap_bk=self.max_min_gap.copy()
        self.bounds_bk=self.bounds.copy()
        self.scalebounds_bk=self.scalebounds.copy()
        self.X_invasion=[]
        
        # the region considered is computed as follows: NewVol~OldVol*T/t
        # alternatively, we compute the radius NewL~Oldl*pow(T/t,1/d)        
        new_radius=self.l_radius*np.power(self.MaxIter/len(self.Y_original),1.0/self.dim)
        
        # extra proportion
        extra_proportion=new_radius*1.0/self.l_radius
        
        #extra_radius=(new_radius-self.l_radius)/2
        
        # check if extra radius is negative        
        if extra_proportion<1:
            extra_proportion=1
                      
        max_bounds=self.bounds.copy()
        
        # expand half to the lower bound and half to the upper bound, X'_t
        max_bounds[:,0]=max_bounds[:,0]-self.max_min_gap*(extra_proportion-1)
        max_bounds[:,1]=max_bounds[:,1]+self.max_min_gap*(extra_proportion-1)

        #max_bounds[:,0]=max_bounds[:,0]-extra_radius
        #max_bounds[:,1]=max_bounds[:,1]+extra_radius
        
        
        # make sure the max_bounds is within the limit
        if not(self.b_limit_lower is None):
            temp_max_bounds_lower=[np.maximum(max_bounds[idx,0],self.b_limit_lower[idx]) for idx in xrange(self.dim)]
            max_bounds[:,0]=temp_max_bounds_lower
        
        if not(self.b_limit_upper is None):
            temp_max_bounds_upper=[np.minimum(max_bounds[idx,1],self.b_limit_upper[idx]) for idx in xrange(self.dim)]
            max_bounds[:,1]=temp_max_bounds_upper        
            
        temp=[ (max_bounds[d,:]-self.bounds[d,0])*1.0/self.max_min_gap[d] for d in xrange(self.dim)]
        max_bounds_scale=np.asarray(temp)
        
        # find suitable candidates in new regions
        # ucb(x) > max_lcb st max L(x)
        
        # new bound in scale space
        # we note that the scalebound will be changed inside this function
        self.max_volume(self.gp, max_bounds_scale,max_lcb)
        
        #print "new bounds scale"
        #print self.scalebounds
        
        # perform standard BO on the new bound (scaled)
        x_max_scale = acq_max(ac=self.acq_func.acq_kind,gp=self.gp,y_max=y_max,bounds=self.scalebounds,opt_toolbox=self.opt_toolbox)

        val_acq=self.acq_func.acq_kind(x_max_scale,self.gp,y_max)

        # record the optimization time
        finished_opt=time.time()
        elapse_opt=finished_opt-start_opt
        self.time_opt=np.hstack((self.time_opt,elapse_opt))               
        
        # Test if x_max is repeated, if it is, draw another one at random
        if np.any((self.X - x_max_scale).sum(axis=1) == 0):
            x_max_scale = np.random.uniform(self.scalebounds[:, 0],
                                      self.scalebounds[:, 1],
                                      size=self.scalebounds.shape[0])                                     

        # check if the estimated data point is in the old bound or new
        flagOutside=0
        for d in xrange(self.dim):
            if x_max_scale[d]> self.scalebounds_bk[d,1] or x_max_scale[d]<self.scalebounds_bk[d,0]: #outside the old bound
                flagOutside=1                
                self.scalebounds[d,0]=np.minimum(x_max_scale[d],self.scalebounds_bk[d,0])
                self.scalebounds[d,1]=np.maximum(x_max_scale[d],self.scalebounds_bk[d,1]) 
            else:
                self.scalebounds[d,:]=self.scalebounds_bk[d,:]
                
                # now the scalebounds is no longer 0-1
        
        if flagOutside==0: # not outside the old bound, use the old bound
            self.scalebounds=self.scalebounds_bk
            self.bounds=self.bounds_bk.copy()
        else: # outside the old bound => expand the bound as the minimum bound containing the old bound and the selected point
            temp=[self.scalebounds[d,:]*self.max_min_gap[d]+self.bounds_bk[d,0] for d in xrange(self.dim)]
            if self.dim>1:
                self.bounds=np.reshape(temp,(self.dim,2))
            else:
                self.bounds=np.array(temp)
                
        self.bounds_list=np.hstack((self.bounds_list,self.bounds))
                 
        # compute X in original scale
        temp_X_new_original=x_max_scale*self.max_min_gap+self.bounds_bk[:,0]
        self.X_original=np.vstack((self.X_original, temp_X_new_original))
        
        # clone the self.X for updating GP
        self.max_min_gap=self.bounds[:,1]-self.bounds[:,0]
        temp=np.divide((self.X_original-self.bounds[:,0]),self.max_min_gap)
        self.X = np.asarray(temp)
        
        scalebounds=np.array([np.zeros(self.dim), np.ones(self.dim)])
        self.scalebounds=scalebounds.T
        # evaluate Y using original X
        
        self.Y_original = np.append(self.Y_original, self.f(temp_X_new_original))
        
        # update Y after change Y_original
        self.Y=(self.Y_original-np.mean(self.Y_original))/(np.max(self.Y_original)-np.min(self.Y_original))
        
        # for plotting
        self.gp=PradaGaussianProcess(gp_params)
        ur = unique_rows(self.X)
        self.gp.fit(self.X[ur], self.Y[ur])
        
        # update volume and radius
        #self.vol=prod(self.max_min_gap)
        #self.l_radius=np.power(self.vol,1/self.dim)
        self.l_radius=np.exp(1.0*np.sum(np.log(self.max_min_gap))/self.dim)

#======================================================================================
#======================================================================================================
#======================================================================================================
#======================================================================================================

    
    def maximize_volume_doubling(self,gp_params):
        """
        Volume Doubling, double the volume (e.g., gamma=2) after every 3d evaluations

        Input parameters
        ----------
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
            self.Y_original = np.append(self.Y_original, self.f(x_max))
            
            # update Y after change Y_original
            self.Y=(self.Y_original-np.mean(self.Y_original))/(np.max(self.Y_original)-np.min(self.Y_original))
            
            self.time_opt=np.hstack((self.time_opt,0))
            return         

        # init a new Gaussian Process
        self.gp=PradaGaussianProcess(gp_params)
        
        # scale the data before updating the GP
        # convert it to scaleX
        self.max_min_gap=self.bounds[:,1]-self.bounds[:,0]
        temp=np.divide((self.X_original-self.bounds[:,0]),self.max_min_gap)
        self.X = np.asarray(temp)
        
        ur = unique_rows(self.X)
        self.gp.fit(self.X[ur], self.Y[ur])
        
        # Find unique rows of X to avoid GP from breaking

        # Set acquisition function
        start_opt=time.time()

        y_max = self.Y.max()

        # select the acquisition function
        self.acq_func = AcquisitionFunction(self.acq)

        self.scalebounds_bk=self.scalebounds.copy()
        self.bounds_bk=self.bounds
        
        # consider the expansion step after 3 iterations

        if (len(self.Y) % 3)==0:
            new_radius=2.0*self.l_radius
            extra_radius=(new_radius-self.l_radius)/2
            
            max_bounds=self.bounds.copy()
            max_bounds[:,0]=max_bounds[:,0]-extra_radius
            max_bounds[:,1]=max_bounds[:,1]+extra_radius
            
            # make sure it is within the limit
            if not(self.b_limit_lower is None):
                temp_max_bounds_lower=[np.maximum(max_bounds[idx,0],self.b_limit_lower[idx]) for idx in xrange(self.dim)]
                max_bounds[:,0]=temp_max_bounds_lower
            
            if not(self.b_limit_upper is None):
                temp_max_bounds_upper=[np.minimum(max_bounds[idx,1],self.b_limit_upper[idx]) for idx in xrange(self.dim)]
                max_bounds[:,1]=temp_max_bounds_upper
            
            self.bounds=np.asarray(max_bounds).copy()            
            
            temp=[ (max_bounds[d,:]-self.bounds_bk[d,0])*1.0/self.max_min_gap[d] for d in xrange(self.dim)]
            self.scalebounds=np.asarray(temp)
        
        

            
        # perform standard BO on the new bound (scaled)
        x_max_scale = acq_max(ac=self.acq_func.acq_kind,gp=self.gp,y_max=y_max,bounds=self.scalebounds,opt_toolbox=self.opt_toolbox)

        #val_acq=self.acq_func.acq_kind(x_max_scale,self.gp,y_max)
        #print "alpha[x_max]={:.5f}".format(np.ravel(val_acq)[0])

        # record the optimization time
        finished_opt=time.time()
        elapse_opt=finished_opt-start_opt
        self.time_opt=np.hstack((self.time_opt,elapse_opt))               
        
        # Test if x_max is repeated, if it is, draw another one at random
        if np.any((self.X - x_max_scale).sum(axis=1) == 0):
            x_max_scale = np.random.uniform(self.scalebounds[:, 0],
                                      self.scalebounds[:, 1],
                                      size=self.scalebounds.shape[0])                                     
       
        # compute X in original scale
        temp_X_new_original=x_max_scale*self.max_min_gap+self.bounds_bk[:,0]
        self.X_original=np.vstack((self.X_original, temp_X_new_original))
        
        # clone the self.X for updating GP
        self.max_min_gap=self.bounds[:,1]-self.bounds[:,0]
        temp=np.divide((self.X_original-self.bounds[:,0]),self.max_min_gap)
        self.X = np.asarray(temp)
        
        scalebounds=np.array([np.zeros(self.dim), np.ones(self.dim)])
        self.scalebounds=scalebounds.T
        # evaluate Y using original X
        
        self.Y_original = np.append(self.Y_original, self.f(temp_X_new_original))
        
        # update Y after change Y_original
        self.Y=(self.Y_original-np.mean(self.Y_original))/(np.max(self.Y_original)-np.min(self.Y_original))
        
        # for plotting
        self.gp=PradaGaussianProcess(gp_params)
        ur = unique_rows(self.X)
        
        try:
            self.gp.fit(self.X[ur], self.Y[ur])
        except:
            print "bug"
                
        
        # update volume and radius
        #self.vol=prod(self.max_min_gap)
        #self.l_radius=np.power(self.vol,1/self.dim)
        self.l_radius=np.exp(1.0*np.sum(np.log(self.max_min_gap))/self.dim)

        
        
    def maximize_unbounded_regularizer(self,gp_params):
        """
        Unbounded Regularizer AISTAST 2016 Bobak

        Input parameters
        ----------
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
            self.Y_original = np.append(self.Y_original, self.f(x_max))
            
            # update Y after change Y_original
            self.Y=(self.Y_original-np.mean(self.Y_original))/(np.max(self.Y_original)-np.min(self.Y_original))
            
            self.time_opt=np.hstack((self.time_opt,0))
            return         

        # init a new Gaussian Process
        self.gp=PradaGaussianProcess(gp_params)
        
        # scale the data before updating the GP
        self.max_min_gap=self.bounds[:,1]-self.bounds[:,0]
        temp=np.divide((self.X_original-self.bounds[:,0]),self.max_min_gap)
        self.X = np.asarray(temp)
        
        ur = unique_rows(self.X)
        self.gp.fit(self.X[ur], self.Y[ur])
        
        # Find unique rows of X to avoid GP from breaking

        # Set acquisition function
        start_opt=time.time()

        y_max = self.Y.max()


        self.scalebounds_bk=self.scalebounds.copy()
        self.bounds_bk=self.bounds
        # consider the expansion step after 3 iterations

        if (len(self.Y) % 3)==0:
            new_radius=2.0*self.l_radius
            extra_radius=(new_radius-self.l_radius)/2
            
            max_bounds=self.bounds.copy()
            max_bounds[:,0]=max_bounds[:,0]-extra_radius
            max_bounds[:,1]=max_bounds[:,1]+extra_radius
            
            
                            # make sure it is within the limit
            if not(self.b_limit_lower is None):
                temp_max_bounds_lower=[np.maximum(max_bounds[idx,0],self.b_limit_lower[idx]) for idx in xrange(self.dim)]
                max_bounds[:,0]=temp_max_bounds_lower
            
            if not(self.b_limit_upper is None):
                temp_max_bounds_upper=[np.minimum(max_bounds[idx,1],self.b_limit_upper[idx]) for idx in xrange(self.dim)]
                max_bounds[:,1]=temp_max_bounds_upper
            
            self.bounds=np.asarray(max_bounds)            
            
            temp=[ (max_bounds[d,:]-self.bounds[d,0])*1.0/self.max_min_gap[d] for d in xrange(self.dim)]
            self.scalebounds=np.asarray(temp)
        
        
        
        # select the acquisition function
        self.acq['x_bar']=np.mean(self.bounds)
        self.acq['R']=np.power(self.l_radius,1.0/self.dim)


        self.acq_func = AcquisitionFunction(self.acq)
        
        # mean of the domain

        #acq['R']
        
        x_max_scale = acq_max(ac=self.acq_func.acq_kind,gp=self.gp,y_max=y_max,bounds=self.scalebounds,opt_toolbox=self.opt_toolbox)

        #val_acq=self.acq_func.acq_kind(x_max_scale,self.gp,y_max)
        #print "alpha[x_max]={:.5f}".format(np.ravel(val_acq)[0])

        # record the optimization time
        finished_opt=time.time()
        elapse_opt=finished_opt-start_opt
        self.time_opt=np.hstack((self.time_opt,elapse_opt))               
        
        # Test if x_max is repeated, if it is, draw another one at random
        if np.any((self.X - x_max_scale).sum(axis=1) == 0):
            x_max_scale = np.random.uniform(self.scalebounds[:, 0],
                                      self.scalebounds[:, 1],
                                      size=self.scalebounds.shape[0])                                     

        # check if the estimated data point is in the old bound or new
        flagOutside=0
        for d in xrange(self.dim):
            if x_max_scale[d]> self.scalebounds_bk[d,1] or x_max_scale[d]<self.scalebounds_bk[d,0]: #outside the old bound
                flagOutside=1                
                self.scalebounds[d,0]=np.minimum(x_max_scale[d],self.scalebounds_bk[d,0])
                self.scalebounds[d,1]=np.maximum(x_max_scale[d],self.scalebounds_bk[d,1])                
                
                # now the scalebounds is no longer 0-1
        
        if flagOutside==0: # not outside the old bound
            self.scalebounds=self.scalebounds_bk
        else: # inside the old bound => recompute bound
            temp=[self.scalebounds[d,:]*self.max_min_gap[d]+self.bounds_bk[d,0] for d in xrange(self.dim)]
            if self.dim>1:
                self.bounds=np.reshape(temp,(self.dim,2))
            else:
                self.bounds=np.array(temp)
                
        # compute X in original scale
        temp_X_new_original=x_max_scale*self.max_min_gap+self.bounds_bk[:,0]
        self.X_original=np.vstack((self.X_original, temp_X_new_original))
        
        # clone the self.X for updating GP
        self.max_min_gap=self.bounds[:,1]-self.bounds[:,0]
        temp=np.divide((self.X_original-self.bounds[:,0]),self.max_min_gap)
        self.X = np.asarray(temp)
        
        scalebounds=np.array([np.zeros(self.dim), np.ones(self.dim)])
        self.scalebounds=scalebounds.T
        # evaluate Y using original X
        
        self.Y_original = np.append(self.Y_original, self.f(temp_X_new_original))
        
        # update Y after change Y_original
        self.Y=(self.Y_original-np.mean(self.Y_original))/(np.max(self.Y_original)-np.min(self.Y_original))
        
        # for plotting
        self.gp=PradaGaussianProcess(gp_params)
        ur = unique_rows(self.X)
        self.gp.fit(self.X[ur], self.Y[ur])
        
        # update volume and radius
        #self.vol=prod(self.max_min_gap)
        #self.l_radius=np.power(self.vol,1/self.dim)
        self.l_radius=np.exp(1.0*np.sum(np.log(self.max_min_gap))/self.dim)
        
        
        
    def maximize_expanding_volume_L(self,gp_params):
        """
        Expanding volume following L ~ MaxIter

        Input parameters
        ----------

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
            self.Y_original = np.append(self.Y_original, self.f(x_max))
            
            # update Y after change Y_original
            self.Y=(self.Y_original-np.mean(self.Y_original))/(np.max(self.Y_original)-np.min(self.Y_original))
            
            self.time_opt=np.hstack((self.time_opt,0))
            return         

            
        # init a new Gaussian Process
        self.gp=PradaGaussianProcess(gp_params)
        
        # scale the data before updating the GP
        # convert it to scaleX
        self.max_min_gap=self.bounds[:,1]-self.bounds[:,0]
        temp=np.divide((self.X_original-self.bounds[:,0]),self.max_min_gap)
        self.X = np.asarray(temp)
        
        ur = unique_rows(self.X)
        self.gp.fit(self.X[ur], self.Y[ur])
        
        # Set acquisition function
        start_opt=time.time()

        y_max = self.Y.max()

        #self.L=self.estimate_L(self.scalebounds)
        # select the acquisition function

        self.acq_func = AcquisitionFunction(self.acq)

        # consider the expansion step       


        # backup the previous bounds        
        self.bounds_bk=self.bounds.copy()
        self.scalebounds_bk=self.scalebounds.copy()
        
        # the region considered is computed as follows: NewVol~OldVol*T/t
        # alternatively, we compute the radius NewL~Oldl*pow(T/t,1/d)        
        new_radius=self.l_radius*np.power(self.MaxIter/len(self.Y_original),1.0/self.dim)
        # extra proportion
        extra_proportion=new_radius*1.0/self.l_radius
        
        #extra_radius=(new_radius-self.l_radius)/2
        
        if extra_proportion<1:
            extra_proportion=1
                      
        max_bounds=self.bounds.copy()
        
        # expand half to the lower bound and half to the upper bound
        max_bounds[:,0]=max_bounds[:,0]-self.max_min_gap*(extra_proportion-1)*0.5
        max_bounds[:,1]=max_bounds[:,1]+self.max_min_gap*(extra_proportion-1)*0.5
        
                        # make sure it is within the limit
        if not(self.b_limit_lower is None):
            temp_max_bounds_lower=[np.maximum(max_bounds[idx,0],self.b_limit_lower[idx]) for idx in xrange(self.dim)]
            max_bounds[:,0]=temp_max_bounds_lower
        
        if not(self.b_limit_upper is None):
            temp_max_bounds_upper=[np.minimum(max_bounds[idx,1],self.b_limit_upper[idx]) for idx in xrange(self.dim)]
            max_bounds[:,1]=temp_max_bounds_upper
            
        temp=[ (max_bounds[d,:]-self.bounds_bk[d,0])*1.0/self.max_min_gap[d] for d in xrange(self.dim)]
        self.scalebounds=np.asarray(temp)
        
        
        # perform standard BO on the new bound (scaled)
        x_max_scale = acq_max(ac=self.acq_func.acq_kind,gp=self.gp,y_max=y_max,bounds=self.scalebounds,opt_toolbox=self.opt_toolbox)

        #val_acq=self.acq_func.acq_kind(x_max_scale,self.gp,y_max)
        #print "alpha[x_max]={:.5f}".format(np.ravel(val_acq)[0])

        # record the optimization time
        finished_opt=time.time()
        elapse_opt=finished_opt-start_opt
        self.time_opt=np.hstack((self.time_opt,elapse_opt))               
        
        # Test if x_max is repeated, if it is, draw another one at random
        if np.any((self.X - x_max_scale).sum(axis=1) == 0):
            x_max_scale = np.random.uniform(self.scalebounds[:, 0],
                                      self.scalebounds[:, 1],
                                      size=self.scalebounds.shape[0])                                     

        # check if the estimated data point is in the old bound or new for cropping
        IsCropping=0
        if IsCropping==1:
            flagOutside=0
            for d in xrange(self.dim):
                if x_max_scale[d]> self.scalebounds_bk[d,1] or x_max_scale[d]<self.scalebounds_bk[d,0]: #outside the old bound
                    flagOutside=1                
                    self.scalebounds[d,0]=np.minimum(x_max_scale[d],self.scalebounds_bk[d,0])
                    self.scalebounds[d,1]=np.maximum(x_max_scale[d],self.scalebounds_bk[d,1])                
                    
                    # now the scalebounds is no longer 0-1
            
            if flagOutside==0: # not outside the old bound
                self.scalebounds=self.scalebounds_bk
                self.bounds=self.bounds_bk.copy()
            else: # inside the old bound => recompute bound
                temp=[self.scalebounds[d,:]*self.max_min_gap[d]+self.bounds_bk[d,0] for d in xrange(self.dim)]
                if self.dim>1:
                    self.bounds=np.reshape(temp,(self.dim,2))
                else:
                    self.bounds=np.array(temp)
        else:
                temp=[self.scalebounds[d,:]*self.max_min_gap[d]+self.bounds_bk[d,0] for d in xrange(self.dim)]
                if self.dim>1:
                    self.bounds=np.reshape(temp,(self.dim,2))
                else:
                    self.bounds=np.array(temp)
                                     
        # compute X in original scale
        temp_X_new_original=x_max_scale*self.max_min_gap+self.bounds_bk[:,0]
        self.X_original=np.vstack((self.X_original, temp_X_new_original))
        
        # clone the self.X for updating GP
        self.max_min_gap=self.bounds[:,1]-self.bounds[:,0]
        temp=np.divide((self.X_original-self.bounds[:,0]),self.max_min_gap)
        self.X = np.asarray(temp)
        
        scalebounds=np.array([np.zeros(self.dim), np.ones(self.dim)])
        self.scalebounds=scalebounds.T
        # evaluate Y using original X
        
        self.Y_original = np.append(self.Y_original, self.f(temp_X_new_original))
        
        # update Y after change Y_original
        self.Y=(self.Y_original-np.mean(self.Y_original))/(np.max(self.Y_original)-np.min(self.Y_original))
        
        # for plotting
        self.gp=PradaGaussianProcess(gp_params)
        ur = unique_rows(self.X)
        self.gp.fit(self.X[ur], self.Y[ur])
        
        # update volume and radius
        #self.vol=prod(self.max_min_gap)
        #self.l_radius=np.power(self.vol,1/self.dim)
        self.l_radius=np.exp(1.0*np.sum(np.log(self.max_min_gap))/self.dim)
        
    def maximize_expanding_volume_L_Cropping(self,gp_params):
        """
        Expanding volume following L ~ MaxIter

        Input parameters
        ----------

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
            self.Y_original = np.append(self.Y_original, self.f(x_max))
            
            # update Y after change Y_original
            self.Y=(self.Y_original-np.mean(self.Y_original))/(np.max(self.Y_original)-np.min(self.Y_original))
            
            self.time_opt=np.hstack((self.time_opt,0))
            return         

            
        # init a new Gaussian Process
        self.gp=PradaGaussianProcess(gp_params)
        
        # scale the data before updating the GP
        # convert it to scaleX
        self.max_min_gap=self.bounds[:,1]-self.bounds[:,0]
        temp=np.divide((self.X_original-self.bounds[:,0]),self.max_min_gap)
        self.X = np.asarray(temp)
        
        ur = unique_rows(self.X)
        self.gp.fit(self.X[ur], self.Y[ur])
        
        # Set acquisition function
        start_opt=time.time()

        y_max = self.Y.max()

        #self.L=self.estimate_L(self.scalebounds)
        # select the acquisition function

        self.acq_func = AcquisitionFunction(self.acq)

        # consider the expansion step       


        # finding the region outside the box, that has the ucb > max_lcb
        
        self.bounds_bk=self.bounds.copy()
        self.scalebounds_bk=self.scalebounds.copy()
        
        # the region considered is computed as follows: NewVol~OldVol*T/t
        # alternatively, we compute the radius NewL~Oldl*pow(T/t,1/d)        
        new_radius=self.l_radius*np.power(self.MaxIter/len(self.Y_original),1.0/self.dim)
        # extra proportion
        extra_proportion=new_radius*1.0/self.l_radius
        
        #extra_radius=(new_radius-self.l_radius)/2
        
        # check if extra radius is negative        
        #if extra_radius<0:
            #extra_radius=0
                      
        max_bounds=self.bounds.copy()
        
        # expand half to the lower bound and half to the upper bound
        max_bounds[:,0]=max_bounds[:,0]-self.max_min_gap*extra_proportion
        max_bounds[:,1]=max_bounds[:,1]+self.max_min_gap*extra_proportion
        
        # make sure the max_bound is still within the limit
        if not(self.b_limit_lower is None):
            temp_max_bounds_lower=[np.maximum(max_bounds[idx,0],self.b_limit_lower[idx]) for idx in xrange(self.dim)]
            max_bounds[:,0]=temp_max_bounds_lower
        
        if not(self.b_limit_upper is None):
            temp_max_bounds_upper=[np.minimum(max_bounds[idx,1],self.b_limit_upper[idx]) for idx in xrange(self.dim)]
            max_bounds[:,1]=temp_max_bounds_upper
            
        temp=[ (max_bounds[d,:]-self.bounds_bk[d,0])*1.0/self.max_min_gap[d] for d in xrange(self.dim)]
        self.scalebounds=np.asarray(temp)
        
        
        # perform standard BO on the new bound (scaled)
        x_max_scale = acq_max(ac=self.acq_func.acq_kind,gp=self.gp,y_max=y_max,bounds=self.scalebounds,opt_toolbox=self.opt_toolbox)

        #val_acq=self.acq_func.acq_kind(x_max_scale,self.gp,y_max)
        #print "alpha[x_max]={:.5f}".format(np.ravel(val_acq)[0])

        # record the optimization time
        finished_opt=time.time()
        elapse_opt=finished_opt-start_opt
        self.time_opt=np.hstack((self.time_opt,elapse_opt))               
        
        # Test if x_max is repeated, if it is, draw another one at random
        if np.any((self.X - x_max_scale).sum(axis=1) == 0):
            x_max_scale = np.random.uniform(self.scalebounds[:, 0],
                                      self.scalebounds[:, 1],
                                      size=self.scalebounds.shape[0])                                     

        # check if the estimated data point is in the old bound or new for cropping
        IsCropping=1
        if IsCropping==1:
            flagOutside=0
            for d in xrange(self.dim):
                if x_max_scale[d]> self.scalebounds_bk[d,1] or x_max_scale[d]<self.scalebounds_bk[d,0]: #outside the old bound
                    flagOutside=1                
                    self.scalebounds[d,0]=np.minimum(x_max_scale[d],self.scalebounds_bk[d,0])
                    self.scalebounds[d,1]=np.maximum(x_max_scale[d],self.scalebounds_bk[d,1])  
                else:
                    self.scalebounds[d,:]=self.scalebounds_bk[d,:]
                    
                    # now the scalebounds is no longer 0-1
            
            if flagOutside==0: # not outside the old bound
                self.scalebounds=self.scalebounds_bk
                self.bounds=self.bounds_bk.copy()
            else: # inside the old bound => recompute bound
                temp=[self.scalebounds[d,:]*self.max_min_gap[d]+self.bounds_bk[d,0] for d in xrange(self.dim)]
                if self.dim>1:
                    self.bounds=np.reshape(temp,(self.dim,2))
                else:
                    self.bounds=np.array(temp)
        else:
                temp=[self.scalebounds[d,:]*self.max_min_gap[d]+self.bounds_bk[d,0] for d in xrange(self.dim)]
                if self.dim>1:
                    self.bounds=np.reshape(temp,(self.dim,2))
                else:
                    self.bounds=np.array(temp)
                                     
        # compute X in original scale
        temp_X_new_original=x_max_scale*self.max_min_gap+self.bounds_bk[:,0]
        self.X_original=np.vstack((self.X_original, temp_X_new_original))
        
        # clone the self.X for updating GP
        self.max_min_gap=self.bounds[:,1]-self.bounds[:,0]
        temp=np.divide((self.X_original-self.bounds[:,0]),self.max_min_gap)
        self.X = np.asarray(temp)
        
        scalebounds=np.array([np.zeros(self.dim), np.ones(self.dim)])
        self.scalebounds=scalebounds.T
        # evaluate Y using original X
        
        self.Y_original = np.append(self.Y_original, self.f(temp_X_new_original))
        
        # update Y after change Y_original
        self.Y=(self.Y_original-np.mean(self.Y_original))/(np.max(self.Y_original)-np.min(self.Y_original))
        
        # for plotting
        self.gp=PradaGaussianProcess(gp_params)
        ur = unique_rows(self.X)
        self.gp.fit(self.X[ur], self.Y[ur])
        
        # update volume and radius
        #self.vol=prod(self.max_min_gap)
        #self.l_radius=np.power(self.vol,1/self.dim)
        self.l_radius=np.exp(1.0*np.sum(np.log(self.max_min_gap))/self.dim)