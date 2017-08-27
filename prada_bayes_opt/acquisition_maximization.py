# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 11:51:41 2016

@author: Vu
"""

from __future__ import division
import itertools
import numpy as np
#from sklearn.gaussian_process import GaussianProcess
from scipy.optimize import minimize
from scipy.optimize import fmin_cobyla
#from helpers import UtilityFunction, unique_rows
#from visualization import Visualization
#from prada_gaussian_process import PradaGaussianProcess
#from prada_gaussian_process import PradaMultipleGaussianProcess
import random

#from ..util.general import multigrid, samples_multidimensional_uniform, reshape

__author__ = 'Vu'

def acq_max_nlopt(ac,gp,y_max,bounds):
    """
    A function to find the maximum of the acquisition function using
    the 'NLOPT' library.

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
    
    try:
        import nlopt
    except:
        print("Cannot find nlopt library")
    
    def objective(x, grad):
            """Objective function in the form required by nlopt."""
            #print "=================================="
            if grad.size > 0:
                fx, gx = ac(x[None], grad=True)
                grad[:] = gx[0][:]
            else:
                try:
                    if ac['name']=='pes':
                        fx = ac(x)
                    else:
                        fx = ac(x,gp,y_max)
                    if isinstance(fx,list):
                        fx=fx[0]
                    
                    #print fx
                except:
                    return 0
            return fx[0]
            
    tol=1e-6
    bounds = np.array(bounds, ndmin=2)

    dim=bounds.shape[0]
    opt = nlopt.opt(nlopt.GN_DIRECT, dim)
    #opt = nlopt.opt(nlopt.LD_MMA, bounds.shape[0])

    opt.set_lower_bounds(bounds[:, 0])
    opt.set_upper_bounds(bounds[:, 1])
    #opt.set_ftol_rel(tol)
    opt.set_maxeval(5000*dim)
    opt.set_xtol_abs(tol)

    #opt.set_ftol_abs(tol)#Set relative tolerance on function value.
    #opt.set_xtol_rel(tol)#Set absolute tolerance on function value.
    #opt.set_xtol_abs(tol) #Set relative tolerance on optimization parameters.

    opt.set_maxtime=5000*dim
    
    opt.set_max_objective(objective)    

    xinit=random.uniform(bounds[:,0],bounds[:,1])
    #xinit=np.asarray(0.2)
    #xoptimal = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0])*1.0 / 2
    #print xoptimal
    
    try:
        xoptimal = opt.optimize(xinit.copy())

    except:
        xoptimal=xinit
        #xoptimal = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0])*1.0 / 2
     
    fmax= opt.last_optimum_value()
    
    #print "nlopt force stop ={:s}".format(nlopt_result)
    #fmax=opt.last_optimize_result()
    
    code=opt.last_optimize_result()
    status=1

    if code<4:
        print "nlopt code = {:d}".format(code)
        status=0

    return xoptimal, fmax, status

    
def acq_max_direct(ac,gp,y_max,bounds):
    """
    A function to find the maximum of the acquisition function using
    the 'DIRECT' library.

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
    
    try:
        from DIRECT import solve
    except:
        print("Cannot find DIRECT library")
        
    def DIRECT_f_wrapper(ac):
        def g(x, user_data):
            fx=ac(np.array([x]),gp,y_max)
            #print fx[0]
            return fx[0], 0
        return g
            
    lB = np.asarray(bounds)[:,0]
    uB = np.asarray(bounds)[:,1]
    
    #x,_,_ = solve(DIRECT_f_wrapper(f),lB,uB, maxT=750, maxf=2000,volper=0.005) # this can be used to speed up DIRECT (losses precission)
    x,_,_ = solve(DIRECT_f_wrapper(ac),lB,uB,maxT=2000,maxf=2000,volper=0.0005)
    return np.reshape(x,len(bounds))
    
def acq_max(ac, gp, y_max, bounds, opt_toolbox='scipy'):
    """
    A function to find the maximum of the acquisition function using
    the scipy python

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
    if opt_toolbox=='nlopt':
        x_max,f_max,status = acq_max_nlopt(ac=ac,gp=gp,y_max=y_max,bounds=bounds)
        
        if status==0:# if nlopt fails, let try scipy
            opt_toolbox='scipy'
            
    if opt_toolbox=='direct':
        x_max = acq_max_direct(ac=ac,gp=gp,y_max=y_max,bounds=bounds)
    if opt_toolbox=='scipy':
        x_max = acq_max_scipy(ac=ac,gp=gp,y_max=y_max,bounds=bounds)
    if opt_toolbox=='cobyla':
        x_max = acq_max_cobyla(ac=ac,gp=gp,y_max=y_max,bounds=bounds)
    return x_max
        
def acq_max_scipy(ac, gp, y_max, bounds):
    """
    A function to find the maximum of the acquisition function using
    the scipy python

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

    dim=bounds.shape[0]
    # Start with the lower bound as the argmax
    x_max = bounds[:, 0]
    max_acq = None

    #x_tries = np.array([ np.linspace(i,j,500) for i,j in zip( bounds[:, 0], bounds[:, 1])])
    #x_tries=x_tries.T

    myopts ={'maxiter':2000,'fatol':0.0001,'xatol':0.0001}


    # multi start
    for i in xrange(5*dim):
        # Find the minimum of minus the acquisition function
        
        x_tries = np.random.uniform(bounds[:, 0], bounds[:, 1],size=(100*dim, dim))
    
    
        # evaluate
        y_tries=ac(x_tries,gp=gp, y_max=y_max)
        
        #find x optimal for init
        idx_max=np.argmax(y_tries)
        x_init_max=x_tries[idx_max]
    
        res = minimize(lambda x: -ac(x.reshape(1, -1), gp=gp, y_max=y_max),
           #x_init_max.reshape(1, -1),bounds=bounds,options=myopts,method="nelder-mead")#L-BFGS-B
                       x_init_max.reshape(1, -1),bounds=bounds,options=myopts,method="L-BFGS-B")#L-BFGS-B

        # value at the estimated point
        val=ac(res.x,gp,y_max)
            
        # Store it if better than previous minimum(maximum).
        if max_acq is None or val >= max_acq:
            x_max = res.x
            max_acq = val
            #print max_acq

    # Clip output to make sure it lies within the bounds. Due to floating
    # point technicalities this is not always the case.
    return np.clip(x_max, bounds[:, 0], bounds[:, 1])

def acq_max_single_seed(ac, gp, y_max, bounds):
    """
    A function to find the maximum of the acquisition function using
    the 'L-BFGS-B' method.

    Input Parameters
    ----------
    ac: The acquisition function object that return its point-wise value.
    gp: A gaussian process fitted to the relevant data.
    y_max: The current maximum known value of the target function.
    bounds: The variables bounds to limit the search of the acq max.
    
    Returns
    -------
    :return: x_max, The arg max of the acquisition function.
    """

    # Start with the lower bound as the argmax
    x_max = bounds[:, 0]
    #max_acq = None

   
    x_try=np.array(bounds[:, 0])

    # Find the minimum of minus the acquisition function
    res = minimize(lambda x: -ac(x.reshape(1, -1), gp=gp, y_max=y_max),
                   x_try.reshape(1, -1),
                   bounds=bounds,
                   method="L-BFGS-B")

    x_max = res.x
    #max_acq = -res.fun

    # Clip output to make sure it lies within the bounds. Due to floating
    # point technicalities this is not always the case.
    return np.clip(x_max, bounds[:, 0], bounds[:, 1])
    
def acq_batch_generalized_slice_sampling_generate(acq_fun,gp,pbounds,N,y_max,SliceBlock=200):
    """
    A Batch Generalized Slice Sampling technique to draw samples under the acquisition function

    Input Parameters
    ----------
    acq_fun: The acquisition function object that return its point-wise value.
    gp: A gaussian process fitted to the relevant data.
    y_max: The current maximum known value of the target function.
    pbounds: The variables bounds to limit the search of the acq max.
    N: number of points to generate
    SliceBlock: block size for batch slice sampling
    
    Returns
    -------
    samples
    """
    
    if isinstance(pbounds,dict):
        # Get the name of the parameters
        #keys = list(pbounds.keys())
        mybounds = []
        for key in pbounds.keys():
            mybounds.append(pbounds[key])
        mybounds = np.asarray(mybounds)
    else:
        mybounds=np.asarray(pbounds)
    
    D=mybounds.size/2
    D=np.int(D)
    
    N=N*D       

    # find minimum value in acquisition function
    x0=np.zeros(D)
    for idx in range(0,D):
        #x0[idx]=np.random.uniform(bounds[idx,0],bounds[idx,1],1)
        x0[idx]=np.random.uniform(0,1,1)#the scaled data is from 0 to 1
        
    x_max = mybounds[:, 0]
    y_min = None

    #scaled_bounds=np.vstack((np.zeros(D), np.ones(D)))
    #scaled_bounds=np.asarray(scaled_bounds).T

    # finding the minimum from the acquisition function
    x_tries = np.random.uniform(np.zeros(D), np.ones(D), size=(5, mybounds.shape[0]))
    myopts ={'maxiter':100}

    for x_try in x_tries:
        #res = minimize(acq_fun, x_try.reshape(1, -1), args=(gp,y_max), bounds=bounds, options=myopts, method="L-BFGS-B")#L-BFGS-B
        res = minimize(acq_fun, x_try.reshape(1, -1), args=(gp,y_max), bounds=mybounds, options=myopts, method="L-BFGS-B")#L-BFGS-B

        # Store it if better than previous minimum(maximum).
        if y_min is None or res.fun <= y_min:
            #x_min = res.x
            y_min = res.fun

    # make sure the dimension of y_min
    y_min=y_min.ravel()
    
    #     counter for #samples rejected
    count_reject=0
    count=0
    
    samples=[]

    # init x0
    x0=np.zeros(D)
    for idx in range(0,D):
        x0[idx]=np.random.uniform(mybounds[idx,0],mybounds[idx,1],1)
                
    # evaluate alpha(x0) from the acquisition function
    fx=acq_fun(x0,gp,y_max)
    fx=fx.ravel()
    fx=np.dot(fx,np.ones((1,SliceBlock)))
    idxAccept=range(0,SliceBlock)
    y=fx
    cut_min=np.dot(y_min,np.ones((1,SliceBlock)))
    
    while(count<N):
        # sampling y

        # make a threshold (cut_min) to draw samples under the peaks, but above the threshold    
        for idx in range(0,SliceBlock):
            if idx in idxAccept:
                temp=np.linspace(y_min,fx[idx],100)
                temp=temp.ravel()
                cut_min[idx]=np.percentile(temp,85)
        
        y[idxAccept]=np.random.uniform(cut_min[idxAccept],fx[idxAccept],len(idxAccept))
          
        # sampling x
        x=np.zeros((SliceBlock,D))
        for idx in range(0,D):
            x[:,idx]=np.random.uniform(mybounds[idx,0],mybounds[idx,1],SliceBlock)
        
        # get f(x)=alpha(x)
        fx=acq_fun(x,gp,y_max)             
        fx=fx.ravel()

        idxAccept=[idx for idx,val in enumerate(fx) if val>cut_min[idx] and val>y[idx]]
            
        if len(samples)==0:
            samples=x[idxAccept,:]
        else:
            samples=np.vstack((samples,x[idxAccept,:]))
        count=len(samples)            
        count_reject=count_reject+len(fx)-len(idxAccept)

        # stop the sampling process if #rejected samples excesses a threshold            
        if count_reject>N*10:
            print 'BGSS count_reject={:d}, count_accept={:d}'.format(count_reject,count)
            
            if count<5:
                samples=[]
            return samples     
        
    return np.asarray(samples)
    
def local_slice_sampling(acq_fun,gp,N,D,bounds,SliceBlock,y_min,y_max):
    """
    A Batch Generalized Slice Sampling technique to draw samples under the acquisition function

    Input Parameters
    ----------
    acq_fun: The acquisition function object that return its point-wise value.
    gp: A gaussian process fitted to the relevant data.
    y_max: The current maximum known value of the target function.
    pbounds: The variables bounds to limit the search of the acq max.
    N: number of points to generate
    SliceBlock: block size for batch slice sampling
    
    Returns
    -------
    samples
    """
    
    samples=[]
    count=0
    count_reject=0
    # init x0
    x0=np.zeros(D)
    for idx in range(0,D):
        x0[idx]=np.random.uniform(bounds[idx,0],bounds[idx,1],1)
                
    
    fx=acq_fun(x0,gp,y_max)
    fx=np.dot(fx,np.ones((1,SliceBlock)))
    idxAccept=range(0,SliceBlock)
    y=fx
    cut_min=np.dot(y_min,np.ones((1,SliceBlock)))
    
    while(count<N):
        # sampling y
    
        for idx in range(0,SliceBlock):
            if idx in idxAccept:
                temp=np.linspace(y_min,fx[idx],100)
                cut_min[idx]=np.percentile(temp,80)
        
        y[idxAccept]=np.random.uniform(cut_min[idxAccept],fx[idxAccept],len(idxAccept))
          
        # sampling x
        x=np.zeros((SliceBlock,D))
        for idx in range(0,D):
            x[:,idx]=np.random.uniform(bounds[idx,0],bounds[idx,1],SliceBlock)
        
        
        
        # get f(x)
        fx=acq_fun(x,gp,y_max)             
        
        idxAccept=[idx for idx,val in enumerate(fx) if val>cut_min[idx] and val>y[idx]]
        if len(samples)==0:
            samples=x[idxAccept,:]
        else:
            samples=np.vstack((samples,x[idxAccept,:]))
        count=len(samples)            
        count_reject=count_reject+len(fx)-len(idxAccept)
            
        if count_reject>N*SliceBlock*0.5:
            print 'count_reject={:d}, count_accept={:d}'.format(count_reject,count)
            #samples=samples[0:count,:]
            
            if count<5:
                samples=[]
            return samples     
    return np.asarray(samples)
        
def acq_slice_sampling_generate_parallel(acq_fun,gp,pbounds,N,y_max,SliceBlock=1):
    """
    A Batch Generalized Slice Sampling technique to draw samples under the acquisition function

    Input Parameters
    ----------
    acq_fun: The acquisition function object that return its point-wise value.
    gp: A gaussian process fitted to the relevant data.
    y_max: The current maximum known value of the target function.
    pbounds: The variables bounds to limit the search of the acq max.
    N: number of points to generate
    SliceBlock: block size for batch slice sampling
    
    Returns
    -------
    samples
    """
    
    #from multiprocessing import Process, Pipe, Pool
    from itertools import izip    
    
    if isinstance(pbounds,dict):
        # Get the name of the parameters
        #keys = list(pbounds.keys())
    
        bounds = []
        for key in pbounds.keys():
            bounds.append(pbounds[key])
        bounds = np.asarray(bounds)
    else:
        bounds=np.asarray(pbounds)
    
    D=bounds.size/2
    D=np.int(D)
    
    N=N*D       

    # find minimum value in acquisition function
    x0=np.zeros(D)
    for idx in range(0,D):
        x0[idx]=np.random.uniform(bounds[idx,0],bounds[idx,1],1)
        
    #res = minimize(lambda x: acq_fun(x.reshape(1, -1), gp=gp, y_max=y_max),
                   #x_try.reshape(1, -1),bounds=bounds,method="L-BFGS-B")
    res = minimize(acq_fun,x0.reshape(1, -1),args=(gp,y_max),bounds=bounds,method="L-BFGS-B")
    
    y_min=res.fun
    
    n_procs=2
    pool = Pool(n_procs)

    def fun_parallel(num):
        return local_slice_sampling(acq_fun,gp,N,D,bounds,SliceBlock,y_min,y_max)

    #mysamples = pool.map(fun_parallel, range(1,3))
    results = [pool.apply_async(fun_parallel, args=(x)) for x in range(1,3)]
    mysamples = [p.get() for p in results]
    #mysamples=fun_parallel(1)  


    return mysamples