# -*- coding: utf-8 -*-
"""
Created on Thu Sep 01 14:49:35 2016

@author: tvun
"""
import numpy          as np
import warnings
import time
import spearmint
import os
import pickle

from spearmint.models.gp import GP
from spearmint.utils import parsing # for testing
#from spearmint.acquisition_functions.predictive_entropy_search import global_optimization_of_GP_approximation
from spearmint.acquisition_functions.abstract_acquisition_function import AbstractAcquisitionFunction

from scipy.optimize import minimize
import numpy.random as random


def acq_max_nlopt_pes(ac,bounds):
    import nlopt
    
    def objective(x, grad):
            """Objective function in the form required by nlopt."""
            #print "=================================="
            if grad.size > 0:
                fx, gx = ac(x[None], grad=True)
                grad[:] = gx[0][:]
            else:
                try:
                    fx = ac(x)
                    if isinstance(fx,list):
                        fx=fx[0]
                except:
                    return 0
            return fx[0]
            
    tol=1e-6
    bounds = np.array(bounds, ndmin=2)

    dim=bounds.shape[0]
    opt = nlopt.opt(nlopt.GN_DIRECT, dim)

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
    
    try:
        xoptimal = opt.optimize(xinit.copy())

    except:
        xoptimal=xinit
     
    fmax= opt.last_optimum_value()
    
    code=opt.last_optimize_result()
    status=1

    if code<4:
        #print "nlopt code = {:d}".format(code)
        status=0

    return xoptimal, fmax, status
    
def acq_max_scipy_pes(ac, bounds):
    x_max = bounds[:, 0]
    max_acq = None

    # multi start
    x_tries = np.random.uniform(bounds[:, 0], bounds[:, 1],size=(20, bounds.shape[0]))

    myopts ={'maxiter':200}

    for x_try in x_tries:
        # Find the minimum of minus the acquisition function
        res = minimize(lambda x: -ac(x.reshape(1, -1)),x_try.reshape(1, -1),bounds=bounds,options=myopts,method="L-BFGS-B")#L-BFGS-B
                       
        # Store it if better than previous minimum(maximum).
        if max_acq is None or -res.fun >= max_acq:
            x_max = res.x
            max_acq = -res.fun

    # Clip output to make sure it lies within the bounds. Due to floating
    # point technicalities this is not always the case.
    return np.clip(x_max, bounds[:, 0], bounds[:, 1])
    
    
def run_experiment(myfunction,n_init=3,NN=10):
    
    time_opt=[]
    
    D = len(myfunction.bounds)

    # Create an array with parameters bounds
    if isinstance(myfunction.bounds,dict):  
        bounds = []
        for key in myfunction.bounds.keys():
            bounds.append(myfunction.bounds[key])
        bounds = np.asarray(bounds)
    else:
        bounds=np.asarray(myfunction.bounds)
    
    start_time = time.time()

    # create a scalebounds 0-1
    scalebounds=np.array([np.zeros(D), np.ones(D)])
    scalebounds=scalebounds.T
    max_min_gap=bounds[:,1]-bounds[:,0]
        
    # Generate random points
    l = [np.random.uniform(x[0], x[1], size=n_init) for x in bounds]
    temp=np.asarray(l).T
    init_X=list(temp.reshape((n_init,-1)))   
    X_original = np.asarray(init_X)
    
    # Evaluate target function at all initialization           
    y_init=myfunction.func(init_X)
    y_init=np.reshape(y_init,(n_init,1))

    Y_original = np.asarray(y_init)        
    Y=(Y_original-np.mean(Y_original))/(np.max(Y_original)-np.min(Y_original))
    Y=Y.ravel()

    # convert it to scaleX
    temp_init_point=np.divide((init_X-bounds[:,0]),max_min_gap)
    
    X = np.asarray(temp_init_point)
    
    
    
    # the noise is 1e-6, the stability jitter is 1e-10
    STABILITY_JITTER = 1e-10

    cfg = parsing.parse_config({'mcmc_iters':0, 
        'acquisition':'PES',
        'likelihood':'gaussian', 
        'kernel':"SquaredExp", 
        'stability_jitter':STABILITY_JITTER,
        'initial_noise':1e-6})['tasks'].values()[0]
        

    
    # number of recommended parameters
    for index in range(0,NN-1):

        # Set acquisition function
        start_opt=time.time()
        
        objective = GP(D, **cfg)
        objective.fit(X, Y, fit_hypers=False)
        constraint = GP(D, **cfg) # will not use it
        
    
        constraint.fit(X, Y, fit_hypers=False)
        cons = {'c1':constraint}
        acq_class = spearmint.acquisition_functions.PES(D)
        acq_f = acq_class.create_acquisition_function({'obj':objective},cons)
    
    
        funs={}
        funs['objective']=acq_f
        funs['constraints']=[] 
        
        #x_max,f_max,status=acq_max_nlopt_pes(acq_f,bounds)
        #if status==0:# if nlopt fails, let try scipy
        x_max=acq_max_scipy_pes(acq_f, scalebounds) # xt

        # record the optimization time
        finished_opt=time.time()
        elapse_opt=finished_opt-start_opt
        time_opt=np.hstack((time_opt,elapse_opt))
        
        # Test if x_max is repeated, if it is, draw another one at random
        if np.any((X - x_max).sum(axis=1) == 0):

            x_max = np.random.uniform(scalebounds[:, 0],
                                      scalebounds[:, 1],
                                      size=scalebounds.shape[0])
                                     
        # store X                                     
        X = np.vstack((X, x_max.reshape((1, -1))))

        # compute X in original scale
        temp_X_new_original=x_max*max_min_gap+bounds[:,0]
        X_original=np.vstack((X_original, temp_X_new_original))
        # evaluate Y using original X
        
        Y_original = np.append(Y_original, myfunction.func(temp_X_new_original))
        
        # update Y after change Y_original
        Y=(Y_original-np.mean(Y_original))/(np.max(Y_original)-np.min(Y_original))
        Y=Y.ravel()

    y_init=np.max(Y_original[0:n_init])
    #GAP=(y_init-Y.max())*1.0/(y_init-yoptimal)
    GAP=0
    #Regret=[np.abs(val-yoptimal) for idx,val in enumerate(bo.Y)]
    Regret=0
    
    fxoptimal=Y_original
    elapsed_time = time.time() - start_time

    return GAP, fxoptimal, Regret,time_opt

def print_result_pes(myfunction,Score,mybatch_type,acq_type='pes'):
    Regret=Score["Regret"]
    ybest=Score["ybest"]
    GAP=Score["GAP"]
    MyTime=Score["MyTime"]
    toolbox='Spearmint'
    print '{:s} {:d}'.format(myfunction.name,myfunction.input_dim)

    #AveRegret=[np.mean(val) for idx,val in enumerate(Regret)]
    #StdRegret=[np.std(val) for idx,val in enumerate(Regret)]
    

    MaxFx=[val.max() for idx,val in enumerate(ybest)]
        
    print '[{:s} {:s} {:s}] ElapseTime={:.3f}({:.2f})'\
                .format(toolbox,mybatch_type,acq_type,np.mean(MyTime),np.std(MyTime))
    
    
    if myfunction.ismax==1:
        print 'MaxBest={:.4f}({:.2f})'.format(myfunction.ismax*np.mean(MaxFx),np.std(MaxFx))    
    else:
        print 'MinBest={:.4f}({:.2f})'.format(myfunction.ismax*np.mean(MaxFx),np.std(MaxFx))

    
    if 'MyOptTime' in Score:
        MyOptTime=Score["MyOptTime"]

        SumOptTime=np.sum(MyOptTime,axis=1)
        print 'OptTime={:.1f}({:.1f})'.format(np.mean(SumOptTime),np.std(SumOptTime))
        
    out_dir="P:\\03.Research\\05.BayesianOptimization\\PradaBayesianOptimization\\pickle_storage"
    strFile="{:s}_{:d}_{:s}_{:s}.pickle".format(myfunction.name,myfunction.input_dim,mybatch_type,'pes')
    path=os.path.join(out_dir,strFile)
    with open(path, 'w') as f:
        if 'BatchSz' in Score:
            pickle.dump([ybest, Regret, MyTime,BatchSz], f)
        else:
            pickle.dump([ybest, Regret, MyTime], f)