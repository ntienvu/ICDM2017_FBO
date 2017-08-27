# -*- coding: utf-8 -*-
"""
Created on Tue Mar 01 21:37:03 2016

"""
#from prada_bayes_opt import PradaBayOptFn

#from sklearn.gaussian_process import GaussianProcess
#from scipy.stats import norm
#import matplotlib as plt
from mpl_toolkits.mplot3d import Axes3D
from prada_bayes_opt import bayesian_optimization_batch
from prada_bayes_opt import bayesian_optimization_function
import matplotlib.pyplot as plt
from matplotlib import gridspec
#from bayes_opt import PradaBayesianOptimization
import numpy as np
import random
import time
import pickle
import os
import sys

def run_experiment_fbo(bo,gp_params,algorithm_chooser,yoptimal,n_init=10,NN=10):
    # create an empty object for BO
    
    start_time = time.time()
    bo.init(gp_params,n_init_points=n_init)
    
    # number of recommended parameters
    for index in range(0,NN-1):
        if algorithm_chooser=="EI_H" or algorithm_chooser=="EI_Q":
            bo.maximize_unbounded_regularizer(gp_params) 
        if algorithm_chooser=="VolumeDoubling":
            bo.maximize_volume_doubling(gp_params)
        if algorithm_chooser=="VolumeL":
            bo.maximize_expanding_volume_L(gp_params)
        if algorithm_chooser=="VolumeLCropping":
            bo.maximize_expanding_volume_L_Cropping(gp_params)

        if algorithm_chooser=="FBO":
            bo.run_FBO(gp_params)
            
    # evaluation
    # computing GAP G=(y_1 - y_best) / (y_1 - y_opt)
    
    y_init=np.max(bo.Y_original[0:n_init])
    GAP=(y_init-bo.Y.max())*1.0/(y_init-yoptimal)
    
    Regret=[np.abs(val-yoptimal) for idx,val in enumerate(bo.Y)]
    
    if GAP<0.00001:
        GAP=0
    fxoptimal=bo.Y_original
    elapsed_time = time.time() - start_time

    return GAP, fxoptimal, Regret,elapsed_time

def run_experiment_with_init_bound(bo,gp_params,yoptimal,n_init=3,NN=10):
    #    init from the small box and optimize in the big box 
    start_time = time.time()
    bo.init(gp_params,n_init_points=n_init)
    
    # number of recommended parameters
    for index in range(0,NN-1):
        #print index
        bo.maximize(gp_params)
        
    # evaluation
    # computing GAP G=(y_1 - y_best) / (y_1 - y_opt)
    
    y_init=np.max(bo.Y_original[0:n_init])
    GAP=(y_init-bo.Y.max())*1.0/(y_init-yoptimal)
    
    Regret=[np.abs(val-yoptimal) for idx,val in enumerate(bo.Y)]
    
    if GAP<0.00001:
        GAP=0
    fxoptimal=bo.Y_original
    elapsed_time = time.time() - start_time

    return GAP, fxoptimal, Regret,elapsed_time
       
def run_experiment(bo,gp_params,yoptimal,n_init=3,NN=10):
    # create an empty object for BO
    
    start_time = time.time()
    bo.init(gp_params,n_init_points=n_init)
    
    # number of recommended parameters
    for index in range(0,NN-1):
        #print index
        bo.maximize(gp_params)
        
    # evaluation
    # computing GAP G=(y_1 - y_best) / (y_1 - y_opt)
    
    y_init=np.max(bo.Y_original[0:n_init])
    GAP=(y_init-bo.Y.max())*1.0/(y_init-yoptimal)
    
    Regret=[np.abs(val-yoptimal) for idx,val in enumerate(bo.Y)]
    
    if GAP<0.00001:
        GAP=0
    fxoptimal=bo.Y_original
    elapsed_time = time.time() - start_time

    return GAP, fxoptimal, Regret,elapsed_time
    
def run_experiment_batch(bo,gp_params,yoptimal,batch_type='gmm',B=3,n_init=3,NN=10):
    # create an empty object for BO
  
    start_time = time.time()

    # Initialize x, y and find current y_max
    bo.init(n_init_points=n_init)
    
    # number of recommended parameters
    #NN=10
    for index in range(0,NN):# because it has already runned another maximize step above
        #print index
        #bo.maximize(init_points=0,kappa=5,**gp_params)
        if batch_type=='nei':
            bo.maximize_batch_NEI(gp_params,B=B)  
        if batch_type=='cl':
            bo.maximize_batch_CL(gp_params,B=B)  
            #bo.maximize_batch_CL_incremental(gp_params,B=B)  
        if batch_type=="b3o":
            bo.maximize_batch_B3O(gp_params)
        if batch_type=="ps":
            bo.maximize_batch_PS(gp_params)
        if batch_type=="bucb":
            bo.maximize_batch_BUCB(gp_params,B=B) 
            
    # evaluation
    # computing GAP G=(y_1 - y_best) / (y_1 - y_opt)
    y_init=np.max(bo.Y[0:B])
    GAP=(y_init-bo.Y.max())*1.0/(y_init-yoptimal)
    
    
     #find the best value up to the current point
    #current_best_y=bo.Y[0]
    #for idx in range(len(bo.Y)):
        #current_best_y=np.vstack((current_best_y,bo.Y[0:idx+1].max()))
        
    Regret=[np.abs(val-yoptimal) for idx,val in enumerate(bo.Y)]
    #Regret=[np.abs(val-yoptimal) for idx,val in enumerate(current_best_y)] 
    
    elapsed_time = time.time() - start_time

    
    return GAP, Regret,elapsed_time
    
def run_experiment_batch_GPyOpt(bo_gpy,yoptimal,batch_type='lp',B=3,n_init=3,NN=10):
    # batch type: Local Penalize
 
    start_time = time.time()
    
    myinit_points=n_init
    
    # --- Run the optimization
    
    # --- Number of cores to use in the optimization (parallel evaluations of f)
    if bo_gpy.dim<5:                                                          
        B = 3                                                          
    else:
        B = bo_gpy.dim*3                                                          

    # --- Run the optimization                                              
    bo_gpy.run_optimization(max_iter=NN,acqu_optimize_method = 'random',
                            n_inbatch = B,batch_method=batch_type,
                            acqu_optimize_restarts = 5,eps = 1e-6)                            
       
    # evaluation
    # computing GAP G=(y_1 - y_best) / (y_1 - y_opt)
    y_init=np.max(bo_gpy.Y[0:myinit_points])
    GAP=(y_init-bo_gpy.Y.min())*1.0/(y_init-yoptimal)
    
    #find the best value up to the current point
    #current_best_y=bo_gpy.Y[0]
    #for idx in range(len(bo_gpy.Y)):
        #current_best_y=np.vstack((current_best_y,bo_gpy.Y[0:idx+1].min()))
        
    Regret=[np.abs(val-yoptimal) for idx,val in enumerate(bo_gpy.Y)]
    #Regret=[np.abs(val-yoptimal) for idx,val in enumerate(current_best_y)]
  
    fxoptimal=bo_gpy.Y
    
    elapsed_time = time.time() - start_time

    return GAP, fxoptimal, Regret,elapsed_time,bo_gpy.opt_time
    
def run_experiment_batch_Glasses(bo,gp_params,yoptimal,batch_type='nei'):
    # create an empty object for BO
 
    #bo.maximize(init_points=2, n_iter=0, acq='ucb', kappa=5, **gp_params)
    myinit_points=5
    bo.maximize(init_points=myinit_points, n_iter=0, **gp_params)
    
    # number of recommended parameters
    NN=10*bo.dim
    for index in range(0,NN):
        #print index
        #bo.maximize(init_points=0,kappa=5,**gp_params)
        my_n_batch=NN-index
        #print my_n_batch
        if batch_type=='nei':
            bo.maximize_batch_NEI(init_points=0,n_batch=my_n_batch,kappa=5,**gp_params)  
        else:
            bo.maximize_batch_CL(init_points=0,n_batch=my_n_batch,kappa=5,**gp_params)  
            
    # evaluation
    # computing GAP G=(y_1 - y_best) / (y_1 - y_opt)
    y_init=np.max(bo.Y[0:myinit_points])
    GAP=(y_init-bo.Y.max())*1.0/(y_init-yoptimal)
    
    fxoptimal=bo.Y.max()
    
    return GAP, fxoptimal
    
def print_result(bo,myfunction,Score,mybatch_type,acq_type,toolbox='GPyOpt'):
    Regret=Score["Regret"]
    ybest=Score["ybest"]
    GAP=Score["GAP"]
    MyTime=Score["MyTime"]
    
    print '{:s} {:d}'.format(myfunction.name,myfunction.input_dim)

    AveRegret=[np.mean(val) for idx,val in enumerate(Regret)]
    StdRegret=[np.std(val) for idx,val in enumerate(Regret)]
    
    if toolbox=='GPyOpt':
        MaxFx=[val.min() for idx,val in enumerate(ybest)]
    else:
        MaxFx=[val.max() for idx,val in enumerate(ybest)]
        
    print '[{:s} {:s} {:s}] GAP={:.3f}({:.2f}) AvgRegret={:.3f}({:.2f}) ElapseTime={:.3f}({:.2f})'\
                .format(toolbox,mybatch_type,acq_type,np.mean(GAP),np.std(GAP),np.mean(AveRegret),\
                np.std(StdRegret),np.mean(MyTime),np.std(MyTime))
    
    if toolbox=='GPyOpt':
        if myfunction.ismax==1:
            print 'MaxBest={:.4f}({:.2f})'.format(-1*np.mean(MaxFx),np.std(MaxFx))    
        else:
            print 'MinBest={:.4f}({:.2f})'.format(np.mean(MaxFx),np.std(MaxFx))
    else:            
        if myfunction.ismax==1:
            print 'MaxBest={:.4f}({:.2f})'.format(myfunction.ismax*np.mean(MaxFx),np.std(MaxFx))    
        else:
            print 'MinBest={:.4f}({:.2f})'.format(myfunction.ismax*np.mean(MaxFx),np.std(MaxFx))
            
    if 'BatchSz' in Score:
        BatchSz=Score["BatchSz"]
        if toolbox=='GPyOpt':
            print 'BatchSz={:.3f}({:.2f})'.format(np.mean(BatchSz),np.std(BatchSz))
        else:
            SumBatch=np.sum(BatchSz,axis=1)
            print 'BatchSz={:.3f}({:.2f})'.format(np.mean(SumBatch),np.std(SumBatch))
    
    if 'MyOptTime' in Score:
        MyOptTime=Score["MyOptTime"]
        if toolbox=='GPyOpt':
            print 'OptTime={:.1f}({:.1f})'.format(np.mean(MyOptTime),np.std(MyOptTime))
        else:
            SumOptTime=np.sum(MyOptTime,axis=1)
            print 'OptTime={:.1f}({:.1f})'.format(np.mean(SumOptTime),np.std(SumOptTime))
        
    out_dir="P:\\03.Research\\05.BayesianOptimization\\PradaBayesianOptimization\\pickle_storage"
    
    if acq_type['name']=='lei': # if ELI acquisition function, we have extra variable 'k'
        strFile="{:s}_{:d}_{:s}_{:s}_c_{:f}.pickle".format(myfunction.name,myfunction.input_dim,
                                                    mybatch_type,acq_type['name'],acq_type['k'])
    else:
        strFile="{:s}_{:d}_{:s}_{:s}.pickle".format(myfunction.name,myfunction.input_dim,mybatch_type,acq_type['name'])
        
    """
    path=os.path.join(out_dir,strFile)
    with open(path, 'w') as f:
        if 'BatchSz' in Score:
            pickle.dump([ybest, Regret, MyTime,BatchSz,bo.bounds], f)
        else:
            pickle.dump([ybest, Regret, MyTime,bo.bounds], f)
            
    """

def print_result_bubo(bo,myfunction,Score,mybatch_type,acq_type,alg_type,toolbox='GPyOpt'):
    Regret=Score["Regret"]
    ybest=Score["ybest"]
    GAP=Score["GAP"]
    MyTime=Score["MyTime"]
    
    print '{:s} {:d}'.format(myfunction.name,myfunction.input_dim)

    AveRegret=[np.mean(val) for idx,val in enumerate(Regret)]
    StdRegret=[np.std(val) for idx,val in enumerate(Regret)]
    
    if toolbox=='GPyOpt':
        MaxFx=[val.min() for idx,val in enumerate(ybest)]
    else:
        MaxFx=[val.max() for idx,val in enumerate(ybest)]
        
    print '[{:s} {:s} {:s} {:s}] GAP={:.3f}({:.2f}) AvgRegret={:.3f}({:.2f}) ElapseTime={:.3f}({:.2f})'\
                .format(toolbox,mybatch_type,acq_type,alg_type,np.mean(GAP),np.std(GAP),np.mean(AveRegret),\
                np.std(StdRegret),np.mean(MyTime),np.std(MyTime))
    
    if toolbox=='GPyOpt':
        if myfunction.ismax==1:
            print 'MaxBest={:.4f}({:.2f})'.format(-1*np.mean(MaxFx),np.std(MaxFx))    
        else:
            print 'MinBest={:.4f}({:.2f})'.format(np.mean(MaxFx),np.std(MaxFx))
    else:            
        if myfunction.ismax==1:
            print 'MaxBest={:.4f}({:.3f})'.format(myfunction.ismax*np.mean(MaxFx),np.std(MaxFx))    
        else:
            print 'MinBest={:.4f}({:.3f})'.format(myfunction.ismax*np.mean(MaxFx),np.std(MaxFx))
            
    if 'BatchSz' in Score:
        BatchSz=Score["BatchSz"]
        if toolbox=='GPyOpt':
            print 'BatchSz={:.3f}({:.2f})'.format(np.mean(BatchSz),np.std(BatchSz))
        else:
            SumBatch=np.sum(BatchSz,axis=1)
            print 'BatchSz={:.3f}({:.2f})'.format(np.mean(SumBatch),np.std(SumBatch))
    
    if 'MyOptTime' in Score:
        MyOptTime=Score["MyOptTime"]
        if toolbox=='GPyOpt':
            print 'OptTime={:.1f}({:.1f})'.format(np.mean(MyOptTime),np.std(MyOptTime))
        else:
            SumOptTime=np.sum(MyOptTime,axis=1)
            print 'OptTime={:.1f}({:.1f})'.format(np.mean(SumOptTime),np.std(SumOptTime))
    
    print    "lower bound",
    print bo.bounds[:,0]
    print "upper bound",
    print bo.bounds[:,1]
    
    print    "init lower bound",
    print bo.b_init_lower
    print "init upper bound",
    print bo.b_init_upper

    print    "limit lower bound",
    print bo.b_limit_lower
    print "limit upper bound",
    print bo.b_limit_upper
    
    out_dir="P:\\03.Research\\05.BayesianOptimization\\PradaBayesianOptimization\\pickle_storage"
    
    if acq_type['name']=='lei': # if ELI acquisition function, we have extra variable 'k'
        strFile="{:s}_{:d}_{:s}_{:s}_c_{:f}.pickle".format(myfunction.name,myfunction.input_dim,
                                                    mybatch_type,acq_type['name'],acq_type['k'],alg_type)
    else:
        strFile="{:s}_{:d}_{:s}_{:s}_{:s}.pickle".format(myfunction.name,myfunction.input_dim,mybatch_type,acq_type['name'],alg_type)
        
    """
    path=os.path.join(out_dir,strFile)
    with open(path, 'w') as f:
        if 'BatchSz' in Score:
            pickle.dump([ybest, Regret, MyTime,BatchSz,bo.bounds,bo.b_init_lower,bo.b_init_upper
            ,bo.b_limit_lower,bo.b_limit_upper], f)
        else:
            pickle.dump([ybest, Regret, MyTime,bo.bounds,bo.b_init_lower,bo.b_init_upper
            ,bo.b_limit_lower,bo.b_limit_upper], f)
    """
            
            
def yBest_Iteration(YY,BatchSzArray,IsPradaBO=0,Y_optimal=0):
    
    nRepeat=len(YY)
    YY=np.asarray(YY)
    ##YY_mean=np.mean(YY,axis=0)
    #YY_std=np.std(YY,axis=0)
    
    mean_TT=[]
    #temp_std=np.std(YY[:,0:BatchSzArray[0]+1])
    #temp_std=np.std(YY_mean[0:BatchSzArray[0]+1])
    
    mean_cum_TT=[]
    
    for idxtt,tt in enumerate(range(0,nRepeat)): # TT run
    
        
        if IsPradaBO==1:
            temp_mean=YY[idxtt,0:BatchSzArray[0]+1].max()
        else:
            temp_mean=YY[idxtt,0:BatchSzArray[0]+1].min()
        
        
        #temp_mean=np.median(YY[idxtt,0:BatchSzArray[0]+1])
        
        temp_mean_cum=YY[idxtt,0:BatchSzArray[0]+1].mean()

        start_point=0
        for idx,bz in enumerate(BatchSzArray): # batch
            if idx==len(BatchSzArray)-1:
                break
            bz=np.int(bz)

            # find maximum in each batch            
            if IsPradaBO==1:
                temp_mean=np.vstack((temp_mean,YY[idxtt,start_point:start_point+bz+1].max()))
            else:
                temp_mean=np.vstack((temp_mean,YY[idxtt,start_point:start_point+bz+1].min()))

            #    get the average in this batch
            temp_mean_cum=np.vstack((temp_mean_cum,YY[idxtt,start_point:start_point+bz+1].mean()))
            
            start_point=start_point+bz

        if IsPradaBO==1:
            myYbest=[temp_mean[:idx+1].max()*-1 for idx,val in enumerate(temp_mean)]
        else:
            myYbest=[temp_mean[:idx+1].min() for idx,val in enumerate(temp_mean)]

        # cumulative regret  
        myYbest_cum=[np.mean(np.abs(temp_mean_cum[BatchSzArray[0]:idx+1]-Y_optimal)) for idx,val in enumerate(temp_mean_cum)]
            
        if len(mean_TT)==0:
            mean_TT=myYbest
            mean_cum_TT=myYbest_cum
        else:
            #mean_TT.append(temp_mean)
            mean_TT=np.vstack((mean_TT,myYbest))
            mean_cum_TT=np.vstack((mean_cum_TT,myYbest_cum))
            
    mean_TT    =np.array(mean_TT)
    std_TT=np.std(mean_TT,axis=0)
    std_TT=np.array(std_TT).ravel()
    mean_TT=np.mean(mean_TT,axis=0)

    
    mean_cum_TT=np.array(mean_cum_TT)   
    std_cum_TT=np.std(mean_cum_TT,axis=0)
    std_cum_TT=np.array(std_cum_TT).ravel()
    mean_cum_TT=np.mean(mean_cum_TT,axis=0)
   
    return mean_TT[::5],std_TT[::5]#,mean_cum_TT[::5],std_cum_TT[::5]