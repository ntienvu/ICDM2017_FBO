import sys
sys.path.insert(0,'..')
from prada_bayes_opt import PradaBayOptFBO
import numpy as np
from prada_bayes_opt import auxiliary_functions
from prada_bayes_opt import functions

import random
import warnings
import itertools
warnings.filterwarnings("ignore")
#%matplotlib inline.

random.seed(6789)


# define a function f
def synthetic(x):
    return np.exp(-(x - 4)**2) + np.exp(-(x - 6)**2/10) + 1/ (x**2 + 1)

def sincos(x):
    return x*np.sin(x)+x*np.cos(2*x)
    

myfunction_list=[]

myfunction_list.append(functions.rosenbrock())
myfunction_list.append(functions.branin())
myfunction_list.append(functions.dropwave())
myfunction_list.append(functions.goldstein())
myfunction_list.append(functions.hartman_3d())

myfunction_list.append(functions.ackley(input_dim=5))
myfunction_list.append(functions.alpine1(input_dim=5))
myfunction_list.append(functions.alpine2(input_dim=5))
myfunction_list.append(functions.hartman_6d())
myfunction_list.append(functions.alpine2(input_dim=10))
myfunction_list.append(functions.gSobol(a=np.array([1,1,1,1,1,1,1,1,1,1])))
  
algorithm_list=[]

algorithm_list.append('FBO')
algorithm_list.append('VolumeDoubling')
algorithm_list.append('EI_H')
algorithm_list.append('EI_Q')


mybatch_type_list={'Single'}

for idx, (myfunction,alg_type,mybatch_type,) in enumerate(itertools.product(myfunction_list,algorithm_list,mybatch_type_list)):
    func=myfunction.func
    mybound=myfunction.bounds
    gp_params = {'theta':myfunction.input_dim,'noise_delta':0.1,'MaxIter':10*myfunction.input_dim}

    yoptimal=myfunction.fmin*myfunction.ismax
    
    nRepeat=20

    GAP=[0]*nRepeat
    ybest=[0]*nRepeat
    Regret=[0]*nRepeat
    MyTime=[0]*nRepeat
    MyOptTime=[0]*nRepeat

    acq_func={}
    acq_func['dim']=myfunction.input_dim

    if alg_type=="FBO" or alg_type=="VolumeL" or alg_type=="VolumeLCropping" or alg_type=="VolumeDoubling":
        #acq_func['name']='ei'
        acq_func['name']='ucb'
        acq_func['kappa']=2
    if alg_type=="EI_H":
        acq_func['name']='ei_H'
    if alg_type=="EI_Q":
        acq_func['name']='ei_Q'

    # Create an array with parameters bounds
    if isinstance(myfunction.bounds,dict):
        # Get the name of the parameters    
        bounds0 = []
        for key in myfunction.bounds.keys():
            bounds0.append(myfunction.bounds[key])
        bounds0 = np.asarray(bounds0)
    else:
        bounds0=np.asarray(myfunction.bounds)
            
    gap=bounds0[:,1]-bounds0[:,0]

    b_init_lower=bounds0[:,0]+0.1*gap
    b_init_upper=bounds0[:,1]+0.3*gap
    
    #bounds_limit_lower=np.zeros(myfunction.input_dim)+0.000001
    b_limit_lower=bounds0[:,0]-10*gap
    b_limit_upper=bounds0[:,1]+10*gap
    
    for ii in range(nRepeat):
        bo=PradaBayOptFBO(gp_params,f=func, b_init_lower=b_init_lower, b_init_upper=b_init_upper,b_limit_lower=b_limit_lower,
                                  b_limit_upper=b_limit_upper, acq=acq_func,opt_toolbox='scipy')

        GAP[ii],ybest[ii],Regret[ii],MyTime[ii]=auxiliary_functions.run_experiment_fbo(bo,gp_params,alg_type,
                                                yoptimal,n_init=3*myfunction.input_dim,NN=10*myfunction.input_dim)
                                                
        MyOptTime[ii]=bo.time_opt

        
    Score={}
    Score["GAP"]=GAP
    Score["ybest"]=ybest
    Score["Regret"]=Regret
    Score["MyTime"]=MyTime
    Score["MyOptTime"]=MyOptTime
    auxiliary_functions.print_result_bubo(bo,myfunction,Score,mybatch_type,acq_func,alg_type,toolbox='BatchBO')        