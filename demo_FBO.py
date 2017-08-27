from prada_bayes_opt import PradaBayOptFBO
import numpy as np
import matplotlib.pyplot as plt
from prada_bayes_opt import visualization
import random
from prada_bayes_opt import functions
import warnings

import sys

warnings.filterwarnings("ignore")


random.seed(6789)

  
#myfunction=functions.alpine1(input_dim=1)


#myfunction=functions.sincos()
#myfunction=functions.forrester()
#myfunction=functions.saddlepoint()
myfunction=functions.branin()
#myfunction=functions.sixhumpcamel()
#myfunction=functions.dropwave()

func=myfunction.func

    
if myfunction.input_dim<=2:   
    visualization.plot_original_function(myfunction)


gp_params = {'theta':1,'noise_delta':0.1,'MaxIter':10*myfunction.input_dim}

# create an empty object for BO
acq_func={}
acq_func['name']='ei'
acq_func['kappa']=2
acq_func['dim']=myfunction.input_dim

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

b_init_lower=np.asarray([0,20])

b_init_upper=np.asarray([15,50])


b_limit_lower=np.asarray([-50,-70])
b_limit_upper=bounds0[:,1]+5*gap[1]

        
    
bo=PradaBayOptFBO(gp_params,f=func, b_init_lower=b_init_lower,b_init_upper=b_init_upper
     ,b_limit_lower=b_limit_lower, b_limit_upper=b_limit_upper, acq=acq_func,opt_toolbox='scipy')

bo.init(gp_params,n_init_points=1*myfunction.input_dim)

print "lower_bound=",
print bo.bounds[:,0]
print "upper_bound",
print bo.bounds[:,1]
    
# number of iterations
NN=5*myfunction.input_dim


for index in range(0,NN):
    bo.run_FBO(gp_params)

    myfunction.bounds=bo.bounds.copy()

    if bo.stop_flag==1:
        break
    
    print "Iteration={:d}".format(index)

    print "lower_bound=",
    print bo.bounds[:,0]
    print "upper_bound",
    print bo.bounds[:,1]

    sys.stdout.flush()

    if myfunction.input_dim<=2:
        visualization.plot_bo_2d_FBO(bo,myfunction)
        

    print 'recommended x=',
    print bo.X_original[-1]
    
    if myfunction.ismax==1:
        print 'current y={:.2f}, ymax={:.2f}'.format(bo.Y_original[-1],bo.Y_original.max())
    else:
        print 'current y={:.2f}, ymin={:.2f}'.format(myfunction.ismax*bo.Y_original[-1],myfunction.ismax*bo.Y_original.max())
    sys.stdout.flush()

fig=plt.figure(figsize=(6, 3))
myYbest=[bo.Y_original[:idx+1].max()*-1 for idx,val in enumerate(bo.Y_original)]
plt.plot(xrange(len(myYbest)),myYbest,linewidth=2,color='m',linestyle='-',marker='o')
plt.xlabel('Iteration',fontsize=14)
plt.ylabel('Best-found-value',fontsize=14)
plt.title('FBO Performance',fontsize=14)



