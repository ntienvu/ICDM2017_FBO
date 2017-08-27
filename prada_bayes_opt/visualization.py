# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 23:22:32 2016

@author: Vu
"""

from __future__ import division
import numpy as np
#import mayavi.mlab as mlab
#from scipy.stats import norm
#import matplotlib as plt
from mpl_toolkits.mplot3d import Axes3D
from prada_bayes_opt import PradaBayOptFn
#from prada_bayes_opt import PradaBayOptBatch
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib import gridspec
import random
from acquisition_functions import AcquisitionFunction, unique_rows
import os
from pylab import *

cdict = {'red': ((0.0, 0.0, 0.0),
                  (0.5, 1.0, 0.7),
                  (1.0, 1.0, 1.0)),
          'green': ((0.0, 0.0, 0.0),
                    (0.5, 1.0, 0.0),
                    (1.0, 1.0, 1.0)),
          'blue': ((0.0, 0.0, 0.0),
                   (0.5, 1.0, 0.0),
                   (1.0, 0.5, 1.0))}

#my_cmap = matplotlib.colors.LinearSegmentedColormap('my_colormap',cdict,256)
#my_cmap = plt.get_cmap('cubehelix')
my_cmap = plt.get_cmap('Blues')

        
counter = 0

#class Visualization(object):
    
    #def __init__(self,bo):
       #self.plot_gp=0     
       #self.posterior=0
       #self.myBo=bo
       
        
def plot_bo(bo):
    if bo.dim==1:
        plot_bo_1d(bo)
    if bo.dim==2:
        plot_bo_2d(bo)
    
def plot_histogram(bo,samples):
    if bo.dim==1:
        plot_histogram_1d(bo,samples)
    if bo.dim==2:
        plot_histogram_2d(bo,samples)

def plot_mixturemodel(g,bo,samples):
    if bo.dim==1:
        plot_mixturemodel_1d(g,bo,samples)
    if bo.dim==2:
        plot_mixturemodel_2d(g,bo,samples)

def plot_mixturemodel_1d(g,bo,samples):
    samples_original=samples*bo.max_min_gap+bo.bounds[:,0]

    x_plot = np.linspace(np.min(samples), np.max(samples), len(samples))
    x_plot = np.reshape(x_plot,(len(samples),-1))
    y_plot = g.score_samples(x_plot)[0]
    
    x_plot_ori = np.linspace(np.min(samples_original), np.max(samples_original), len(samples_original))
    x_plot_ori=np.reshape(x_plot_ori,(len(samples_original),-1))
    
    
    fig=plt.figure(figsize=(8, 3))

    plt.plot(x_plot_ori, np.exp(y_plot), color='red')
    plt.xlim(bo.bounds[0,0],bo.bounds[0,1])
    plt.xlabel("X",fontdict={'size':16})
    plt.ylabel("f(X)",fontdict={'size':16})
    plt.title("IGMM Approximation",fontsize=16)
        
def plot_mixturemodel_2d(dpgmm,bo,samples):
    
    samples_original=samples*bo.max_min_gap+bo.bounds[:,0]
    dpgmm_means_original=dpgmm.truncated_means_*bo.max_min_gap+bo.bounds[:,0]

    #fig=plt.figure(figsize=(12, 5))
    fig=plt.figure()
    myGmm=fig.add_subplot(1,1,1)  

    x1 = np.linspace(bo.scalebounds[0,0],bo.scalebounds[0,1], 100)
    x2 = np.linspace(bo.scalebounds[1,0],bo.scalebounds[1,1], 100)
    
    x1g,x2g=np.meshgrid(x1,x2)
    
    x_plot=np.c_[x1g.flatten(), x2g.flatten()]
    
    y_plot2 = dpgmm.score_samples(x_plot)[0]
    y_plot2=np.exp(y_plot2)
    #y_label=dpgmm.predict(x_plot)[0]
    
    x1_ori = np.linspace(bo.bounds[0,0],bo.bounds[0,1], 100)
    x2_ori = np.linspace(bo.bounds[1,0],bo.bounds[1,1], 100)
    x1g_ori,x2g_ori=np.meshgrid(x1_ori,x2_ori)

    CS_acq=myGmm.contourf(x1g_ori,x2g_ori,y_plot2.reshape(x1g.shape),cmap=plt.cm.bone,origin='lower')
    CS2_acq = plt.contour(CS_acq, levels=CS_acq.levels[::2],colors='r',origin='lower',hold='on')
    
    myGmm.scatter(dpgmm_means_original[:,0],dpgmm_means_original[:,1], marker='*',label=u'Estimated Peaks by IGMM', s=100,color='green')    


    myGmm.set_title('IGMM Approximation',fontsize=16)
    myGmm.set_xlim(bo.bounds[0,0],bo.bounds[0,1])
    myGmm.set_ylim(bo.bounds[1,0],bo.bounds[1,1])
    myGmm.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)


def plot_histogram_2d(bo,samples):
    
    # convert samples from 0-1 to original scale
    samples_original=samples*bo.max_min_gap+bo.bounds[:,0]
    
    #fig=plt.figure(figsize=(12, 5))
    fig=plt.figure()
    myhist=fig.add_subplot(1,1,1)
    
    myhist.set_title("Histogram of Samples under Acq Func",fontsize=16)
    
    #xedges = np.linspace(myfunction.bounds['x1'][0], myfunction.bounds['x1'][1], 10)
    #yedges = np.linspace(myfunction.bounds['x2'][0], myfunction.bounds['x2'][1], 10)
    
    xedges = np.linspace(bo.bounds[0,0], bo.bounds[0,1], 10)
    yedges = np.linspace(bo.bounds[1,0], bo.bounds[1,1], 10)

    H, xedges, yedges = np.histogram2d(samples_original[:,0], samples_original[:,1], bins=50)   
    
    #data = [go.Histogram2d(x=vu[:,1],y=vu[:,0])]
    #plot_url = py.plot(data, filename='2d-histogram')

    # H needs to be rotated and flipped
    H = np.rot90(H)
    H = np.flipud(H)
     
    # Mask zeros
    Hmasked = np.ma.masked_where(H==0,H) # Mask pixels with a value of zero
     
    # Plot 2D histogram using pcolor
    myhist.pcolormesh(xedges,yedges,Hmasked)
    myhist.set_xlim(bo.bounds[0,0], bo.bounds[0,1])
    myhist.set_ylim(bo.bounds[1,0], bo.bounds[1,1])

def plot_histogram_1d(bo,samples):
    samples_original=samples*bo.max_min_gap+bo.bounds[:,0]


    fig=plt.figure(figsize=(8, 3))
    fig.suptitle("Histogram",fontsize=16)
    myplot=fig.add_subplot(111)
    myplot.hist(samples_original,50)
    myplot.set_xlim(bo.bounds[0,0],bo.bounds[0,1])
    
    myplot.set_xlabel("Value",fontsize=16)
    myplot.set_ylabel("Frequency",fontsize=16)
        
    
def plot_bo_1d(bo):
    func=bo.f
    #x_original = np.linspace(bo.bounds[0,0], bo.bounds[0,1], 100)
    x = np.linspace(bo.scalebounds[0,0], bo.scalebounds[0,1], 1000)
    x_original=x*bo.max_min_gap+bo.bounds[:,0]

    y_original = func(x_original)
    #y = func(x)
    #y_original=mu*(np.max(bo.Y_original)-np.min(bo.Y_original))+np.mean(bo.Y_original)

    
    fig=plt.figure(figsize=(8, 5))
    fig.suptitle('Gaussian Process and Utility Function After {} Points'.format(len(bo.X)), fontdict={'size':18})
    
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1]) 
    axis = plt.subplot(gs[0])
    acq = plt.subplot(gs[1])
    
    mu, sigma = bo.posterior(x)
    #mu_original=mu*(np.max(y_original)-np.min(y_original))+np.mean(y_original)
    mu_original=mu*(np.max(bo.Y_original)-np.min(bo.Y_original))+np.mean(bo.Y_original)
    sigma_original=sigma*(np.max(bo.Y_original)-np.min(bo.Y_original))+np.mean(bo.Y_original)
    
    axis.plot(x_original, y_original, linewidth=3, label='Real Function')
    axis.plot(bo.X_original.flatten(), bo.Y_original, 'D', markersize=8, label=u'Observations', color='r')
    axis.plot(x_original, mu_original, '--', color='k', label='GP mean')
    
    #samples*bo.max_min_gap+bo.bounds[:,0]
    
    temp_xaxis=np.concatenate([x_original, x_original[::-1]])
    #temp_xaxis=temp*bo.max_min_gap+bo.bounds[:,0]
    
    #temp_yaxis_original=np.concatenate([mu_original - 1.9600 * sigma_original, (mu_original + 1.9600 * sigma_original)[::-1]])
    temp_yaxis=np.concatenate([mu - 1.9600 * sigma, (mu + 1.9600 * sigma)[::-1]])
    temp_yaxis_original=temp_yaxis*(np.max(bo.Y_original)-np.min(bo.Y_original))+np.mean(bo.Y_original)
    axis.fill(temp_xaxis, temp_yaxis_original,alpha=.6, fc='c', ec='None', label='95% CI')
    
    axis.set_xlim((np.min(x_original), np.max(x_original)))
    #axis.set_ylim((None, None))
    axis.set_ylabel('f(x)', fontdict={'size':16})
    axis.set_xlabel('x', fontdict={'size':16})

    utility = bo.acq_func.acq_kind(x.reshape((-1, 1)), bo.gp, np.max(bo.Y))
    acq.plot(x_original, utility, label='Utility Function', color='purple')
    acq.plot(x_original[np.argmax(utility)], np.max(utility), '*', markersize=15, 
             label=u'Next Best Guess', markerfacecolor='gold', markeredgecolor='k', markeredgewidth=1)
             
    # check batch BO     
    try:
        nSelectedPoints=np.int(bo.NumPoints[-1])
    except:
        nSelectedPoints=1
    max_point=np.max(utility)
    
    acq.plot(bo.X_original[-nSelectedPoints:], max_point.repeat(nSelectedPoints), 'v', markersize=15, 
         label=u'Previous Selection', markerfacecolor='green', markeredgecolor='k', markeredgewidth=1)
             
    acq.set_xlim((np.min(x_original), np.max(x_original)))
    #acq.set_ylim((0, np.max(utility) + 0.5))
    acq.set_ylim((np.min(utility), 1.1*np.max(utility)))
    acq.set_ylabel('Acq', fontdict={'size':16})
    acq.set_xlabel('x', fontdict={'size':16})
    
    axis.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    acq.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)

def plot_bo_2d(bo):
    
    x1 = np.linspace(bo.scalebounds[0,0], bo.scalebounds[0,1], 100)
    x2 = np.linspace(bo.scalebounds[1,0], bo.scalebounds[1,1], 100)
    
    x1g,x2g=np.meshgrid(x1,x2)
    
    X=np.c_[x1g.flatten(), x2g.flatten()]
    
    x1_ori = np.linspace(bo.bounds[0,0], bo.bounds[0,1], 100)
    x2_ori = np.linspace(bo.bounds[1,0], bo.bounds[1,1], 100)
    
    x1g_ori,x2g_ori=np.meshgrid(x1_ori,x2_ori)
    
    X_ori=np.c_[x1g_ori.flatten(), x2g_ori.flatten()]
    
  
    fig = plt.figure()
    
    #axis2d = fig.add_subplot(1, 2, 1)
    acq2d = fig.add_subplot(1, 1, 1)
    
    #mu, sigma = bo.posterior(X)
    # plot the acquisition function

    utility = bo.acq_func.acq_kind(X, bo.gp, np.max(bo.Y))
    #acq3d.plot_surface(x1g,x1g,utility.reshape(x1g.shape))
    
    CS_acq=acq2d.contourf(x1g_ori,x2g_ori,utility.reshape(x1g.shape),cmap=my_cmap,origin='lower')
    CS2_acq = plt.contour(CS_acq, levels=CS_acq.levels[::2],colors='r',origin='lower',hold='on')
    
    idxBest=np.argmax(utility)
    
    #acq2d.scatter(X_ori[idxBest,0],X_ori[idxBest,1],color='b',s=30,label='Current Peak')
    #acq2d.scatter(bo.X_original[:,0],bo.X_original[:,1],color='g',label='Observations')  
    #acq2d.scatter(bo.X_original[-1,0],bo.X_original[-1,1],color='r',s=30,label='Previous Selection')
    acq2d.scatter(bo.X_original[-1,0],bo.X_original[-1,1],marker='*', color='green',s=100,label='Previous Selection')

    acq2d.set_title('Acquisition Function',fontsize=16)
    acq2d.set_xlim(bo.bounds[0,0], bo.bounds[0,1])
    acq2d.set_ylim(bo.bounds[1,0], bo.bounds[1,1])
    
    #acq2d.legend(loc=1, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    #acq2d.legend(loc='center left',bbox_to_anchor=(1.01, 0.5))
      
    fig.colorbar(CS_acq, ax=acq2d, shrink=0.9)

    #acq.set_xlim((np.min(x), np.max(x)))
    #acq.set_ylim((np.min(utility), 1.1*np.max(utility)))
    #acq.set_ylabel('Acq', fontdict={'size':16})
    #acq.set_xlabel('x', fontdict={'size':16})
    
    #axis.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    #acq.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)


def plot_bo_2d_FBO(bo,myfunction):

    global counter
    counter=counter+1
    
    strFolder="P:\\03.Research\\05.BayesianOptimization\\PradaBayesianOptimization\\plot_Nov_2016"

    x1 = np.linspace(bo.scalebounds[0,0], bo.scalebounds[0,1], 100)
    x2 = np.linspace(bo.scalebounds[1,0], bo.scalebounds[1,1], 100)
    
    x1g,x2g=np.meshgrid(x1,x2)
    
    X=np.c_[x1g.flatten(), x2g.flatten()]
    
    x1_ori = np.linspace(bo.bounds[0,0], bo.bounds[0,1], 100)
    x2_ori = np.linspace(bo.bounds[1,0], bo.bounds[1,1], 100)
    
    x1g_ori,x2g_ori=np.meshgrid(x1_ori,x2_ori)
    
    X_ori=np.c_[x1g_ori.flatten(), x2g_ori.flatten()]
        
  
    fig = plt.figure(figsize=(10, 3.5))
    
    #axis2d = fig.add_subplot(1, 2, 1)
    
    # plot invasion set
    acq_expansion = fig.add_subplot(1, 2, 1)

    x1 = np.linspace(bo.b_limit_lower[0], bo.b_limit_upper[0], 100)
    x2 = np.linspace(bo.b_limit_lower[1], bo.b_limit_upper[1], 100)
    x1g_ori_limit,x2g_ori_limit=np.meshgrid(x1,x2)
    X_plot=np.c_[x1g_ori_limit.flatten(), x2g_ori_limit.flatten()]
    Y = myfunction.func(X_plot)
    Y=-np.log(np.abs(Y))
    CS_expansion=acq_expansion.contourf(x1g_ori_limit,x2g_ori_limit,Y.reshape(x1g_ori.shape),cmap=my_cmap,origin='lower')
  
    if len(bo.X_invasion)!=0:
        myinvasion_set=acq_expansion.scatter(bo.X_invasion[:,0],bo.X_invasion[:,1],color='m',s=1,label='Invasion Set')   
    else:
        myinvasion_set=[] 
    
    myrectangle=patches.Rectangle(bo.bounds_bk[:,0], bo.max_min_gap_bk[0],bo.max_min_gap_bk[1],
                                              alpha=0.3, fill=False, facecolor="#00ffff",linewidth=3)
                                              
    acq_expansion.add_patch(myrectangle)
    
    acq_expansion.set_xlim(bo.b_limit_lower[0]-0.2, bo.b_limit_upper[0]+0.2)
    acq_expansion.set_ylim(bo.b_limit_lower[1]-0.2, bo.b_limit_upper[1]+0.2)
    


    if len(bo.X_invasion)!=0:
        acq_expansion.legend([myrectangle,myinvasion_set],[ur'$X_{t-1}$',ur'$I_t$'],loc=4,ncol=1,prop={'size':16},scatterpoints = 5)
        strTitle_Inv="[t={:d}] Invasion Set".format(counter)

        acq_expansion.set_title(strTitle_Inv,fontsize=16)
    else:
        acq_expansion.legend([myrectangle,myinvasion_set],[ur'$X_{t-1}$',ur'Empty $I_t$'],loc=4,ncol=1,prop={'size':16},scatterpoints = 5)
        strTitle_Inv="[t={:d}] Empty Invasion Set".format(counter)
        acq_expansion.set_title(strTitle_Inv,fontsize=16)
   
        
    # plot acquisition function    
    
    acq2d = fig.add_subplot(1, 2, 2)

    utility = bo.acq_func.acq_kind(X, bo.gp, np.max(bo.Y))
    #acq3d.plot_surface(x1g,x1g,utility.reshape(x1g.shape))
    
    CS_acq=acq2d.contourf(x1g_ori,x2g_ori,utility.reshape(x1g.shape),cmap=my_cmap,origin='lower')
    CS2_acq = plt.contour(CS_acq, levels=CS_acq.levels[::2],colors='r',origin='lower',hold='on')
    
    idxBest=np.argmax(utility)
    
    myrectangle=patches.Rectangle(bo.bounds[:,0], bo.max_min_gap[0],bo.max_min_gap[1],
                                  alpha=0.3, fill=False, facecolor="#00ffff",linewidth=3)
                                              
    acq2d.add_patch(myrectangle)
    
    #acq2d.scatter(X_ori[idxBest,0],X_ori[idxBest,1],color='b',s=30,label='Current Peak')
    myobs=acq2d.scatter(bo.X_original[:,0],bo.X_original[:,1],color='g',s=6,label='Data')  
    #acq2d.scatter(bo.X_original[-1,0],bo.X_original[-1,1],color='r',s=30,label='Previous Selection')

    #acq2d.set_xlim(bo.bounds[0,0], bo.bounds[0,1])
    #acq2d.set_ylim(bo.bounds[1,0], bo.bounds[1,1])
    
    

    
    
    acq2d.set_xlim(bo.b_limit_lower[0]-0.2, bo.b_limit_upper[0]+0.2)
    acq2d.set_ylim(bo.b_limit_lower[1]-0.2, bo.b_limit_upper[1]+0.2)
    
    #acq2d.legend(loc=1, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    #acq2d.legend(loc='center left',bbox_to_anchor=(1.2, 0.5))
    #acq2d.legend(loc=4)
    acq2d.legend([myrectangle,myobs],[ur'$X_{t}$','Data'],loc=4,ncol=1,prop={'size':16}, scatterpoints = 3)

    strTitle_Acq="[t={:d}] Acquisition Func".format(counter)
    acq2d.set_title(strTitle_Acq,fontsize=16)

    fig.colorbar(CS_expansion, ax=acq_expansion, shrink=0.9)
    fig.colorbar(CS_acq, ax=acq2d, shrink=0.9)

    #acq.set_xlim((np.min(x), np.max(x)))
    #acq.set_ylim((np.min(utility), 1.1*np.max(utility)))
    #acq.set_ylabel('Acq', fontdict={'size':16})
    #acq.set_xlabel('x', fontdict={'size':16})
    
    #axis.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    #acq.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    
    strFileName="{:d}_bubo.eps".format(counter)
    strPath=os.path.join(strFolder,strFileName)
    #print strPath
    #fig.savefig(strPath, bbox_inches='tight')
    
def plot_bo_2d_withGPmeans(bo):
    
    x1 = np.linspace(bo.scalebounds[0,0], bo.scalebounds[0,1], 100)
    x2 = np.linspace(bo.scalebounds[1,0], bo.scalebounds[1,1], 100)
    
    x1g,x2g=np.meshgrid(x1,x2)
    
    X=np.c_[x1g.flatten(), x2g.flatten()]
    
    x1_ori = np.linspace(bo.bounds[0,0], bo.bounds[0,1], 100)
    x2_ori = np.linspace(bo.bounds[1,0], bo.bounds[1,1], 100)
    
    x1g_ori,x2g_ori=np.meshgrid(x1_ori,x2_ori)
    
    X_ori=np.c_[x1g_ori.flatten(), x2g_ori.flatten()]
    
    #fig.suptitle('Gaussian Process and Utility Function After {} Points'.format(len(bo.X)), fontdict={'size':18})
    
    fig = plt.figure(figsize=(12, 5))
    
    #axis3d = fig.add_subplot(1, 2, 1, projection='3d')
    axis2d = fig.add_subplot(1, 2, 1)
    #acq3d = fig.add_subplot(2, 2, 3, projection='3d')
    acq2d = fig.add_subplot(1, 2, 2)
    
    mu, sigma = bo.posterior(X)
    #axis.plot(x, y, linewidth=3, label='Target')
    #axis3d.plot_surface(x1g,x1g,mu.reshape(x1g.shape))
    #axis3d.scatter(bo.X[:,0],bo.X[:,1], bo.Y,zdir='z',  label=u'Observations', color='r')    

    
    CS=axis2d.contourf(x1g_ori,x2g_ori,mu.reshape(x1g.shape),cmap=plt.cm.bone,origin='lower')
    CS2 = plt.contour(CS, levels=CS.levels[::2],colors='r',origin='lower',hold='on')
    
    axis2d.scatter(bo.X_original[:,0],bo.X_original[:,1], label=u'Observations', color='g')    
    axis2d.set_title('Gaussian Process Mean',fontsize=16)
    axis2d.set_xlim(bo.bounds[0,0], bo.bounds[0,1])
    axis2d.set_ylim(bo.bounds[1,0], bo.bounds[1,1])
    fig.colorbar(CS, ax=axis2d, shrink=0.9)

    #plt.colorbar(ax=axis2d)

    #axis.plot(x, mu, '--', color='k', label='Prediction')
    
    
    #axis.set_xlim((np.min(x), np.max(x)))
    #axis.set_ylim((None, None))
    #axis.set_ylabel('f(x)', fontdict={'size':16})
    #axis.set_xlabel('x', fontdict={'size':16})
    
    # plot the acquisition function

    utility = bo.acq_func.acq_kind(X, bo.gp, np.max(bo.Y))
    #acq3d.plot_surface(x1g,x1g,utility.reshape(x1g.shape))
    
    #CS_acq=acq2d.contourf(x1g_ori,x2g_ori,utility.reshape(x1g.shape),cmap=plt.cm.bone,origin='lower')
    CS_acq=acq2d.contourf(x1g_ori,x2g_ori,utility.reshape(x1g.shape),cmap=my_cmap,origin='lower')
    CS2_acq = plt.contour(CS_acq, levels=CS_acq.levels[::2],colors='r',origin='lower',hold='on')
    
    idxBest=np.argmax(utility)

    
    acq2d.scatter(bo.X_original[:,0],bo.X_original[:,1],color='g')  
    
        
    acq2d.scatter(bo.X_original[-1,0],bo.X_original[-1,1],color='r',s=60)
    acq2d.scatter(X_ori[idxBest,0],X_ori[idxBest,1],color='b',s=60)
    
    
    acq2d.set_title('Acquisition Function',fontsize=16)
    acq2d.set_xlim(bo.bounds[0,0]-0.2, bo.bounds[0,1]+0.2)
    acq2d.set_ylim(bo.bounds[1,0]-0.2, bo.bounds[1,1]+0.2)
             
    #acq.set_xlim((np.min(x), np.max(x)))
    #acq.set_ylim((np.min(utility), 1.1*np.max(utility)))
    #acq.set_ylabel('Acq', fontdict={'size':16})
    #acq.set_xlabel('x', fontdict={'size':16})
    
    #axis.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    #acq.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    
    fig.colorbar(CS_acq, ax=acq2d, shrink=0.9)

def plot_bo_2d_withGPmeans_Sigma(bo):
    
    x1 = np.linspace(bo.scalebounds[0,0], bo.scalebounds[0,1], 100)
    x2 = np.linspace(bo.scalebounds[1,0], bo.scalebounds[1,1], 100)
    
    x1g,x2g=np.meshgrid(x1,x2)
    
    X=np.c_[x1g.flatten(), x2g.flatten()]
    
    x1_ori = np.linspace(bo.bounds[0,0], bo.bounds[0,1], 100)
    x2_ori = np.linspace(bo.bounds[1,0], bo.bounds[1,1], 100)
    
    x1g_ori,x2g_ori=np.meshgrid(x1_ori,x2_ori)
    
    X_ori=np.c_[x1g_ori.flatten(), x2g_ori.flatten()]
    
    #fig.suptitle('Gaussian Process and Utility Function After {} Points'.format(len(bo.X)), fontdict={'size':18})
    
    fig = plt.figure(figsize=(12, 3))
    
    #axis3d = fig.add_subplot(1, 2, 1, projection='3d')
    axis2d = fig.add_subplot(1, 2, 1)
    #acq3d = fig.add_subplot(2, 2, 3, projection='3d')
    acq2d = fig.add_subplot(1, 2, 2)
    
    mu, sigma = bo.posterior(X)
    #axis.plot(x, y, linewidth=3, label='Target')
    #axis3d.plot_surface(x1g,x1g,mu.reshape(x1g.shape))
    #axis3d.scatter(bo.X[:,0],bo.X[:,1], bo.Y,zdir='z',  label=u'Observations', color='r')    

    utility = bo.acq_func.acq_kind(X, bo.gp, np.max(bo.Y))

    CS=axis2d.contourf(x1g_ori,x2g_ori,mu.reshape(x1g.shape),cmap=plt.cm.bone,origin='lower')
    CS2 = plt.contour(CS, levels=CS.levels[::2],colors='r',origin='lower',hold='on')
    
    axis2d.scatter(bo.X_original[:,0],bo.X_original[:,1], label=u'Observations', color='g')    
    axis2d.set_title('Gaussian Process Mean',fontsize=16)
    axis2d.set_xlim(bo.bounds[0,0], bo.bounds[0,1])
    axis2d.set_ylim(bo.bounds[1,0], bo.bounds[1,1])
    fig.colorbar(CS, ax=axis2d, shrink=0.9)

    
    #CS_acq=acq2d.contourf(x1g_ori,x2g_ori,utility.reshape(x1g.shape),cmap=plt.cm.bone,origin='lower')
    CS_acq=acq2d.contourf(x1g_ori,x2g_ori,sigma.reshape(x1g.shape),cmap=my_cmap,origin='lower')
    CS2_acq = plt.contour(CS_acq, levels=CS_acq.levels[::2],colors='r',origin='lower',hold='on')
    
    idxBest=np.argmax(utility)

    
    acq2d.scatter(bo.X_original[:,0],bo.X_original[:,1],color='g')  
    
        
    acq2d.scatter(bo.X_original[-1,0],bo.X_original[-1,1],color='r',s=60)
    acq2d.scatter(X_ori[idxBest,0],X_ori[idxBest,1],color='b',s=60)
    
    
    acq2d.set_title('Gaussian Process Variance',fontsize=16)
    #acq2d.set_xlim(bo.bounds[0,0]-0.2, bo.bounds[0,1]+0.2)
    #acq2d.set_ylim(bo.bounds[1,0]-0.2, bo.bounds[1,1]+0.2)
             
    
    fig.colorbar(CS_acq, ax=acq2d, shrink=0.9)
    
def plot_gp_batch(self,x,y):
    
    bo=self.myBo
    n_batch=bo.NumPoints
    
    fig=plt.figure(figsize=(16, 10))
    fig.suptitle('Gaussian Process and Utility Function After {} Steps'.format(len(bo.X)), fontdict={'size':30})
    
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1]) 
    axis = plt.subplot(gs[0])
    acq = plt.subplot(gs[1])
    
    mu, sigma = posterior(bo)
    axis.plot(x, y, linewidth=3, label='Target')
    axis.plot(bo.X.flatten(), bo.Y, 'D', markersize=8, label=u'Observations', color='r')
    axis.plot(x, mu, '--', color='k', label='GP mean')
    
    axis.fill(np.concatenate([x, x[::-1]]), 
              np.concatenate([mu - 1.9600 * sigma, (mu + 1.9600 * sigma)[::-1]]),
        alpha=.6, fc='c', ec='None', label='95% confidence interval')
    
    axis.set_xlim((-2, 10))
    axis.set_ylim((None, None))
    axis.set_ylabel('f(x)', fontdict={'size':20})
    axis.set_xlabel('x', fontdict={'size':20})
    
    utility = bo.acq_func.acq_kind(x.reshape((-1, 1)), bo.gp, 0)
    acq.plot(x, utility, label='Utility Function', color='purple')
    
    #selected_x=x[np.argmax(utility)]
    #selected_y=np.max(utility)
    
    selected_x=bo.X[-1-n_batch:]
    selected_y=utility(selected_x)
    
    acq.plot(selected_x, selected_y,'*', markersize=15, 
             label=u'Next Best Guess', markerfacecolor='gold', markeredgecolor='k', markeredgewidth=1)
             
    acq.set_xlim((-2, 10))
    acq.set_ylim((0, np.max(utility) + 0.5))
    acq.set_ylabel('Utility', fontdict={'size':20})
    acq.set_xlabel('x', fontdict={'size':20})
    
    axis.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    acq.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)

def plot_original_function(myfunction):
    
    origin = 'lower'

    func=myfunction.func


    if myfunction.input_dim==1:    
        x = np.linspace(myfunction.bounds['x'][0], myfunction.bounds['x'][1], 1000)
        y = func(x)
    
        fig=plt.figure(figsize=(8, 5))
        plt.plot(x, y)
        strTitle="{:s}".format(myfunction.name)

        plt.title(strTitle)
    
    if myfunction.input_dim==2:    
        
        # Create an array with parameters bounds
        if isinstance(myfunction.bounds,dict):
            # Get the name of the parameters        
            bounds = []
            for key in myfunction.bounds.keys():
                bounds.append(myfunction.bounds[key])
            bounds = np.asarray(bounds)
        else:
            bounds=np.asarray(myfunction.bounds)
            
        x1 = np.linspace(bounds[0][0], bounds[0][1], 100)
        x2 = np.linspace(bounds[1][0], bounds[1][1], 100)
        x1g,x2g=np.meshgrid(x1,x2)
        X_plot=np.c_[x1g.flatten(), x2g.flatten()]
        Y = func(X_plot)
    
        #fig=plt.figure(figsize=(8, 5))
        
        #fig = plt.figure(figsize=(12, 3.5))
        fig = plt.figure(figsize=(6, 3.5))
        
        ax3d = fig.add_subplot(1, 1, 1, projection='3d')
        #ax2d = fig.add_subplot(1, 2, 2)
        
        ax3d.plot_surface(x1g,x2g,Y.reshape(x1g.shape),cmap=my_cmap) 
        
        alpha = 30  # degrees
        #mlab.view(azimuth=0, elevation=90, roll=-90+alpha)

        strTitle="{:s}".format(myfunction.name)
        #print strTitle
        ax3d.set_title(strTitle)
        #plt.plot(x, y)
        #CS=ax2d.contourf(x1g,x2g,Y.reshape(x1g.shape),cmap=my_cmap,origin=origin)   
       
        #CS2 = plt.contour(CS, levels=CS.levels[::2],colors='r',origin=origin,hold='on')
        #plt.colorbar(CS2, ax=ax2d, shrink=0.9)

        
        
    strFolder="P:\\03.Research\\05.BayesianOptimization\\PradaBayesianOptimization\\plot_August_2016\\ei_eli"
    strFileName="{:s}.eps".format(myfunction.name)
    strPath=os.path.join(strFolder,strFileName)
    #fig.savefig(strPath, bbox_inches='tight')
        
def plot_bo_multiple_gp_1d(bo):
        
    func=bo.f
    x = np.linspace(bo.scalebounds[0,0], bo.scalebounds[0,1], 1000)
    x_original=x*bo.max_min_gap+bo.bounds[:,0]

    y_original = func(x_original)
    
    fig=plt.figure(figsize=(10, 5))
    fig.suptitle('Gaussian Process and Utility Function After {} Points'.format(len(bo.X)), fontdict={'size':18})
    
    gs = gridspec.GridSpec(3, 1, height_ratios=[3,1,1]) 
    axis = plt.subplot(gs[0])
    acq = plt.subplot(gs[1])
    acq_integrated=plt.subplot(gs[2])
    
    mu, sigma = bo.posterior(x)
    #mu_original=mu*(np.max(bo.Y_original)-np.min(bo.Y_original))+np.mean(bo.Y_original)

    nGP=len(mu)
    
    axis.plot(x_original, y_original, linewidth=3, label='Real Function')
    axis.plot(bo.X_original.flatten(), bo.Y_original, 'D', markersize=8, label=u'Observations', color='r')
    
    for idx in range(nGP):
        
        mu_original=mu[idx]*(np.max(bo.Y_original)-np.min(bo.Y_original))+np.mean(bo.Y_original)

        axis.plot(x_original,mu_original,'--',color = "#%06x" % random.randint(0, 0xFFFFFF),label='GP Theta={:.2f}'.format(bo.theta[idx]),linewidth=2)
        
        temp_xaxis=np.concatenate([x_original, x_original[::-1]])
        temp_yaxis=np.concatenate([mu[idx] - 1.9600 * sigma[idx], (mu[idx] + 1.9600 * sigma[idx])[::-1]])
        temp_yaxis_original=temp_yaxis*(np.max(bo.Y_original)-np.min(bo.Y_original))+np.mean(bo.Y_original)
        
        axis.fill(temp_xaxis, temp_yaxis_original,alpha=.6, fc='c', ec='None', label='95% CI')

    
    #axis.set_xlim((np.min(x), np.max(x)))
    axis.set_ylim((np.min(y_original)*2, np.max(y_original)*2))
    
    axis.set_ylabel('f(x)', fontdict={'size':16})
    axis.set_xlabel('x', fontdict={'size':16})
    
    ## estimate the utility
    utility = bo.acq_func.acq_kind(x.reshape((-1, 1)), bo.gp, bo.Y.max())
    
    for idx in range(nGP):
        acq.plot(x_original, utility[idx], label='Acq Func GP {:.2f}'.format(bo.theta[idx]),
                 color="#%06x" % random.randint(0, 0xFFFFFF),linewidth=2)
                 
                 
        acq.plot(x_original[np.argmax(utility[idx])], np.max(utility[idx]), '*', markersize=15, 
             label=u'Next Guess GP {:.2f}'.format(bo.theta[idx]), markerfacecolor='gold', markeredgecolor='k', markeredgewidth=1)
             
             
    acq.set_xlim((np.min(x_original), np.max(x_original)))
    #acq.set_ylim((0, np.max(utility[0]) + 0.5))
    acq.set_ylabel('Acq', fontdict={'size':16})
    acq.set_xlabel('x', fontdict={'size':16})
    
    axis.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    acq.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    
    
    
    ## estimate the integrated acquisition function
    util_integrated = bo.acq_func.utility_plot(x.reshape((-1, 1)), bo.gp, bo.Y.max())
    
    acq_integrated.plot(x, util_integrated, label='Acq Int-Func GP',
             color="#%06x" % random.randint(0, 0xFFFFFF),linewidth=2)
    acq_integrated.plot(x[np.argmax(util_integrated)], np.max(util_integrated), '*', markersize=15, 
         label=u'Next Guess', markerfacecolor='gold', markeredgecolor='k', markeredgewidth=1)
             
             
    acq_integrated.set_xlim((np.min(x), np.max(x)))
    acq_integrated.set_ylim((0, np.max(util_integrated) + 0.1))
    acq_integrated.set_ylabel('Int-Acq', fontdict={'size':16})
    acq_integrated.set_xlabel('x', fontdict={'size':16})
    
    axis.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    acq.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    acq_integrated.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    
    #===========================================
    
def plot_gp_batch(bo,x,y):
    
    n_batch=bo.NumPoints[-1]
    
    x1 = np.linspace(bo.scalebounds[0,0], bo.scalebounds[0,1], 100)
    x2 = np.linspace(bo.scalebounds[1,0], bo.scalebounds[1,1], 100)
    
    x1g,x2g=np.meshgrid(x1,x2)
    
    X=np.c_[x1g.flatten(), x2g.flatten()]
    
    x1_ori = np.linspace(bo.bounds[0,0], bo.bounds[0,1], 100)
    x2_ori = np.linspace(bo.bounds[1,0], bo.bounds[1,1], 100)
    
    x1g_ori,x2g_ori=np.meshgrid(x1_ori,x2_ori)
    
    X_ori=np.c_[x1g_ori.flatten(), x2g_ori.flatten()]
    
  
    fig = plt.figure()
    
    #axis2d = fig.add_subplot(1, 2, 1)
    acq2d = fig.add_subplot(1, 1, 1)
    
    #mu, sigma = bo.posterior(X)
    # plot the acquisition function

    utility = bo.acq_func.acq_kind(X, bo.gp, np.max(bo.Y))
    #acq3d.plot_surface(x1g,x1g,utility.reshape(x1g.shape))
    
    CS_acq=acq2d.contourf(x1g_ori,x2g_ori,utility.reshape(x1g.shape),cmap=my_cmap,origin='lower')
    CS2_acq = plt.contour(CS_acq, levels=CS_acq.levels[::2],colors='r',origin='lower',hold='on')
    
    idxBest=np.argmax(utility)
    
    #acq2d.scatter(X_ori[idxBest,0],X_ori[idxBest,1],color='b',s=30,label='Current Peak')
    #acq2d.scatter(bo.X_original[:,0],bo.X_original[:,1],color='g',label='Observations')  
    #acq2d.scatter(bo.X_original[-1,0],bo.X_original[-1,1],color='r',s=30,label='Previous Selection')

    acq2d.scatter(bo.X_original[:,0],bo.X_original[:,1], marker='*',label=u'Estimated Peaks by IGMM', s=100,color='green')    


    acq2d.set_title('Acquisition Function',fontsize=16)
    acq2d.set_xlim(bo.bounds[0,0], bo.bounds[0,1])
    acq2d.set_ylim(bo.bounds[1,0], bo.bounds[1,1])
    
    #acq2d.legend(loc=1, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    #acq2d.legend(loc='center left',bbox_to_anchor=(1.01, 0.5))
      
    fig.colorbar(CS_acq, ax=acq2d, shrink=0.9)

def plot_gp_sequential_batch(bo,x_seq,x_batch):
    
    global counter
    counter=counter+1
    
    x1 = np.linspace(bo.scalebounds[0,0], bo.scalebounds[0,1], 100)
    x2 = np.linspace(bo.scalebounds[1,0], bo.scalebounds[1,1], 100)
    
    x1g,x2g=np.meshgrid(x1,x2)
    
    X=np.c_[x1g.flatten(), x2g.flatten()]
    
    x1_ori = np.linspace(bo.bounds[0,0], bo.bounds[0,1], 100)
    x2_ori = np.linspace(bo.bounds[1,0], bo.bounds[1,1], 100)
    
    x1g_ori,x2g_ori=np.meshgrid(x1_ori,x2_ori)
    
    X_ori=np.c_[x1g_ori.flatten(), x2g_ori.flatten()]
    
    fig=plt.figure(figsize=(10, 3))
    
  
    
    #axis2d = fig.add_subplot(1, 2, 1)
    acq2d_seq = fig.add_subplot(1, 2, 1)
    acq2d_batch = fig.add_subplot(1, 2, 2)
    
    #mu, sigma = bo.posterior(X)
    # plot the acquisition function

    utility = bo.acq_func.acq_kind(X, bo.gp, np.max(bo.Y))
    #acq3d.plot_surface(x1g,x1g,utility.reshape(x1g.shape))
    
    CS_acq=acq2d_seq.contourf(x1g_ori,x2g_ori,utility.reshape(x1g.shape),cmap=my_cmap,origin='lower')
    #CS2_acq = plt.contour(CS_acq, levels=CS_acq.levels[::2],colors='r',origin='lower',hold='on')

    acq2d_seq.scatter(x_seq[0],x_seq[1], marker='*',label=u'Estimated Peaks by IGMM', s=100,color='green')    


    acq2d_seq.set_title('Sequential Bayesian Optimization',fontsize=16)
    acq2d_seq.set_xlim(bo.bounds[0,0]-0.2, bo.bounds[0,1]+0.2)
    acq2d_seq.set_ylim(bo.bounds[1,0]-0.2, bo.bounds[1,1]+0.2)

    #acq2d.legend(loc=1, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    #acq2d.legend(loc='center left',bbox_to_anchor=(1.01, 0.5))
      
    fig.colorbar(CS_acq, ax=acq2d_seq, shrink=0.9)
    
    
    
    CS_acq_batch=acq2d_batch.contourf(x1g_ori,x2g_ori,utility.reshape(x1g.shape),cmap=my_cmap,origin='lower')
    #CS2_acq_batch = plt.contour(CS_acq_batch, levels=CS_acq_batch.levels[::2],colors='r',origin='lower',hold='on')

    acq2d_batch.scatter(x_batch[:,0],x_batch[:,1], marker='*',label=u'Estimated Peaks by IGMM', s=100,color='green')    


    acq2d_batch.set_title('Batch Bayesian Optimization',fontsize=16)
    acq2d_batch.set_xlim(bo.bounds[0,0]-0.2, bo.bounds[0,1]+0.2)
    acq2d_batch.set_ylim(bo.bounds[1,0]-0.2, bo.bounds[1,1]+0.2)
    
    fig.colorbar(CS_acq_batch, ax=acq2d_batch, shrink=0.9)

        
    strFolder="V:\\plot_Nov_2016\\sequential_batch"
    strFileName="{:d}.eps".format(counter)
    strPath=os.path.join(strFolder,strFileName)
    fig.savefig(strPath, bbox_inches='tight')