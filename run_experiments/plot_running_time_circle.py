# -*- coding: utf-8 -*-
"""
Created on Sat Sep 03 18:10:00 2016

@author: tvun
"""

# plot computational time
import sys
sys.path.insert(0,'..')
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from prada_bayes_opt import visualization
from prada_bayes_opt import auxiliary_functions
import matplotlib
#from pylab import *



ax=plt.subplot(aspect='equal')
sns.set(style="ticks")

matplotlib.rc('xtick', labelsize=14)
matplotlib.rc('ytick', labelsize=14)

fig=plt.figure(figsize=(8, 4.3))


x_axis = [2,3,5,6,10] # dimension

# FBO
Time_FBO=np.array([23,73,241,636,6500])
LogTime_FBO=np.log(Time_FBO)
Radius_FBO=Time_FBO

y = [0]*len(x_axis)
plt.scatter(x_axis,y,s=Radius_FBO)
#out = circles(x_axis, y, Radius_FBO*0.1, c=Radius_FBO, alpha=0.5, edgecolor='red',ec='red')


# EI_H
Time_EI_H=np.array([3.4,31,81.3,148,2668])
LogTime_EI_H=np.log(Time_EI_H)
Radius_EI_H=Time_EI_H
#Radius_EI_H=Time_EI_H
y = [1.5]*len(x_axis)
plt.scatter(x_axis,y,s=Radius_EI_H,color='red')
#out = circles(x_axis, y, Radius_EI_H*0.1, c=Radius_EI_H, alpha=0.5, edgecolor='blue',ec='red')
#colorbar(out)

# EI_Q
Time_EI_Q=np.array([2.3,30.11,74.33,124.5,1912])
LogTime_EI_Q=np.log(Time_EI_Q)
Radius_EI_Q=Time_EI_Q
y = [3]*len(x_axis)
plt.scatter(x_axis,y,s=Radius_EI_Q,color='m')
#out = circles(x_axis, y, Radius_EI*0.5, c=Radius_EI*0.5, alpha=0.5, edgecolor='none')


# VolL
Time_VolL=np.array([8.7,52.5,65,100,1730])
Radius_VolL=Time_VolL
y = [4.5]*len(x_axis)
plt.scatter(x_axis,y,s=Radius_VolL,color='k')


# BigB
Time_BigB=np.array([4.5,16.2,86,176.8,1889])
Radius_BigB=Time_BigB
y = [6]*len(x_axis)
plt.scatter(x_axis,y,s=Radius_BigB,color='g')

        
        
plt.annotate(
    '6500', 
    xy = (9.13, -.6), xytext = (3, 5),
    textcoords = 'offset points', ha = 'right', va = 'bottom',
    bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
    #arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'), fontsize=15)     
    fontsize=15)     
    
plt.annotate(
    '2668', 
    xy = (9.35, 1.), xytext = (3, 5),
    textcoords = 'offset points', ha = 'right', va = 'bottom',
    bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
    #arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'), fontsize=15)  
    fontsize=15)  


plt.annotate(
'86', 
xy = (4.7, 5.5), xytext = (3, 5),
textcoords = 'offset points', ha = 'right', va = 'bottom',
bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
#arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'), fontsize=15)  
fontsize=15)  


plt.annotate(
'66', 
xy = (4.7, 4), xytext = (3, 5),
textcoords = 'offset points', ha = 'right', va = 'bottom',
bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
#arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'), fontsize=15)  
fontsize=15)  



plt.annotate(
'74', 
xy = (4.66, 2.5), xytext = (3, 5),
textcoords = 'offset points', ha = 'right', va = 'bottom',
bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
#arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'), fontsize=15)  
fontsize=15)  



plt.annotate(
'82', 
xy = (4.66,1), xytext = (3, 5),
textcoords = 'offset points', ha = 'right', va = 'bottom',
bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
#arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'), fontsize=15)  
fontsize=15)  

plt.annotate(
'240', 
xy = (4.65, -0.5), xytext = (3, 5),
textcoords = 'offset points', ha = 'right', va = 'bottom',
bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
#arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'), fontsize=15)  
fontsize=15) 


plt.annotate(
'1889', 
xy = (9.42, 5.5), xytext = (3, 5),
textcoords = 'offset points', ha = 'right', va = 'bottom',
bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
#arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'), fontsize=15)  
fontsize=15)  

plt.annotate(
'1912', 
xy = (9.42, 2.5), xytext = (3, 5),
textcoords = 'offset points', ha = 'right', va = 'bottom',
bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
#arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'), fontsize=15)  
fontsize=15)  


plt.annotate(
'1730', 
xy = (9.44, 4), xytext = (3, 5),
textcoords = 'offset points', ha = 'right', va = 'bottom',
bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
#arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'), fontsize=15)  
fontsize=15)  

       
#set labels
labels=['FBO','EI-H','EI-Q','Vol_L','Vanilla BO']
ax = plt.gca()
ax.set_yticks(np.array([0,1.5,3,4.5,6,7.5]))
ax.set_yticklabels(labels,fontsize=14)


plt.xlim([4,10.6])
plt.ylim([-1.6,7])

plt.xlabel('Dimension',fontsize=14)
#plt.ylabel('Methods',fontsize=15)
plt.title('Computational Time (sec) per Iteration',fontsize=18)


strFile='Time_Circle_Comparison_FBO.pdf'
plt.savefig(strFile, bbox_inches='tight')
#plt.savefig(strFile)