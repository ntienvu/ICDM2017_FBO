# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 09:29:31 2016

@author: VuNguyen
"""

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np
import matplotlib 

#matplotlib.rc('xtick', labelsize=12) 
#matplotlib.rc('ytick', labelsize=12) 

def compute_regret_reduction(d,T,alpha): 
    temp=np.divide((d+1)*np.log(alpha),(2*d+1)*np.log(T))
    Z=(1+temp)**(+d+1)
    #Z=np.sqrt(Z)
    return Z


vu=compute_regret_reduction(2,10,50)
print vu

fig = plt.figure(figsize=(7, 4))
ax = fig.gca(projection='3d')
#alpha=10
alpha=np.arange(2,16,0.2)
#T = np.arange(30, 300, 1)
d = np.arange(3, 16, 1)
#T_grid,d_grid = np.meshgrid(T,d)
alpha_grid,d_grid = np.meshgrid(alpha,d)


# compute fraction
#temp=np.divide((d_grid+1)*np.log(alpha),(2*d_grid+1)*np.log(T_grid))
#Z = (1-temp)**(-d_grid-1)

T_grid=d_grid*20
temp=np.divide((d_grid+1)*np.log(alpha_grid),(2*d_grid+1)*np.log(T_grid))
Z = (1-temp)**(-d_grid-1)
Z=np.sqrt(Z)

    
#surf = ax.plot_surface(T_grid,d_grid , Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                       #linewidth=0, antialiased=False)

surf = ax.plot_surface(alpha_grid,d_grid , Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=True)
                       
#ax.set_zlim(-1.01, 1.01)

plt.plot([12],[10],[Z[10,40]],'ro')


#bbox_props = dict(boxstyle="rarrow,pad=0.3", fc="cyan", ec="b", lw=2)

val=compute_regret_reduction(d=10,T=20*10,alpha=12)
val=round(val,1)
t = ax.text(12, 10,8.5, str(val), ha="center", va="center", rotation=90,
            size=14)
 #           bbox=bbox_props)
                        
                
ax.zaxis.set_major_locator(LinearLocator(6))
ax.xaxis.set_major_locator(LinearLocator(6))
ax.yaxis.set_major_locator(LinearLocator(6))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.f'))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.f'))
ax.xaxis.set_major_formatter(FormatStrFormatter('%.f'))
ax.tick_params(axis='x', pad=-3)
ax.tick_params(axis='y', pad=0)
ax.tick_params(axis='z', pad=0)

#plt.xlabel('T')
#plt.xlabel(r'$\alpha$',fontsize=16)
plt.xlabel(r'$h^{Big} / h^{Small}$',fontsize=14)
plt.ylabel('d',fontsize=14)

plt.title(r'$R^{Big}_T$ / $R^{Small}_T$',fontsize=16)

ax.xaxis.set_label_coords(5, 3.025)


ax.set_zlim(0,21)

#fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()


strFile='d_alpha_T_20d.pdf'
#strFile='d_alpha_T.pdf'

fig.savefig(strFile, bbox_inches='tight')