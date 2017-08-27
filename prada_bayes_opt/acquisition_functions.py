from __future__ import division
import numpy as np
from scipy.stats import norm
from sklearn.metrics.pairwise import euclidean_distances
#from prada_gaussian_process import PradaGaussianProcess


class AcquisitionFunction(object):
    """
    An object to compute the acquisition functions.
    """

    def __init__(self, acq):
        """
        If UCB is to be used, a constant kappa is needed.
        """
        self.acq=acq
        acq_name=acq['name']
        if acq_name not in ['bucb','ucb', 'ei','ei_H','ei_Q','ei_multiple', 'poi','nei','lei','random','thompson',
                            'pes','pure_exploration','ei_mu','mu','lcb']:
            err = "The utility function " \
                  "{} has not been implemented, " \
                  "please choose one of ucb, ei, or poi.".format(acq_name)
            raise NotImplementedError(err)
        else:
            self.acq_name = acq_name
            
        # used for Thompson Sampling
        self.dim=acq['dim']
        self.UU_dim=500 # dimension of random feature
        self.UU=np.random.multivariate_normal([0]*self.UU_dim,0.00056*np.eye(self.UU_dim),self.dim)        
        
        # vector theta for thompson sampling
        self.flagTheta_TS=0
            

    def acq_kind(self, x, gp, y_max):
        
        #if type(meta) is dict and 'y_max' in meta.keys():
        #   y_max=meta['y_max']
                
            
            
        #print self.kind
        if np.any(np.isnan(x)):
            return 0
        if self.acq_name == 'bucb':
            return self._bucb(x, gp, self.acq['kappa'])
        if self.acq_name == 'ucb':
            return self._ucb(x, gp, self.acq['kappa'])
        if self.acq_name == 'lcb':
            return self._lcb(x, gp, self.acq['kappa'])
        if self.acq_name == 'ei':
            return self._ei(x, gp, y_max)
        if self.acq_name == 'ei_H':
            return self._ei_regularizerH(x, gp, y_max,self.acq['x_bar'],self.acq['R'])
        if self.acq_name == 'ei_Q':
            return self._ei_regularizerQ(x, gp, y_max,self.acq['x_bar'])
        if self.acq_name == 'nei':
            return self._nei2(x, gp, y_max,self.L)
            #return self._poi(x, gp, y_max)
        if self.acq_name == 'lei':
            return self._lei(x, gp, y_max,self.acq['k'])
        if self.acq_name == 'poi':
            return self._poi(x, gp, y_max)
        if self.acq_name == 'thompson':
            return self._thompson_sampling(x, gp)
        if self.acq_name == 'ei_multiple':
            return self._ei_multiple(x, gp, y_max)
        if self.acq_name == 'pure_exploration':
            return self._pure_exploration(x, gp) 
        if self.acq_name == 'ei_mu':
            return self._ei(x, gp, y_max)
        if self.acq_name == 'mu':
            return self._mu(x, gp)
            
    def utility_plot(self, x, gp, y_max):
        
        if np.any(np.isnan(x)):
            return 0

        if self.acq_name == 'ei':
            return self._ei_plot(x, gp, y_max)

  
    @staticmethod
    def _nei2(x, gp, y_max,L):
        #print "nei 2"
        
        mean, var = gp.predict(x, eval_MSE=True)
        var = np.maximum(var, 1e-9 + 0 * var)
        
        if x.shape[0]==1: # only one data point
        # identify the nearest 
            Euc_dist=euclidean_distances(x,gp.X)
            min_dist=np.min(Euc_dist)
            neighbor_idx=np.argmin(Euc_dist,axis=1)
            #neighbor_val=gp.X[neighbor_idx]
        else:
            Euc_dist=euclidean_distances(x,gp.X)
            min_dist=np.min(Euc_dist,axis=1)
            neighbor_idx=np.argmin(Euc_dist,axis=1)
            #neighbor_val=gp.X[neighbor_idx]
        z = (gp.Y[neighbor_idx]+L*min_dist- y_max)/np.sqrt(var)
        #z = (gp.Y[neighbor_idx]-L*min_dist- y_max)/np.sqrt(var)
        out=(gp.Y[neighbor_idx]+L*min_dist - y_max) * norm.cdf(z) + np.sqrt(var) * norm.pdf(z)    
        #out=(gp.Y[neighbor_idx]-L*min_dist - y_max) * norm.cdf(z) + np.sqrt(var) * norm.pdf(z)    
      
        return out

    #@staticmethod
    def _lei(self,x, gp, y_max,k_neighbor):
        #def k_largest(a,N): return np.argsort(a)[::-1][:N] (N = len(a))
    
        # convert k_neighbor  % to scalar
        if len(gp.Y)<3:# use EI when we have less #data point
            return self._ei(x, gp, y_max)
        #else:
            #k_neighbor=np.ceil(k_neighbor*len(gp.Y))        
        
        
        def k_smallest(a,N): 
            if len(a)<N:
                return xrange(len(a))
                
            #return np.argsort(a)[:N] (N = len(a))
            return np.argsort(a)[:N]
        def k_smallest_matrix(a,N): 
            return np.argsort(a,axis=1)[:,:N]
            
        mean, var = gp.predict(x, eval_MSE=True)
        mean=mean.reshape(len(mean),1)
        var=var.reshape(len(var),1)
        var = np.maximum(var, 1e-9 + 0 * var)
        
        if x.shape[0]==1: # only one data point
        # identify the nearest 
            Euc_dist=euclidean_distances(x,gp.X)
            #min_dist=np.min(Euc_dist)
            #neighbor_idx=heapq.nlargest(k_neighbor, range(len(Euc_dist)), Euc_dist.take)
            neighbor_idx=k_smallest(Euc_dist,k_neighbor)
            
            #z = (mean-np.amax(gp.Y[neighbor_idx]))/np.sqrt(var)
            z = (mean-np.mean(gp.Y[neighbor_idx]))/np.sqrt(var)
            #out=(mean-np.amax(gp.Y[neighbor_idx])) * norm.cdf(z) + np.sqrt(var) * norm.pdf(z)  
            out=(mean-np.mean(gp.Y[neighbor_idx])) * norm.cdf(z) + np.sqrt(var) * norm.pdf(z)  
        else:
            Euc_dist=euclidean_distances(x,gp.X)
            #min_dist=np.min(Euc_dist,axis=1)
            neighbor_idx=k_smallest_matrix(Euc_dist,k_neighbor)
            #neighbor_idx=np.argmin(Euc_dist,axis=1)
            #neighbor_val=gp.X[neighbor_idx]
            
            temp_max=np.amax(gp.Y[neighbor_idx],axis=1)
            temp_max=temp_max.reshape(len(temp_max),1)
            z = (mean-temp_max)/np.sqrt(var)
            #z = (mean-np.mean(gp.Y[neighbor_idx],axis=1))/np.sqrt(var)
            out=(mean-temp_max) * norm.cdf(z) + np.sqrt(var) * norm.pdf(z)    
            #out=(mean-np.mean(gp.Y[neighbor_idx],axis=1)) * norm.cdf(z) + np.sqrt(var) * norm.pdf(z)    
        #out=(gp.Y[neighbor_idx]-L*min_dist - y_max) * norm.cdf(z) + np.sqrt(var) * norm.pdf(z)    
      
        return out

    @staticmethod
    def _mu(x, gp):
        mean, var = gp.predict(x, eval_MSE=True)
        mean=np.atleast_2d(mean).T
               
        return mean
                
    @staticmethod
    def _ucb(x, gp, kappa):
        mean, var = gp.predict(x, eval_MSE=True)
        var.flags['WRITEABLE']=True
        #var=var.copy()
        var[var<1e-10]=0
        mean=np.atleast_2d(mean).T
        var=np.atleast_2d(var).T
                
        return mean + kappa * np.sqrt(var)

    @staticmethod
    def _lcb(x, gp, kappa):
        mean, var = gp.predict(x, eval_MSE=True)
        var.flags['WRITEABLE']=True
        #var=var.copy()
        var[var<1e-10]=0
        mean=np.atleast_2d(mean).T
        var=np.atleast_2d(var).T
                
        return mean - kappa * np.sqrt(var)
        
    @staticmethod
    def _pure_exploration(x, gp):
        mean, var = gp.predict(x, eval_MSE=True)
        var.flags['WRITEABLE']=True
        #var=var.copy()
        var[var<1e-10]=0
        mean=np.atleast_2d(mean).T
        var=np.atleast_2d(var).T
                
        return np.sqrt(var)
        
        
    @staticmethod
    def _bucb(x, gp, kappa):
        mean, var = gp.predict_bucb(x, eval_MSE=True)
        var.flags['WRITEABLE']=True
        #var=var.copy()
        var[var<1e-10]=0
        mean=np.atleast_2d(mean).T
        var=np.atleast_2d(var).T
                
        return mean + kappa * np.sqrt(var)

                
    #@staticmethod
    def _ei(self,x, gp, y_max):
        mean, var = gp.predict(x, eval_MSE=True)

        if gp.nGP==0:

                
            var2 = np.maximum(var, 1e-4 + 0 * var)
            z = (mean - y_max)/np.sqrt(var2)        
            out=(mean - y_max) * norm.cdf(z) + np.sqrt(var2) * norm.pdf(z)
            
            out[var<1e-4]=0
            return out
        else:                
            z=[None]*gp.nGP
            out=[None]*gp.nGP
            # Avoid points with zero variance
            for idx in range(gp.nGP):
                var[idx] = np.maximum(var[idx], 1e-9 + 0 * var[idx])
            
                z[idx] = (mean[idx] - y_max)/np.sqrt(var[idx])
            
                out[idx]=(mean[idx] - y_max) * norm.cdf(z[idx]) + np.sqrt(var[idx]) * norm.pdf(z[idx])
                
            if len(x)==1000:    
                return out
            else:
                return np.mean(out)# get mean over acquisition functions
                return np.prod(out,axis=0) # get product over acquisition functions

    #@staticmethod
    def _ei_regularizerQ(self,x, gp, y_max,x_bar):
        mean, var = gp.predict(x, eval_MSE=True)

        #compute regularizer xi
        xi= np.linalg.norm(x - x_bar)


        if gp.nGP==0:
            var = np.maximum(var, 1e-9 + 0 * var)
            z = (mean - y_max-y_max*xi)/np.sqrt(var)        
            out=(mean - y_max-y_max*xi) * norm.cdf(z) + np.sqrt(var) * norm.pdf(z)
            
            return out
        else:                
            z=[None]*gp.nGP
            out=[None]*gp.nGP
            # Avoid points with zero variance
            for idx in range(gp.nGP):
                var[idx] = np.maximum(var[idx], 1e-9 + 0 * var[idx])
            
                z[idx] = (mean[idx] - y_max-y_max*xi)/np.sqrt(var[idx])
            
                out[idx]=(mean[idx] - y_max-y_max*xi) * norm.cdf(z[idx]) + np.sqrt(var[idx]) * norm.pdf(z[idx])
                
            if len(x)==1000:    
                return out
            else:
                return np.mean(out)# get mean over acquisition functions
                return np.prod(out,axis=0) # get product over acquisition functions

    #@staticmethod
    def _ei_regularizerH(self,x, gp, y_max,x_bar,R):
        mean, var = gp.predict(x, eval_MSE=True)

        #compute regularizer xi
        dist= np.linalg.norm(x - x_bar)
        if dist>R:
            xi=dist/R-1
        else:
            xi=0
                
        
        if gp.nGP==0:
            var = np.maximum(var, 1e-9 + 0 * var)
            z = (mean - y_max-y_max*xi)/np.sqrt(var)        
            out=(mean - y_max-y_max*xi) * norm.cdf(z) + np.sqrt(var) * norm.pdf(z)
            
            return out
        else:                
            z=[None]*gp.nGP
            out=[None]*gp.nGP
            # Avoid points with zero variance
            for idx in range(gp.nGP):
                var[idx] = np.maximum(var[idx], 1e-9 + 0 * var[idx])
            
                z[idx] = (mean[idx] - y_max-y_max*xi)/np.sqrt(var[idx])
            
                out[idx]=(mean[idx] - y_max-y_max*xi) * norm.cdf(z[idx]) + np.sqrt(var[idx]) * norm.pdf(z[idx])
                
            if len(x)==1000:    
                return out
            else:
                return np.mean(out)# get mean over acquisition functions
                return np.prod(out,axis=0) # get product over acquisition functions
                
    #@staticmethod
    def _ei_multiple(self,x, gp, y_max):
        prob_reward=self.acq['prob_reward']
        mean, var = gp.predict(x, eval_MSE=True)

        if gp.nGP==0:
            var = np.maximum(var, 1e-9 + 0 * var)
            z = (mean - y_max)/np.sqrt(var)        
            out=(mean - y_max) * norm.cdf(z) + np.sqrt(var) * norm.pdf(z)
            
            return out
        else:                
            z=[None]*gp.nGP
            out=[None]*gp.nGP
            # Avoid points with zero variance
            for idx in range(gp.nGP):
                var[idx] = np.maximum(var[idx], 1e-9 + 0 * var[idx])
            
                z[idx] = (mean[idx] - y_max)/np.sqrt(var[idx])
            
                out[idx]=(mean[idx] - y_max) * norm.cdf(z[idx]) + np.sqrt(var[idx]) * norm.pdf(z[idx])
                
            if len(x)==1000:    
                return out
            else:
                return np.sum(np.ravel(out)*prob_reward)# get mean over acquisition functions
                return np.prod(out,axis=0) # get product over acquisition functions
                
    #@staticmethod
    def _thompson_sampling(self,x, gp):
        
        if self.flagTheta_TS==0:
            # computing Phi(X)^T=[phi(x_1)....phi(x_n)]
            Phi_X=np.hstack([np.sin(np.dot(gp.X,self.UU)), np.cos(np.dot(gp.X,self.UU))]) # [N x M]
            #Phi_X=np.asarray(Phi_X)
            
            # computing A^-1
            A=np.dot(Phi_X.T,Phi_X)+gp.noise_delta
            
            # theta ~ N( A^-1 Phi_T Y, sigma^2 A^-1
            temp_mean_theta=np.dot(Phi_X.T,gp.Y)
            self.mean_theta_TS=np.linalg.solve(A,temp_mean_theta)
            self.flagTheta_TS=1
        
        phi_x=np.hstack([np.sin(np.dot(x,self.UU)), np.cos(np.dot(x,self.UU))])
        fx=np.dot(phi_x,self.mean_theta_TS)

        return fx        
                
                
    # for plot purpose
    @staticmethod
    def _ei_plot(x, gp, y_max):
        mean, var = gp.predict(x, eval_MSE=True)
        
        if gp.nGP==0:
            var = np.maximum(var, 1e-9 + 0 * var)
    
            #mean=np.mean(mean)
            #var=np.mean(var)
            
            z = (mean - y_max)/np.sqrt(var)
        
            out=(mean - y_max) * norm.cdf(z) + np.sqrt(var) * norm.pdf(z)
            
            return out
        else:                
            z=[None]*gp.nGP
            out=[None]*gp.nGP
            
            prod_out=[1]*len(mean[0])
            # Avoid points with zero variance
            for idx in range(gp.nGP):
                var[idx] = np.maximum(var[idx], 1e-9 + 0 * var[idx])
    
            #mean=np.mean(mean)
            #var=np.mean(var)
                z[idx] = (mean[idx] - y_max)/np.sqrt(var[idx])
            
                out[idx]=(mean[idx] - y_max) * norm.cdf(z[idx]) + np.sqrt(var[idx]) * norm.pdf(z[idx])
                prod_out=prod_out*out[idx]

            out=np.asarray(out)
            
            #return np.mean(out,axis=0) # mean over acquisition functions
            return np.prod(out,axis=0) # product over acquisition functions
            #return prod_out
      
            
    @staticmethod
    def _poi(x, gp, y_max):
        mean, var = gp.predict(x, eval_MSE=True)

        # Avoid points with zero variance
        var = np.maximum(var, 1e-9 + 0 * var)

        z = (mean - y_max)/np.sqrt(var)
        

        return norm.cdf(z)


def unique_rows(a):
    """
    A functions to trim repeated rows that may appear when optimizing.
    This is necessary to avoid the sklearn GP object from breaking

    :param a: array to trim repeated rows from

    :return: mask of unique rows
    """

    # Sort array and kep track of where things should go back to
    order = np.lexsort(a.T)
    reorder = np.argsort(order)

    a = a[order]
    diff = np.diff(a, axis=0)
    ui = np.ones(len(a), 'bool')
    ui[1:] = (diff != 0).any(axis=1)

    return ui[reorder]



class BColours(object):
    BLUE = '\033[94m'
    CYAN = '\033[36m'
    GREEN = '\033[32m'
    MAGENTA = '\033[35m'
    RED = '\033[31m'
    ENDC = '\033[0m'
