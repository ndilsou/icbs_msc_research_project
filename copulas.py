# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 00:55:32 2015

@author: Ndil Sou


"""
import numpy as np
from numpy import exp, log, pi, sqrt

from scipy.stats import t, kendalltau, uniform
from scipy.special import gamma
import scipy.optimize as optimize
import matplotlib.pyplot as plt
###############################################################################
#                       Copula functions
###############################################################################


class Clayton:
    def __init__(self, alpha):
        self._alpha = alpha  #parameter of the Clayton Copula

    def pdf(self, u1, u2):
        a = self._alpha
        return (a+1.0) * ((u1**(-a) + u2**(-a) -1.0)**(-2.0-1.0/a)) \
        * (u1 * u2)**(-a-1)
    def logpdf(self, u1, u2):
        a = self._alpha
        return log(a+1.0) + (-2.0 - 1.0/a) * log(u1**(-a) + u2**(-a) -1.0) \
        - (a + 1.0)   * (log(u1) + log(u2))    
 

    def cdf(self, u1, u2):
        a = self._alpha
        return (u1**(-a) + u2**(-a) - 1.0)**(-1.0/a)
    
    def ccdf(self, u1, u2):
        """
         conditional cdf
        returns the quantile of u1, assuming u2 known.
        """
        a = self._alpha
        return u2**(-(1.0+a)) * (u1**(-a) + u2**(-a) - 1.0)**(-(1.0+a)/a)      
        
    def invccdf(self, u2, v):
        """
        returns the u1, given a u2 and a quantile v.
        To simulate for marginals F1 and F2, use u2 = F2(x) and feet the 
        output of the function to invF1(u1)
        """
        a = self._alpha
        return (1.0 + u2**(-a) * (v**(-a/(1.0 + a)) - 1))**(-1.0/a)
        
    def _get_alpha(self):
        return self._alpha
        
    def _set_alpha(self, alpha):
        if alpha != 0.0:
            self._alpha = alpha
            
    alpha = property(_get_alpha, _set_alpha)            
            
    
    def _get_uTDC(self):
        #Upper Tail Dependence Coefficient
        return 0
        
    uTDC = property(_get_uTDC)
    
    def _get_lTDC(self):
        #lower Tail Dependence Coefficient
        if self._alpha > 0:
            return 2.0**(-1.0/self._alpha)
        else:
            return 0
            
    lTDC = property(_get_lTDC)
    
class Gumbel:
    def __init__(self, delta):
        if delta >= 1:
            self._delta = delta #parameter of the Clayton Copula
        
    def pdf(self, u1, u2):
        d = self._delta
        A = ((-log(u1))**d + (-log(u2))**d)**(1/d)
        return (A + d -1.0) * A**(1.0 - 2.0*d) * exp(-A) * (u1 * u2)**(-1.0) * \
        (-log(u1))**(d - 1.0) * (-log(u2))**(d - 1.0)
        
    def logpdf(self, u1, u2):
        d = self._delta
        A = ((-log(u1))**d + (-log(u2))**d)**(1/d)
        return log(A + d -1.0) + (1.0 - 2.0*d) * log(A) - A  -log(u1 * u2)  \
        + (d - 1.0) * log(-log(u1)) + (d - 1.0) * log(-log(u2))
        
        
    def cdf(self, u1, u2):
        d = self._delta
        return exp(-((-log(u1))**d + (-log(u2))**d)**(1.0/d))
        
    def ccdf(self, u1, u2):
        """
        conditional cdf
        returns the quantile of u1, assuming u2 known.
        """
        d = self._delta
        A = ((-log(u1))**d + (-log(u2))**d)
        return u2**(-1) * (-log(u2))**(d - 1) * A**((1-d)/d) * exp(-A**(1/d))     
        
    def invccdf(self, u2, v, tol = 1e-6):
        """
        returns the u1, given a u2 and a quantile v.
        To simulate for marginals F1 and F2, use u2 = F2(x) and feet the 
        output of the function to invF1(u1)
        
        Given that there is no closed form formula we use a simple bisection 
        search in the unit range to find the correct value. 
        """

        funct = lambda u1 : self.ccdf(u1, u2)
        err = 10000
        low = 0.0
        hi = 1.0
        while (abs(err) > tol):
            mid = (hi + low)/2.0
            q = funct(mid)
            err = q - v
            if err >= 0.0:
                hi = mid
            else:
                low = mid
        return mid
        
    def _get_delta(self):
        return self._delta
    def _set_delta(self, delta):
        if delta >= 1:
            self._delta = delta
    delta = property(_get_delta, _set_delta)
        
    def _get_uTDC(self):
        #Upper Tail Dependence Coefficient
        return 2.0 - 2.0**(1.0/self._delta)
    uTDC = property(_get_uTDC)
            
    def _get_lTDC(self):
        #lower Tail Dependence Coefficient
        return 0
    lTDC = property(_get_lTDC)

###############################################################################
#                       Skew-T Distribution (Hansen, 1994)
###############################################################################
class Skew_t:
    def __init__(self, df, skew):
        self._df = df
        self._skew = skew
        self._a = float() 
        self._b = float()
        self._c = float()
        self._update()
    
    def _update(self):
        self._c = gamma(0.5 * (self._df + 1.0)) \
        / ( gamma(self._df / 2.0) * sqrt(pi * (self._df -2.0)))
        
        self._a = 4.0 * self._skew * self._c * ((self._df - 2.0)/(self._df - 1.0))
        
        self._b = sqrt(1.0 + 3.0 * self._skew**2 - self._a**2)
                
    def cdf(self, y):
        a = self._a
        b = self._b

        s = self._skew
        df = self._df
        pasting = -(a/b)
        
        if y < pasting:
            x = sqrt(df/(df - 2.0)) * ((b * y + a)/(1.0 - s))
            return (1.0 - s) * t.cdf(x, df)
        elif y == pasting:
            return (1.0 - s)/2.0
        else: #y > pasting
            x = sqrt(df/(df - 2.0)) * ((b * y + a)/(1.0 + s))
            return (1.0 + s) * t.cdf(x, df) - s

    
    def pdf(self, y):
        a = self._a
        b = self._b
        c = self._c
        s = self._skew
        df = self._df
        pasting = -(a/b)
        
        if y < pasting:
            x = (b * y + a)/(1.0 - s)
        else: # y > pasting
            x = (b * y + a)/(1.0 + s)
        return b * c * (1.0 + 1.0/(df - 2.0) * (x)**2)**(-(df + 1.0)/2.0)
        
        
    def logpdf(self, y):
        a = self._a
        b = self._b
        c = self._c
        s = self._skew
        df = self._df
        pasting = -(a/b)

        if y < pasting:
            x = (b * y + a)/(1 - s)
        else: # y > pasting
            x = (b * y + a)/(1 + s)
        return log(b*c) - ((df + 1.0)/2.0)*log(1.0 + 1.0/(df - 2.0) * (x)**2)
        
    def invcdf(self, u):            
        a = self._a
        b = self._b
        s = self._skew
        df = self._df
        pasting = (1.0 - s)/2.0
        sqr_df = sqrt(1.0 - 2.0/df)
        
        if u < pasting:
            t_u = u / (1.0 - s)
            return 1.0 / b * ((1.0 - s)*sqr_df*t.ppf(t_u, df) - a)
        else: #u <= pasting
            t_u = (u + s) / (1.0 + s)
            return 1.0 / b * ((1.0 + s)*sqr_df*t.ppf(t_u, df) - a)
            
    def _get_skew(self):
        return self._skew
    def _set_skew(self, skew):
        self._skew = skew
    skew = property(_get_skew, _set_skew)
    
    def _get_df(self):
        return self._df
    def _set_df(self, df):
        self._df = df
    df = property(_get_df, _set_df)

###############################################################################
#                       GARCH + LIKELIHOOD
###############################################################################

def maker_garch(alpha, beta, omega, theta = 0):
    """
    alpha, beta and omega are the parameter of the model. 
    
    """
    def garch(r_t, sigma2_t):
        """
        r_t and sigma_t are the current returns and current Garch-variance.
        """
        return omega + alpha * r_t**2 + beta * sigma2_t
    
    def agarch(r_t, sigma2_t):
        """
        r_t and sigma_t are the current returns and current Garch-variance.
        """
        return omega + alpha * (r_t - theta * sqrt(sigma2_t))**2 + beta * sigma2_t
    func_list = [garch, agarch]
    if theta != 0:
        idx = 1
    else:
        idx = 0
    return func_list[idx]
    
def log_likelihood_skew_t(R, df, skew, alpha, beta, omega, theta):
    """
    We compute the logL of a series of returns under the assumption that the 
    the variance follows a GARCH process.
    Note that we directly embed the Constraints on the GARCH and the skew_t
    in the likelihood
    """
    #This is an incoherency if we look at the copula likelihood but we only support
    #flat python lists in this function.
    if isinstance(R, np.ndarray):
        R = R.flatten().tolist()
    
    garch = maker_garch(alpha, beta, omega, theta)        
    skewT = Skew_t(df, skew)
    
    sample_variance = np.var(R)
    log_likelihood = skewT.logpdf(R[0]/sqrt(sample_variance)) \
                    - log(sample_variance)/2.0
                    
    var_garch = garch(R[0], sample_variance)
    for r_t in R[1:]: 

        if 1 - (alpha * (1 + theta**2) + beta) >= 0 \
        and var_garch > 0: 
            log_likelihood = log_likelihood + skewT.logpdf(r_t/sqrt(var_garch)) \
                                            - log(var_garch)/2.0
        else:
            log_likelihood = log_likelihood - 100000
        var_garch = garch(r_t, var_garch)

    return log_likelihood
    
def log_likelihood_garch(R, alpha, beta, omega):        
    logL = lambda r_t, sigma2_t: \
        -0.5 * (log(2 * pi) + log(sigma2_t) + r_t**2 / sigma2_t)
    garch = maker_garch(alpha, beta, omega)
    sample_variance = np.var(R)
   
    log_likelihood = logL(R[0], sample_variance)
    
    var_garch = garch(R[0], sample_variance)     
    for r_t in R[1:]: 

        if 1 - (alpha + beta) >= 0 and var_garch > 0:
            log_likelihood = log_likelihood + logL(r_t, var_garch) 
        else:
            log_likelihood = log_likelihood - 100000
        var_garch = garch(r_t, var_garch)
    return log_likelihood
      
    
def maker_ieqcons_garch(R):
    """
    x = [alpha, beta, omega]
    returns a list of all the constraint defined for the problem
    """
    def check_positive_variance(x):
#        garch = maker_garch( x[0], x[1], x[2])
        garch = lambda r_t, sigma2_t, alpha, beta, omega : \
                omega + alpha * r_t**2 + beta * sigma2_t
        var_garch = garch(R[0],R[0]**2, x[0], x[1], x[2])
        total_neg = 0 #We return the sum of the negative values
        for r_t in R[1:]: 
            var_garch = garch(r_t,var_garch, x[0], x[1], x[2])
            if var_garch < 0:
                total_neg = total_neg + var_garch
        return total_neg
        
    def check_garch_persistence(x):
        return 1 - (x[0] + x[1])

        
    return [check_positive_variance, check_garch_persistence]


def log_likelihood_clayton(u1, u2, alpha):
    """
    The function returns the gradient as well
    """
    if not isinstance(u1, np.ndarray) or not isinstance(u2, np.ndarray):
        raise TypeError("Only support ndarray for u1 and u2")
    clayton = Clayton(alpha)
    #note that we take full advantage of ndarrays vectorizated operations
    return clayton.logpdf(u1, u2).sum()


###############################################################################
#                       OPTIMIZATION PROCEDURE
###############################################################################

#Given that the joint optimization of the skewT and the Garch is a highly nontrivial
#problem for the various optimization algorithm available in Python, we need another approach.
#Of course we admit that the issue might come from our lack of understanding of 
#the full problem, in the first place.
#Our procedure is a sequential maximum likelihood procedure. We take inspiration both
#from the QMLE procedure and the EM algorithm. 
#1) define x0Skew and x0Garch the initial set of parameters for,
# the skewT and the garch problem, respectively.
#2) assume the parameters of the garch known, optimize for the skewT
#3) assume the parameters of the skewT known, optimize for the garch
#4) check current likelihood distance from its value at 1. If distance below tolerance
#or max number of iteration reached, stop. Else, go back to 2.
def fit_copula(u1, u2, verbose = False, alpha0 = None):
    #Default behavior is to use the kendall Tau as the initial value.
    if not isinstance(u1, np.ndarray):
        u1 = np.array(u1)
    if not isinstance(u2, np.ndarray):
        u2 = np.array(u2)
    if alpha0 is None:
        tau, pval = kendalltau(u1, u2, 
                                  initial_lexsort=False)
        alpha0 = 2 * tau / (1 - tau)
    log_copula = lambda alpha: -log_likelihood_clayton(u1, u2, alpha)
    res = optimize.fmin(log_copula, alpha0, disp=verbose)
    if verbose:
        print "\nOptimal Parameter: {0}".format(res)
    return res
    
    
def fit_marginal(R, x0, maxiter = 15 , verbose = 0, etol = 1e-3):
    """
    Given a list of asset returns, returns the set of optimal parameters for a 
    A-GARCH skew T model using an iterated maximum likelihood method.
    Solution is returned in a dictionary.
    verbose defines the level of information provided.
    0: no output printed:
    1: print the final parameters
    2: print parameters at each step.
    """
    x_skew = x0[:2]
    x_garch = x0[2:]

    old_likelihood = -np.inf
    new_likelihood = np.inf
    cur_iter = 0
    
    #x_skew = [df, skew] 
    log_marginal_skew = lambda x_skew : \
            -log_likelihood_skew_t(R, x_skew[0], x_skew[1], 
                                   x_garch[0], x_garch[1], 
                                   x_garch[2], x_garch[3])
                                   
    #x_garch = [alpha, beta, omega, theta]                             
    log_marginal_garch = lambda x_garch : \
            -log_likelihood_skew_t(R, x_skew[0], x_skew[1], 
                                   x_garch[0], x_garch[1],
                                   x_garch[2], x_garch[3])
                                       
    log_marginal = lambda x_skew, x_garch : \
                -log_likelihood_skew_t(R, x_skew[0], x_skew[1], 
                                       x_garch[0], x_garch[1], 
                                       x_garch[2], x_garch[3])
    

    while ((new_likelihood - old_likelihood)**2 > etol):        
        old_likelihood = new_likelihood
        old_likelihood = log_marginal(x_skew, x_garch)
        
        #skew step
        res = optimize.fmin(log_marginal_skew, x_skew, disp=False)
        x_skew = res.tolist()

        #garch step                
        res = optimize.fmin(log_marginal_garch, x_garch, disp=False)
        x_garch = res.tolist()

        
        new_likelihood = log_marginal(x_skew, x_garch)
        
        if verbose == 2:
            print "\nCurrent Iteration : %d" % (cur_iter)
            print "Skew Step" 
            template_skew = "\n df: {0[0]} \n skew: {0[1]}"
            print template_skew.format(x_skew)
            print "Garch Step" 
            template_garch = "\n alpha: {0[0]} \n beta: {0[1]} \n omega: {0[2]} \n theta: {0[3]}"
            print template_garch.format(x_garch)
            print "current logLikelihood: {}".format(new_likelihood)
            
            
        cur_iter = cur_iter + 1
        if maxiter <= cur_iter:
            break
    
    x_skew.extend(x_garch)
    if verbose >= 1:
        template = []
        template.append("\nFinal Parameters :")
        template.append("\n df: {0[0]} \n skew: {0[1]} \n alpha: {0[2]} \n beta: {0[3]}")
        template.append(" \n omega: {0[4]} \n theta: {0[5]}")
        template = str.join("", template)
        print template.format(x_skew)
        print "final logLikelihood: {}".format(new_likelihood)
    params = ['df', 'skew', 'alpha', 'beta', 'omega', 'theta']
    res = {param : value for param, value in zip(params, x_skew)}        
    return res

if __name__ == '__main__':
    U = uniform.rvs(size = (500,2))
    alphas = [0.1, 3, 10]
    fig = plt.figure(figsize=(15,5))
    for i, alpha in enumerate(alphas):
        sim_claytons = Clayton(alpha)
        ax = fig.add_subplot(1, 3, i+1)
        plt.ylim([0,1])
        plt.xlim([0,1])   
        U_0 = sim_claytons.invccdf(U[:,1],U[:,0])
        plt.scatter(U_0,U[:,1])
        ax.set_title('alpha = {}'.format(alpha))