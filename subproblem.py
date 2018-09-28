'''
    K arms with identical independent distributed normal reward
    mean unknown, std known
    normal prior on the mean reward, updated Bayesian
    Finite horizon T
'''

from math import *
from helper import *
import numpy as np
from joblib import Parallel, delayed
import time

class sub_instance:
    def __init__(self, mu, delta, K, T):
        self.reward_mean = mu
        self.reward_std = delta
        self.reward_precision = 1/pow(delta,2)
        self.num_arms = K
        self.horizon = T
        self.reward_upperbound = K*T*mu if mu>0 else 0

    def reward(self,k):
        return np.random.normal(self.reward_mean,self.reward_std,k)

class discrete_DP:
    def __init__(self, minstance, mu_hat, tau_hat, D):
        self.instance = instance
        self.mu_prior = mu_hat
        self.tau_prior = tau_hat
        self.T = instance.horizon

        self.reward_total = 0

    def dp(self, mu, tau, t):
        # TODO: DP formulation
        return 0

    def sim(self):
        # TODO: Simulation
        return 0


class sub_thompson:
    def __init__(self, instance, mu_hat, tau_hat):
        self.instance = instance
        self.mu_prior = mu_hat
        self.tau_prior = tau_hat
        self.T = instance.horizon

        self.reward_total = 0

    def sim(self):
        for t in range(self.T):
            # Sample
            x_sampled= np.random.normal(self.mu_prior, 1/np.sqrt(self.tau_prior), self.instance.num_arms)
            # Act greedily
            k = (x_sampled>0).sum()
            # Observe outcome
            x_observed = self.instance.reward(k)
            self.reward_total += sum(x_observed)
            if len(x_observed)>0:
                x_bar = np.average(x_observed)
                # Update
                self.mu_prior = (self.tau_prior*self.mu_prior + k*self.instance.reward_precision*x_bar)/(self.tau_prior+k*self.instance.reward_precision)
                self.tau_prior = self.tau_prior + k*self.instance.reward_precision
        # print(self.reward_total)
        # print(self.instance.reward_upperbound)
        return self.reward_total

class greedy:
    def __init__(self, instance, mu_hat, tau_hat):
        self.instance = instance
        self.mu_prior = mu_hat
        self.tau_prior = tau_hat
        self.T = instance.horizon

        self.reward_total = 0

    def sim(self):
        for t in range(self.T):
            # Act greedily
            k = K if self.mu_prior >= 0 else 0
            # Observe outcome
            x_observed = self.instance.reward(k)
            self.reward_total += sum(x_observed)
            if len(x_observed) > 0:
                x_bar = np.average(x_observed)
                # Update
                self.mu_prior = (self.tau_prior*self.mu_prior + k*self.instance.reward_precision*x_bar)/(self.tau_prior+k*self.instance.reward_precision)
                self.tau_prior = self.tau_prior + k*self.instance.reward_precision
        return self.reward_total

def run_sims_greedy(instance):
    this_greedy = greedy(instance, 0, instance.reward_std)
    return this_greedy.sim()

def run_sims_thompson(instance):
    this_thompson = sub_thompson(instance, 0, instance.reward_std)
    return this_thompson.sim()

def run_sims_dp(instance):
    this_dp = discrete_DP(instance, 0, instance.reward_std, 10)
    return this_dp.sim()



if __name__ == '__main__':
    mu = 0.1
    delta = 1
    K = 100
    T = 10
    test_instance = sub_instance(mu,delta,K,T)
    print(test_instance.reward_upperbound)
    r,rr,rrr = 0,0,0
    N = 100000
    instances = [sub_instance(mu,delta,K,T) for i in range(N)]
    t = time.time()
    r = Parallel(n_jobs=-1, verbose=0)(delayed(run_sims_greedy)(instances[i]) for i in range(N))
    rr = Parallel(n_jobs=-1, verbose=0)(delayed(run_sims_thompson)(instances[i]) for i in range(N))
    rrr = Parallel(n_jobs=-1, verbose=0)(delayed(run_sims_dp)(instances[i]) for i in range(N))
    print(np.average(r))
    print(np.average(rr))
    print(np.average(rrr))
    print time.time() - t
