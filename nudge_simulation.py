from nudge_alloc import nudge_alloc_problem
from random import *
from pulp import *
import numpy as np

class thompson:
    def __init__(self, nudge_instance):
        self.instance = nudge_instance
        self.beta = 0.99
        self.rho = {}
        for sn in nudge_instance.sn_index: # Calculate true precision
            self.rho[sn] = 1/(nudge_instance.sigma[sn] * nudge_instance.sigma[sn])

    def thompson_sim(self):
        discount = 1
        # Prior Belief on Mean effect
        mu_hat = {}
        tau = {}
        total_reward = 0
        for sn in self.instance.sn_index:
            mu_hat[sn] = 1.2
            tau[sn] = 0.1 # Initialize prior belief
        for t in range(2000):
            # if t%100 ==0:
            #     print t
            X = {}
            for sn in self.instance.sn_index:
                X[sn] = gauss(mu_hat[sn], 1/np.sqrt(tau[sn])) # Sample X
            K = self.instance.allocate(X)
            observed_X = {}
            for sn in self.instance.sn_index:
                # Simulate and observe
                if K[sn] > 0:
                    observed_X[sn] = sum([gauss(self.instance.mu[sn], self.instance.sigma[sn]) for i in range(int(K[sn]))]) / K[sn]
            for sn in self.instance.sn_index:
                if K[sn] > 0:
                    total_reward += discount *K[sn]*(self.instance.reward[sn] * observed_X[sn] - self.instance.cost[sn])
            discount *= self.beta
            for sn in self.instance.sn_index:
                # Update belief
                if K[sn] > 0:
                    mu_hat[sn] = (tau[sn]*mu_hat[sn] + K[sn]*self.rho[sn]*observed_X[sn])/(tau[sn] + K[sn]*self.rho[sn])
                    tau[sn] += K[sn]*self.rho[sn]
            if t % 100 == 0:
                print t
                for v in self.instance.LP.variables():
                    if v.varValue > 0:
                        print(v.name, "=", v.varValue)
        return total_reward

class randomized_LP:
    '''
    Maintain and update belief, solve repeated samples, solve for least deviation
    '''
    def __init__(self, nudge_instance):
        self.instance = nudge_instance
        self.beta = 0.99
        self.rho = {}
        for sn in nudge_instance.sn_index: # Calculate true precision
            self.rho[sn] = 1/(nudge_instance.sigma[sn] * nudge_instance.sigma[sn])

    def randomized_LP_sim(self):
        discount = 1
        mu_hat = {}
        tau = {}
        total_reward = 0
        for sn in self.instance.sn_index:
            mu_hat[sn] = 1.2
            tau[sn] = 0.1 # Initialize prior belief
        for t in range(2000):
            K = []
            for j in range(10):
                X = {}
                for sn in self.instance.sn_index:
                    X[sn] = gauss(mu_hat[sn], 1/np.sqrt(tau[sn]))
                K.append(self.instance.allocate(X))
            k_bar = {}
            for sn in self.instance.sn_index:
                k_bar[sn] = sum([k[sn] for k in K]) / len(K)
            T = {}
            for n in self.instance.nudges:
                T[n] = sum([k_bar[s,n] for s in self.instance.segments])
            greedy_K = self.instance.min_deviation(T)
            observed_X = {}
            for sn in self.instance.sn_index:
                if greedy_K[sn] > 0:
                    observed_X[sn] = sum([gauss(self.instance.mu[sn], self.instance.sigma[sn]) for i in range(int(greedy_K[sn]))]) / greedy_K[sn]
            for sn in self.instance.sn_index:
                if greedy_K[sn] > 0:
                    total_reward += discount * greedy_K[sn] * (self.instance.reward[sn] * observed_X[sn] - self.instance.cost[sn])
            discount *= self.beta
            for sn in self.instance.sn_index:
                if greedy_K[sn] > 0:
                    mu_hat[sn] = (tau[sn]*mu_hat[sn] + greedy_K[sn]*self.rho[sn]*observed_X[sn])/(tau[sn] + greedy_K[sn]*self.rho[sn])
                    tau[sn] += greedy_K[sn]*self.rho[sn]
            if t%100 == 0:
                print t
                print T
                for v in self.instance.LP.variables():
                    if v.varValue > 0:
                        print(v.name, "=", v.varValue)
        return total_reward

if __name__ == '__main__':
        S = 3
        M = {1:10,2:5,3:8}
        N = 5
        H = {}
        for s in range(1,4):
            for n in range(1,6):
                H[s,n] = 1
        H[2,4] = 0
        H[1,5] = 0
        H[3,2] = 0

        I = 2
        b = {1:20, 2:30}
        A = {}
        import random
        # random.seed(1)
        for i in range(1,3):
            for s in range(1,4):
                for n in range(1,6):
                    A[i,s,n] = random.randint(2,5)
        R = {}
        C = {}
        for s in range(1,4):
            for n in range(1,6):
                R[s,n] = random.randint(2,6)
                C[s,n] = random.randint(3,8)
        X = {}
        for s in range(1,4):
            for n in range(1,6):
                X[s,n] = random.randint(1,3)
        mean = {}
        std = {}
        for s in range(1,4):
            for n in range(1,6):
                mean[s,n] = random.uniform(0.5,2)
                std[s,n] = random.uniform(0.2,0.4)

        toy_nudge = nudge_alloc_problem(S,M,N,H,I,b,A,R,C,mean,std)
        # print toy_nudge.upper_bound()
        toy_thompson = thompson(toy_nudge)
        toy_RLP = randomized_LP(toy_nudge)
        print toy_thompson.thompson_sim()
        print toy_RLP.randomized_LP_sim()
