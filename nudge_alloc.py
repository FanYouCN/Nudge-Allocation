from pulp import *
import math

class nudge_alloc_problem:
    '''
    nudge allocation instance
    '''
    def __init__(self, S, M, N, H, I, b, A, R, C, mean, std):
        '''
        Initialize instance data
        '''
        self.segments = range(1, S+1)
        self.segments_size = M
        self.nudges = range(1,N+1)
        self.sn_index = [] # Dictionary of the index set for every (segment-nudge) pair
        for s in self.segments:
            for n in self.nudges:
                self.sn_index.append((s,n))
        self.eligibiliy = H
        self.resources = range(1,I+1)
        self.resource_capacities = b
        self.nudge_resource_consumption = A
        self.reward = R
        self.cost = C
        self.mu = mean # True distribution mean
        self.sigma = std # True distribution std

        self.LP = LpProblem("NudgeAllocate", LpMaximize) # Integer Program associated with Instance
        self.k = LpVariable.dicts("Allocation", (self.segments,self.nudges), 0, None, LpInteger)
        for sn in self.sn_index:
            self.LP += self.k[sn[0]][sn[1]] <= self.eligibiliy[sn] * self.segments_size[sn[0]], "Eligibility" + str(sn)
        for s in self.segments:
            self.LP += lpSum([self.k[s][n] for n in self.nudges]) <= self.segments_size[s], "Population Size" + str(s)
        for i in self.resources:
            self.LP += lpSum([self.nudge_resource_consumption[i,sn[0],sn[1]]*self.k[sn[0]][sn[1]] for sn in self.sn_index]) <= self.resource_capacities[i], "resource constraints" + str(i)

    def allocate(self, X):
        # Initialize objective function for given X then solve
        self.LP += lpSum([self.k[sn[0]][sn[1]]*(self.reward[sn]*X[sn] - self.cost[sn])] for sn in self.sn_index), "total improvement"
        self.LP.solve()
        K = {}
        for s in self.segments:
            for n in self.nudges:
                K[s,n] = self.k[s][n].varValue
        return K

    def min_deviation(self, T):
        # Solve for K that minimizes absolute deviation
        this_LP = LpProblem("MinDeviation", LpMinimize)
        k = LpVariable.dicts("Allocation", (self.segments,self.nudges), 0, None, LpInteger)
        t = LpVariable.dicts("artificial", self.nudges, None, None, cat='Continuous')
        for sn in self.sn_index:
            this_LP += k[sn[0]][sn[1]] <= self.eligibiliy[sn] * self.segments_size[sn[0]], "Eligibility" + str(sn)
        for s in self.segments:
            this_LP += lpSum([k[s][n] for n in self.nudges]) <= self.segments_size[s], "Population Size" + str(s)
        for i in self.resources:
            this_LP += lpSum([self.nudge_resource_consumption[i,sn[0],sn[1]] * k[sn[0]][sn[1]] for sn in self.sn_index]) <= self.resource_capacities[i], "resource constraints" + str(i)
        for n in self.nudges:
            this_LP += t[n] >= lpSum([k[s][n] for s in self.segments]) - T[n], "ac1" + str(n)
        for n in self.nudges:
            this_LP += t[n] >= T[n] - lpSum([k[s][n] for s in self.segments]), "ac2" + str(n)
        this_LP += lpSum([t[n] for n in self.nudges]), "total deviation"
        this_LP.solve()
        K = {}
        for s in self.segments:
            for n in self.nudges:
                K[s,n] = k[s][n].varValue
        return K

    def upper_bound(self):
        # Assuming X is the known true distribution mean, solve for the upper bound
        self.LP += lpSum([self.k[sn[0]][sn[1]]*(self.reward[sn]*self.mu[sn] - self.cost[sn])] for sn in self.sn_index), "max expected reward"
        self.LP.solve()
        v = value(self.LP.objective)
        for vv in self.LP.variables():
            if vv.varValue > 0:
                print(vv.name, "=", vv.varValue)
        return v*100
