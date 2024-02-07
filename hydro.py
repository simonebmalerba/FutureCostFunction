import numpy as np
import PULP
from itertools import *
class hydro:
    def __init__(self,
                c = np.array([1.0]),
                u_max = [15.,15.],
                v_max = [30.,30.],
                g_bar = 15.,
                ρ = [1.,1.],**kwargs):
        #Set machine parameters (cost for thermal generation,turbin capacities(maximum energy generation of hydroplant, MW),
        #maximum storage(l), efficiency(MW/l))
        self.c,self.u_max,self.v_max,self.g_bar,self.ρ = c,u_max,v_max,g_bar,ρ
        #Discretization of storage values
        if 'S' in kwargs:
            S = kwargs.get('S')
        else:
            S = np.arange(0,1.1,0.1)
        self.S = S
    def solve1H(self,π,q,d,f,vt):
        #Solve the one stage hydrothermal dispatch, given a flow scenario,initial storage value and piecewise
        #description of future cost function
        stage1 = pulp.LpProblem("1stageHD", pulp.LpMinimize)
        #Number of generators
        Ng = len(self.c)
        #Variable(s) of generators
        g = pulp.LpVariable.dicts('g',(i for i in range(0,Ng)),lowBound=0,upBound=self.g_bar)
        #Variables of Hydroplant
        u0 = pulp.LpVariable('u0',lowBound=0,upBound=self.u_max[0])
        u1 = pulp.LpVariable('u1',lowBound=0,upBound=self.u_max[1])
        #Variables of future reservoir
        v_f0 = pulp.LpVariable('v_f0',lowBound=0,upBound =self.v_max[0])
        v_f1 = pulp.LpVariable('v_f1',lowBound=0,upBound = self.v_max[1])
        #Auxiliar variable(piecewise linear function) == FCF
        fcf = pulp.LpVariable('fcf',lowBound=0,upBound=None)
        #Cost function
        stage1 +=  pulp.lpSum(self.c[i]*g[i] for i in range(0,Ng)) + fcf ,'z'
        #Demand balance (price.pi will be the correspoinding lagrange multiplier, the spot price)
        price = pulp.lpSum(g[i] for i in range(0,Ng)) + u1 +u0 == d
        stage1 += price
        #Flow balance (and corresponding lagrange multipliers)
        st0 = max(vt[0]+f[0]-self.v_max[0],0)
        st1 = max(vt[1]+f[1]-self.v_max[1],0)
        pi0 = 1./self.ρ[0] *u0 + v_f0 == vt[0] +f[0]-st0
        pi1 = 1./self.ρ[1] *u1 + v_f1 == vt[1] +f[1]-st1
        stage1 += pi0
        stage1 += pi1
        #Auxiliar equations for fcf (minimization of piecewise convex linear function)
        for i in range(len(self.S)):
            for j in range(len(self.S)):
                stage1 += fcf - π[(i,j)][0]*v_f0 - q[(i,j)][0] >= 0
                stage1 += fcf - π[(i,j)][1]*v_f1 - q[(i,j)][1] >= 0
        stage1.solve()
        #Optimal value of cost function
        fut_cost = pulp.value(stage1.objective)
        #Constant term for piecewise function
        q0= fut_cost- pi0.pi*vt[0]
        q1 =  fut_cost- pi1.pi*vt[1]
        return fut_cost,pi0.pi,pi1.pi,q0,q1

    def compute_fcf(self,T_max = 10,
                         F = np.arange(2,5,0.5),**kwargs):
        #Return list of future cost function matrix from T to 0, for a demand d(t),
        #averaging over F inflow scenarios with probability fprob (default == uniform)
        if 'fprob' in kwargs:
            fprob = kwargs.get('fprob')
        else:
             fprob = (1./len(F)**2)*np.ones(len(F)**2)
        if 'd' in kwargs:
            d = kwargs.get('d')
        else:
            d = 30*np.ones(T_max+1)

        c,u_max,v_max,g_bar,ρ = self.c,self.u_max,self.v_max,self.g_bar,self.ρ

        T = np.arange(T_max+1,-1,-1); S0 = self.S*v_max[0]; S1 = self.S*v_max[1]
        self.S0,self.S1,self.F = S0,S1,F
        #List of future cost function, dictionary for the parameters of the piecewise fcf
        α,πn,qn = [], {},{}

        for i,j in product(range(len(self.S)),range(len(self.S))):
            πn[(i,j)],qn[(i,j)] = np.array([0.,0.]),np.array([0.,0.])
        T = np.arange(T_max+1,-1,-1)
        #Computation of fcf for each time step
        for t in T[1:]:
            αt = np.matrix(np.zeros((len(self.S),len(self.S))))
            π,q = dict(πn),dict(qn)
            πn,qn = {},{}
            for i,j in product(range(len(self.S)),range(len(self.S))):
                πn[(i,j)],qn[(i,j)] = np.array([0.,0.]),np.array([0.,0.])
            #Computation of fcf for each initial state
            for row,vi1 in enumerate(S0):
                for col,vi2 in enumerate(S1):
                    #Average over possible inflow scenarios
                    for index, [f1,f2] in enumerate(product(F,F)):
                        fut_cost,pi0,pi1,q0,q1 =self.solve1H(π,q,d[t],[f1,f2],[vi1,vi2])
                        αt[row,col]  += fprob[index]*fut_cost
                        πn[(row,col)]  += fprob[index]*np.array([pi0,pi1])
                        qn[(row,col)]  += fprob[index] *np.array([q0,q1])
            α.append(αt)
            print("FCF computed at time", t)
            self.fcf = α
        return α
