import sys
import os
import numpy as np
from scipy.optimize import linprog

# Add parent directory to path to import SeparableOptProblem
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code'))
from separable_opt_problem import NonConvexSeparableOptProblem  # type: ignore


class SCProblem(NonConvexSeparableOptProblem):
    """
    Supply Chain problem as a special case of SeparableOptProblem.
    """

    def __init__(self, K, r, D, I, beta):
        """
        Initialize the SC problem.

        Parameters
        ----------
        """
        self.K = K
        self.r = r
        self.D = D
        self.I = I
        self.beta = beta
        n = K.shape[0]
        if r.shape[0] != n:
            raise ValueError(f"dimension of K and r don't match, got {K.shape[0]} and r.shape[0] = {r.shape[0]}")
        if D.shape[0] != n:
            raise ValueError(f"dimension of K and D don't match, got {K.shape[0]} and D.shape[0] = {D.shape[0]}")
        self.m = D.shape[1]
        if r.shape[1] != self.m:
            raise ValueError(f"Dimension of D and r don't match, got {D.shape} and = {r.shape}")
        if I.shape[0] != self.m:
            raise ValueError(f"Dimension of I and D don't match, got {I.shape[0]} and D.shape[1] = {D.shape[1]}")

        # Create h_list from c_list (linear functions)
        h_list = []
        A_list = []
        Ai = np.zeros((self.m, 1+self.m))
        Ai[:, 1:] = np.eye(self.m)
        for i in range(n):
            c_i = np.zeros(1+self.m)
            c_i[0] = - K[i]
            c_i[1:] = -r[i, :]/D[i, :]
            def h_i(x, c=c_i):
                return c @ x
            h_list.append(h_i)
            A_list.append(Ai)

        b = 1/n * I 
        

        # Call parent constructor (is_convex=True for LP)
        super().__init__(n=n, h_list=h_list, A_list=A_list, b=b)


    def oracle(self, i, gamma, g):
        #two cases: either the first term of the argmin is 0 or 1
        #we must only solve an LP if it is 1
        #set up the LP for that case
        #c_i = np.zeros(self.m)
        c_i = - gamma * self.r[i, :]/self.D[i, :] + g
        A_ub = -np.ones((1, self.m))
        b_ub = -self.beta * np.sum(self.D[i, :])
        #define bounds for each variable
        x_bounds = [(0, self.D[i, j]) for j in range(self.m)]
        res = linprog(c=c_i, A_ub=A_ub, b_ub=b_ub, bounds=x_bounds, method='highs')
        if - gamma * self.K[i] + res.fun < 0:
            #then optimal to set first term to 1
            return np.concatenate((np.array([1.0]), res.x)), -gamma *self.K[i] + res.fun
        else:
            #optimal to set first term to 0
            return np.zeros(1 + self.m), 0.0
    
    #overwrite compute_Ai_dot_x for more efficiency
    def compute_Ai_dot_x(self, i, x_i):
        return x_i[1:]
    
    def build_final_solution_from_caratheodory_output(self, caratheodory_output):
        X_final = np.zeros((1+self.m, self.n))
        y_dic = caratheodory_output.y_dic
        nb_trivial_cvx_comb = 0
        for i in range(self.n):
            X_final[:, i] = y_dic[i][0] @ y_dic[i][1]
            if y_dic[i][0].shape[1] == 1:
                X_final[:, i] = y_dic[i][0][:, 0]
                nb_trivial_cvx_comb += 1
            else:
                X_final[:, i] = np.zeros(1+self.m)

        print(f"Number of blocks with trivial convex combination in Caratheodory: {nb_trivial_cvx_comb} out of {self.n}")
        return X_final