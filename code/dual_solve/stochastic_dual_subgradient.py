
import numpy as np
import sys
import os

# Add parent directory to path to import SeparableOptProblem
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from separable_opt_problem import SeparableOptProblem

class StochasticDualSubgradient:
    
    def __init__(self, problem: SeparableOptProblem):
        """
        Args:
            problem: SeparableOptProblem instance containing h_list, A_list, b
        """
        self.problem = problem
        self.n_components = problem.n
        self.b = problem.b
        #self.A_list = separable_opt_problem.A_list
    
    def optimize(self, lbd_0, max_iter=100, freq_compute_dual=1000, alpha_bar=1):
        
        lbd = lbd_0.copy()
        lbd_avg = lbd_0.copy()
        history = {
            'dual_value': [],
            'iteration': [],
            'step_size': [],
            'nb_oracle_calls': [],
            'primal_value': [],
            'infeasibility': []
        }
        nb_oracle_calls = 0

        #create matrix of primal candidates (this is only if we want to keep track of the primal solution, or to help initialize the block Frank Wolfe algorithm)
        X = np.zeros((self.problem.A_list[0].shape[1], self.n_components))
        for i in range(self.n_components):
            #initialize with a feasible primal point
            X[:, i] = self.problem.oracle(i, 1, lbd)[0]
        index_counters = np.zeros(self.n_components)
        
        for k in range(max_iter):

            #pick index at random
            ik = np.random.randint(self.n_components)

            x_ik = self.problem.oracle(ik, 1, lbd)[0]
            nb_oracle_calls += 1
            X[:, ik] = (1/(index_counters[ik] + 1)) * (index_counters[ik] * X[:, ik] + x_ik)
            index_counters[ik] += 1

            gk = self.problem.compute_Ai_dot_x(ik, x_ik) / self.n_components - self.b/self.n_components
            alpha_k = alpha_bar/np.sqrt(k+1)
            lbd += alpha_k * gk
            #print(lbd, type(lbd_avg))

            lbd_avg = (1/(k+2)) * ((k+1)*lbd_avg + lbd)

            if k%freq_compute_dual == 0:
                history["iteration"].append(k)
                history['dual_value'].append(self.problem.compute_dual(lbd_avg))
                history['nb_oracle_calls'].append(nb_oracle_calls)
                primal_cost = self.problem.h(X)
                infeasibility = np.linalg.norm(self.problem.compute_infeasibility(X))
                history['primal_value'].append(primal_cost)
                history['infeasibility'].append(infeasibility)
                print(f"At iteration {k}, dual value = {history['dual_value'][-1]}")
                print(f"   Primal value = {primal_cost}, infeasibility = {infeasibility}")
        
        history['total_nb_oracle_calls'] = nb_oracle_calls
        history['index_counters'] = index_counters
        return history, X
    
    