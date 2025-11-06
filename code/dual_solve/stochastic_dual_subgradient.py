
import numpy as np

class StochasticDualSubgradient:
    
    def __init__(self, h_list, 
                 A_list, b,
                 oracle_list):
        """
        Args:
            oracle_list: list of oracles, each oracle_i(gamma, g) returns argmin_{x in X_i} gamma * h_i(x) + g^T A_i x
        """
        self.h_list = h_list
        self.A_list = A_list
        self.b = b
        self.oracle_list = oracle_list
        self.n_components = len(h_list)
    
    def compute_dual(self, lbd):
        dual_value = - lbd.dot(self.b)
        for i in range(self.n_components):
            dual_value += 1/self.n_components * self.oracle_list[i](1, lbd)[1]
        return dual_value
    
    def compute_primal(self, X):
        res = 0
        for i in range(self.n_components):
            res += self.h_list[i](X[:, i])
        return res/self.n_components
    
    def compute_infeasibility(self, X):
        infeasibility = -self.b
        for i in range(self.n_components):
            infeasibility += 1/self.n_components * self.A_list[i] @ X[:, i]
        return infeasibility
    
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
        X = np.zeros((self.A_list[0].shape[1], self.n_components))
        for i in range(self.n_components):
            #initialize with a feasible primal point
            X[:, i] = self.oracle_list[i](1, lbd)[0]
        index_counters = np.zeros(self.n_components)
        
        for k in range(max_iter):

            #pick index at random
            ik = np.random.randint(self.n_components)

            x_ik = self.oracle_list[ik](1, lbd)[0]
            nb_oracle_calls += 1
            X[:, ik] = (1/(index_counters[ik] + 1)) * (index_counters[ik] * X[:, ik] + x_ik)
            index_counters[ik] += 1

            gk = self.A_list[ik] @ x_ik / self.n_components - self.b/self.n_components
            alpha_k = alpha_bar/np.sqrt(k+1)
            lbd += alpha_k * gk
            #print(lbd, type(lbd_avg))

            lbd_avg = (1/(k+2)) * ((k+1)*lbd_avg + lbd)

            if k%freq_compute_dual == 0:
                history["iteration"].append(k)
                history['dual_value'].append(self.compute_dual(lbd_avg))
                history['nb_oracle_calls'].append(nb_oracle_calls)
                primal_cost = self.compute_primal(X)
                infeasibility = np.linalg.norm(self.compute_infeasibility(X))
                history['primal_value'].append(primal_cost)
                history['infeasibility'].append(infeasibility)
                print(f"At iteration {k}, dual value = {history['dual_value'][-1]}")
                print(f"   Primal value = {primal_cost}, infeasibility = {infeasibility}")
        
        history['total_nb_oracle_calls'] = nb_oracle_calls
        history['index_counters'] = index_counters
        return history, X
    
    