
import numpy as np
import sys
import os
# Add parent directory to path to import SeparableOptProblem
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from separable_opt_problem import SeparableOptProblem
from utils import insert_column

class StochasticDualSubgradient:
    
    def __init__(self, problem: SeparableOptProblem):
        """
        Args:
            problem: SeparableOptProblem instance
        """
        self.problem = problem
        self.n_components = problem.n
        self.b_eq = problem.b_eq
        self.b_ineq = problem.b_ineq
        #self.A_list = separable_opt_problem.A_list
    
    def optimize(self, lbd_0, mu_0, max_iter=100, freq_compute_dual=1000, alpha_bar=1):
        
        lbd = lbd_0.copy()
        lbd_avg = lbd_0.copy()
        mu = mu_0.copy()
        mu_avg = mu_0.copy()
        if np.any(mu_0 < 0):
            raise ValueError("Initial dual variable for inequality constraints mu_0 must be nonnegative")
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
        X = np.zeros((self.problem.get_di(0), self.n_components)) #assuming here all x_i's have the same dimension
        #for i in range(self.n_components):
            #initialize with a feasible primal point
        #    X[:, i] = np.zeros()
        beta_z_dic = {}
        x_dic = {}
        weights_dic = {}
        if not self.problem.is_convex:
            for i in range(self.n_components):
                beta_z_dic[i] = np.zeros((1 + self.problem.m, int(max_iter / self.n_components) + 1))
                x_dic[i] = np.zeros((self.problem.get_di(i), int(max_iter / self.n_components) + 1 ))
        index_counters = np.zeros(self.n_components, dtype=int)
        
        for k in range(max_iter-1):

            #pick index at random
            ik = np.random.randint(self.n_components)

            x_ik = self.problem.oracle(ik, 1, lbd, mu)[0]
            nb_oracle_calls += 1
            X[:, ik] = (1/(index_counters[ik] + 1)) * (index_counters[ik] * X[:, ik] + x_ik)
            index_counters[ik] += 1

            #gk_eq = self.problem.compute_Ai_eq_dot_x(ik, x_ik) / self.n_components - self.b_eq/self.n_components
            #gk_ineq = self.problem.compute_Ai_ineq_dot_x(ik, x_ik) / self.n_components - self.b_ineq/self.n_components
            Aeq_ik_sik = self.problem.compute_Ai_eq_dot_x(ik, x_ik)
            Aineq_ik_sik = self.problem.compute_Ai_ineq_dot_x(ik, x_ik)
            gk_eq = Aeq_ik_sik  - self.b_eq
            gk_ineq = Aineq_ik_sik - self.b_ineq
            alpha_k = alpha_bar/np.sqrt(k+1)
            lbd += alpha_k * gk_eq
            mu += alpha_k * gk_ineq
            mu = np.maximum(mu, 0)  #project onto nonnegative orthant

            lbd_avg = (1/(k+2)) * ((k+1)*lbd_avg + lbd)
            mu_avg = (1/(k+2)) * ((k+1)*mu_avg + mu)

            #update dictionaries keeping track of values
            if not self.problem.is_convex:
                h_ik_sik = self.problem.h_i(ik, x_ik)
                res = insert_column(beta_z_dic[ik], 
                                    np.concatenate((np.array([1/self.n_components * h_ik_sik]), 1/self.n_components * Aeq_ik_sik, 1/self.n_components * Aineq_ik_sik)),
                                    index=index_counters[ik]-1,
                                    expand_size=int(0.1 * beta_z_dic[ik].shape[1]))
                if res is not None:
                    beta_z_dic[ik] = res.copy()

                #do the same with x_dic
                res = insert_column(x_dic[ik], 
                                    x_ik,
                                    index=index_counters[ik]-1,
                                    expand_size=int(0.1 * beta_z_dic[ik].shape[1]))
                if res is not None:
                    x_dic[ik] = res.copy()
                #beta_z_dic[ik].append(np.concatenate((np.array([1/self.n_components * h_ik_sik]), 1/self.n_components * Aeq_ik_sik, 1/self.n_components * Aineq_ik_sik)))
                #x_dic[ik].append(x_ik.copy())


            if k%freq_compute_dual == 0:
                history["iteration"].append(k)
                history['dual_value'].append(self.problem.compute_dual(lbd_avg, mu_avg))
                history['nb_oracle_calls'].append(nb_oracle_calls)
                primal_cost = self.problem.h(X)
                infeasibility = np.linalg.norm(self.problem.compute_infeasibility(X))
                history['primal_value'].append(primal_cost)
                history['infeasibility'].append(infeasibility)
                print(f"At iteration {k}, dual value = {history['dual_value'][-1]}")
                print(f"   Primal value = {primal_cost}, infeasibility = {infeasibility}")
        
        #Now take a last deterministic step
        last_k = max_iter - 1
        gk_eq = -self.b_eq.copy()
        gk_ineq = -self.b_ineq.copy()
        for i in range(self.n_components):
            x_ik = self.problem.oracle(i, 1, lbd, mu)[0]
            X[:, i] = (1/(index_counters[i] + 1)) * (index_counters[i] * X[:, i] + x_ik)
            index_counters[i] += 1

            h_i_Xi = 1/self.n_components * self.problem.h_i(i, x_ik)
            Ai_eq_Xi = 1/self.n_components * self.problem.compute_Ai_eq_dot_x(i, x_ik)
            Ai_ineq_Xi = 1/self.n_components * self.problem.compute_Ai_ineq_dot_x(i, x_ik)

            gk_eq += Ai_eq_Xi
            gk_ineq += Ai_ineq_Xi

            if not self.problem.is_convex:
                res = insert_column(beta_z_dic[i], 
                                    np.concatenate((np.array([h_i_Xi]), Ai_eq_Xi, Ai_ineq_Xi)),
                                    index=index_counters[i]-1,
                                    expand_size=1)
                if res is not None:
                    beta_z_dic[i] = res.copy()

                #do the same with x_dic
                res = insert_column(x_dic[i], 
                                    x_ik,
                                    index=index_counters[i]-1,
                                    expand_size=1)
                if res is not None:
                    x_dic[i] = res.copy()

                #beta_z_dic[i].append(np.concatenate((np.array([h_i_Xi]), Ai_eq_Xi, Ai_ineq_Xi)))
                #x_dic[i].append(x_ik.copy())

        nb_oracle_calls += self.n_components
        #take dual step
        alpha_k = alpha_bar/np.sqrt(last_k+1)
        lbd += alpha_k * gk_eq
        mu += alpha_k * gk_ineq
        mu = np.maximum(mu, 0)  #project onto nonnegative orthant

        lbd_avg = (1/(last_k+2)) * ((last_k+1)*lbd_avg + lbd)
        mu_avg = (1/(last_k+2)) * ((last_k+1)*mu_avg + mu)

        #finally compute weights dictionnary for the nonconvex case, which is simply the average of the index_counter[i] times we sampled index i
        for i in range(self.n_components):
            weights_dic[i] = (1/index_counters[i] * np.ones(int(index_counters[i])))
            beta_z_dic[i] = beta_z_dic[i][:, :index_counters[i]]
            x_dic[i] = x_dic[i][:, :index_counters[i]]
            #format the other dic as arrays
            #beta_z_dic[i] = np.array(beta_z_dic[i]).T
            #x_dic[i] = np.array(x_dic[i]).T
        
        history['total_nb_oracle_calls'] = nb_oracle_calls
        history['index_counters'] = index_counters.tolist()
        return history, X, beta_z_dic, x_dic, weights_dic