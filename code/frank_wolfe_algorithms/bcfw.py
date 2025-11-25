import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from separable_opt_problem import SeparableOptProblem

class BlockCoordinateFrankWolfe:
    
    def __init__(self, problem:SeparableOptProblem, d_star):
        """
        Args:
            oracles: list of functions, each oracle_i(g) returns argmin_{x in D_i} <g, x>
        """
        self.problem = problem
        self.h_list = problem.h_list
        self.b_eq = problem.b_eq
        self.b_ineq = problem.b_ineq
        #self.oracle_list = problem.oracle_list
        self.n_components = problem.n
        self.d_star = d_star
    
    
    def optimize(self, X_0, max_iter=100, freq_compute_cost=1000, stepsize_strategy="fixed"):
        
        X_k = X_0.copy()


        history = {
            'iteration': [],
            'step_size': [],
            'nb_oracle_calls': [],
            'primal_value': [],
            'infeasibility': [],
            'fw_objective_value': []
        }
        nb_oracle_calls = 0
        
        beta_k_list = np.zeros(self.n_components)
        z_eq_k_list = np.zeros((self.b_eq.shape[0], self.n_components))
        z_ineq_k_list = np.zeros((self.b_ineq.shape[0], self.n_components))
        for i in range(self.n_components):
            beta_k_list[i] = 1/self.n_components * self.problem.h_i(i, X_k[:, i])
            z_eq_k_list[:, i] = 1/self.n_components * self.problem.compute_Ai_eq_dot_x(i, X_k[:, i])
            z_ineq_k_list[:, i] = 1/self.n_components * self.problem.compute_Ai_ineq_dot_x(i, X_k[:, i])
        beta_k = np.sum(beta_k_list)
        z_eq_k = np.sum(z_eq_k_list, axis=1)
        z_ineq_k = np.sum(z_ineq_k_list, axis=1)

        beta_z_dic = {}
        x_dic = {}
        weights_dic = {}
        if not self.problem.is_convex:
            for i in range(self.n_components):
                beta_z_dic[i] = [np.concatenate((np.array([beta_k_list[i]]), z_eq_k_list[:, i], z_ineq_k_list[:, i]))]
                x_dic[i] = [X_k[:, i].copy()]
                weights_dic[i] = [1.0]
                #TODO: should also add check that X_0[:, i] is in the domain of h_i

        for k in range(max_iter):

            #compute gradient
            gamma_k = max(beta_k - self.d_star, 0)
            #v_k = infeasibility_k
            v_eq_k = z_eq_k - self.b_eq
            v_ineq_k = np.clip(z_ineq_k - self.b_ineq, 0, None)

            #pick index at random
            ik = np.random.randint(self.n_components)

            s_ik = self.problem.oracle(ik, gamma_k, v_eq_k, v_ineq_k)[0]
            nb_oracle_calls += 1
            #update cost and infeasibility
            d_ik = s_ik - X_k[:, ik]

            h_ik_sik = self.problem.h_i(ik, s_ik)
            Aeq_ik_sik = self.problem.compute_Ai_eq_dot_x(ik, s_ik)
            Aineq_ik_sik = self.problem.compute_Ai_ineq_dot_x(ik, s_ik)

            if stepsize_strategy == "fixed":
                rho_k = 2 * self.n_components / (k + 2*self.n_components)
            elif stepsize_strategy == "linesearch":
                if np.linalg.norm(d_ik) < 1e-6:
                    #does not matter
                    rho_k = 1
                else:
                    #numerator2 = gamma_k * 1/self.n_components * (self.problem.h_i(ik, X_k[:, ik]) - h_ik_sik) + v_k.dot(1/self.n_components * self.problem.compute_Ai_dot_x(ik, -d_ik))
                    numerator = gamma_k * (beta_k_list[ik] - 1/self.n_components * h_ik_sik)  + v_eq_k.dot( z_eq_k_list[:, ik] - 1/self.n_components * Aeq_ik_sik) + v_ineq_k.dot( z_ineq_k_list[:, ik] - 1/self.n_components * Aineq_ik_sik)
                    #denominator2 = (1/self.n_components * (self.problem.h_i(ik, X_k[:, ik]) - h_ik_sik))**2 + np.linalg.norm(1/self.n_components * self.problem.compute_Ai_dot_x(ik, -d_ik))**2
                    denominator = ((beta_k_list[ik] - 1/self.n_components * h_ik_sik))**2 + np.linalg.norm(z_eq_k_list[:, ik] - 1/self.n_components * Aeq_ik_sik)**2 + np.linalg.norm(z_ineq_k_list[:, ik] - 1/self.n_components * Aineq_ik_sik)**2
                    rho_k = numerator / denominator
                    if np.abs(rho_k) < 1e-6:
                        rho_k = 0
                    if np.abs(numerator) < 1e-8:
                        rho_k = 0
                    rho_k = min(rho_k, 1)
                    assert rho_k >= 0

            #update beta_k and z_k
            beta_k += rho_k * (1/self.n_components * h_ik_sik - beta_k_list[ik])
            beta_k_list[ik] = (1 - rho_k) * beta_k_list[ik] + rho_k * 1/self.n_components * h_ik_sik
            z_eq_k += rho_k  * ( 1/ self.n_components * Aeq_ik_sik - z_eq_k_list[:, ik])
            z_eq_k_list[:, ik] = (1 - rho_k) * z_eq_k_list[:, ik] + rho_k * 1/self.n_components * Aeq_ik_sik
            z_ineq_k += rho_k  * ( 1/ self.n_components * Aineq_ik_sik - z_ineq_k_list[:, ik])
            z_ineq_k_list[:, ik] = (1 - rho_k) * z_ineq_k_list[:, ik] + rho_k * 1/self.n_components * Aineq_ik_sik

            #update iterate
            X_k[:, ik] = (1-rho_k) * X_k[:, ik] + rho_k * s_ik

            if not self.problem.is_convex:
                beta_z_dic[ik].append(np.concatenate((np.array([1/self.n_components * h_ik_sik]), 1/self.n_components * Aeq_ik_sik, 1/self.n_components * Aineq_ik_sik)))
                x_dic[ik].append(s_ik.copy())
                weights_dic[ik] = ((1-rho_k) * np.array(weights_dic[ik])).tolist()
                weights_dic[ik].append(rho_k)
            

            if k%freq_compute_cost == 0 or k== max_iter-1:
                history["iteration"].append(k)
                history['nb_oracle_calls'].append(nb_oracle_calls)
                infeasibility = np.sqrt(np.linalg.norm(z_eq_k - self.b_eq)**2 + np.linalg.norm(np.clip(z_ineq_k - self.b_ineq, 0, None))**2)
                history['infeasibility'].append(infeasibility)
                history['primal_value'].append(beta_k)
                history['fw_objective_value'].append(0.5 * gamma_k**2 + 0.5 * v_eq_k.dot(v_eq_k) + 0.5 * v_ineq_k.dot(v_ineq_k))
                print(f"At iteration {k}, primal value = {beta_k}, infeasibility = {infeasibility}")
        
        #put y_dic and weights as arrays
        if not self.problem.is_convex:
            for i in range(self.n_components):
                beta_z_dic[i] = np.array(beta_z_dic[i]).T
                x_dic[i] = np.array(x_dic[i]).T
                weights_dic[i] = np.array(weights_dic[i])

        #output is concatenation of cost_k and infeasibility_k
        w_K = np.concatenate((np.array([beta_k]), z_eq_k, z_ineq_k))
        return history, X_k, w_K, beta_z_dic, x_dic, weights_dic