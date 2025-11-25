import numpy as np


import numpy as np
import sys
import os

# Add parent directory to path to import SeparableOptProblem
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from separable_opt_problem import SeparableOptProblem
from caratheodory.mnp import MetaCaratheodoryMNPSolve

class DualSubgradient:
    
    def __init__(self, problem: SeparableOptProblem):
        """
        Args:
            problem: SeparableOptProblem instance containing h_list, A_list, b
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
        for i in range(self.n_components):
            #initialize with a feasible primal point
            X[:, i] = self.problem.oracle(i, 1, lbd, mu)[0]

        beta_z_dic = {}
        x_dic = {}
        weights_dic = {}
        beta_z = np.zeros(1 + self.b_eq.shape[0] + self.b_ineq.shape[0])
        if not self.problem.is_convex:
            for i in range(self.n_components):
                h_i_Xi = 1/self.n_components * self.problem.h_i(i, X[:, i])
                Ai_eq_Xi = 1/self.n_components * self.problem.compute_Ai_eq_dot_x(i, X[:, i])
                Ai_ineq_Xi = 1/self.n_components * self.problem.compute_Ai_ineq_dot_x(i, X[:, i])
                beta_z_dic[i] = [np.concatenate((np.array([h_i_Xi]), Ai_eq_Xi, Ai_ineq_Xi))]
                beta_z = np.concatenate((np.array([h_i_Xi]), Ai_eq_Xi, Ai_ineq_Xi))
                x_dic[i] = [X[:, i].copy()]
                weights_dic[i] = [1.0]
        
        for k in range(max_iter):

            #pick index at random
            gk_eq = -self.b_eq.copy()
            gk_ineq = -self.b_ineq.copy()

            for i in range(self.n_components):
                x_ik = self.problem.oracle(i, 1, lbd, mu)[0]
                X[:, i] = (1/(k + 2)) * ((k+1) * X[:, i] + x_ik)

                h_i_Xi = 1/self.n_components * self.problem.h_i(i, x_ik)
                Ai_eq_Xi = 1/self.n_components * self.problem.compute_Ai_eq_dot_x(i, x_ik)
                Ai_ineq_Xi = 1/self.n_components * self.problem.compute_Ai_ineq_dot_x(i, x_ik)

                gk_eq += Ai_eq_Xi
                gk_ineq += Ai_ineq_Xi

                if not self.problem.is_convex:
                    beta_z_dic[i].append(np.concatenate((np.array([h_i_Xi]), Ai_eq_Xi, Ai_ineq_Xi)))
                    x_dic[i].append(x_ik.copy())
                    weights_dic[i] = ((k+1)/(k+2) * np.array(weights_dic[i])).tolist() + [1/(k+2)]

            nb_oracle_calls += self.n_components
            #take dual step
            alpha_k = alpha_bar/np.sqrt(k+1)
            lbd += alpha_k * gk_eq
            mu += alpha_k * gk_ineq
            mu = np.maximum(mu, 0)  #project onto nonnegative orthant

            lbd_avg = (1/(k+2)) * ((k+1)*lbd_avg + lbd)
            mu_avg = (1/(k+2)) * ((k+1)*mu_avg + mu)


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
        
        history['total_nb_oracle_calls'] = nb_oracle_calls

        X_sol = X.copy()
        if not self.problem.is_convex:
            #put y_dic and weights as arrays
            # and compute beta_z
            beta_z = np.zeros(1 + self.b_eq.shape[0] + self.b_ineq.shape[0])
            for i in range(self.n_components):
                beta_z_dic[i] = np.array(beta_z_dic[i]).T
                x_dic[i] = np.array(x_dic[i]).T
                weights_dic[i] = np.array(weights_dic[i])
                beta_z += beta_z_dic[i].dot(weights_dic[i])
            #call Caratheodory algorithm
            carathodory_mnp_solver = MetaCaratheodoryMNPSolve(y_dic=beta_z_dic, 
                                                              x_dic=x_dic,
                                                              weights_dic=weights_dic, verbose=True)

            caratheodory_output = carathodory_mnp_solver.solve(z=beta_z, 
                                                                nb_indices_considered=2*(self.problem.b_eq.shape[0] + self.problem.b_ineq.shape[0]),
                                                                T=self.n_components)
            
            X_sol = self.problem.build_final_solution_from_caratheodory_output(caratheodory_output)

            cost_final = self.problem.h(X_sol)
            infeasibility_final = np.linalg.norm(self.problem.compute_infeasibility(X_sol))
            print(f"After Caratheodory MNP: final cost = {cost_final}, infeasibility = {infeasibility_final}")


        return history, X_sol


def solve_dual_gd(prob, eta="1/k", max_iter=1000, solve_contracted_problem=False, verbose=True):

    lambda_k = np.zeros(prob.m)
    fmax = -np.inf

    if solve_contracted_problem:
        b = prob.b_bar.copy()
    else:
        b = prob.b.copy()
    
    for k in range(max_iter):
        grad_k = b.copy()
        fk = b.dot(lambda_k)

        for i in range(prob.n):
            #Ai = prob.construct_Ai_matrix(i)
            #fi, gi = prob.f_conjugate_i(i, - Ai.T.dot(lambda_k))
            fi, gi = prob.f_conjugate_i(i, - prob.compute_AiT_dot_g(i, lambda_k))
            
            grad_k += - prob.compute_Ai_dot_y(i, gi) 
            fk += fi

        if eta=="1/k":
            etak = 1/(k+1)
        elif eta == "1/sqrt(k)":
            etak = 1/np.sqrt(k+1)
        else:
            etak = eta
        lambda_k = lambda_k - etak * grad_k
        lambda_k = np.clip(lambda_k, 0, None)
        if verbose:
            if k%100 == 0:
                print("    ", k, -fk, np.linalg.norm(grad_k), np.linalg.norm(lambda_k))

        fmax = -fk if -fk > fmax else fmax

    return fmax

            