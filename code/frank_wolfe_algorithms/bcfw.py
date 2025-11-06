import numpy as np

class BlockCoordinateFrankWolfe:
    
    def __init__(self, h_list, 
                 A_list, b,
                 oracle_list,
                 d_star):
        """
        Args:
            oracles: list of functions, each oracle_i(g) returns argmin_{x in D_i} <g, x>
        """
        self.h_list = h_list
        self.A_list = A_list
        self.b = b
        self.oracle_list = oracle_list
        self.n_components = len(h_list)
        self.d_star = d_star
    
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
        
        cost_k = self.compute_primal(X_k) 
        infeasibility_k = self.compute_infeasibility(X_k)

        for k in range(max_iter):

            #compute gradient
            gamma_k = max(cost_k - self.d_star, 0)
            v_k = infeasibility_k

            #pick index at random
            ik = np.random.randint(self.n_components)

            s_ik = self.oracle_list[ik](gamma_k, v_k)[0]
            nb_oracle_calls += 1
            #update cost and infeasibility
            d_ik = s_ik - X_k[:, ik]

            if stepsize_strategy == "fixed":
                rho_k = 2 * self.n_components / (k + 2*self.n_components)
            elif stepsize_strategy == "linesearch":
                if np.linalg.norm(d_ik) < 1e-6:
                    #does not matter
                    rho_k = 1
                else:
                    #numerator = gamma_k * 1/self.n_components * self.c_list[ik]@(-d_ik) + v_k.dot(1/self.n_components * self.A_list[ik] @ (-d_ik))
                    numerator = gamma_k * 1/self.n_components * (self.h_list[ik](X_k[:, ik]) - self.h_list[ik](s_ik)) + v_k.dot(1/self.n_components * self.A_list[ik] @ (-d_ik))
                    #denominator = (1/self.n_components * self.c_list[ik]@(-d_ik))**2 + np.linalg.norm(1/self.n_components * self.A_list[ik] @ (-d_ik))**2
                    denominator = (1/self.n_components * (self.h_list[ik](X_k[:, ik]) - self.h_list[ik](s_ik)))**2 + np.linalg.norm(1/self.n_components * self.A_list[ik] @ (-d_ik))**2
                    rho_k = numerator / denominator
                    if np.abs(rho_k) < 1e-6:
                        rho_k = 0
                    rho_k = min(rho_k, 1)
                    #print(rho_k)
                    assert rho_k >= 0

            #update cost and infeasibility
            cost_k += rho_k / self.n_components * (self.h_list[ik](s_ik) - self.h_list[ik](X_k[:, ik]))
            infeasibility_k += rho_k / self.n_components * self.A_list[ik] @ d_ik

            #update iterate
            X_k[:, ik] = (1-rho_k) * X_k[:, ik] + rho_k * s_ik
            

            if k%freq_compute_cost == 0 or k== max_iter-1:
                history["iteration"].append(k)
                history['nb_oracle_calls'].append(nb_oracle_calls)
                history['infeasibility'].append(np.linalg.norm(infeasibility_k))
                history['primal_value'].append(cost_k)
                history['fw_objective_value'].append(0.5 * gamma_k**2 + 0.5 * infeasibility_k.dot(infeasibility_k))
                print(f"At iteration {k}, primal value = {cost_k}, infeasibility = {np.linalg.norm(infeasibility_k)}")
        
        return history, X_k