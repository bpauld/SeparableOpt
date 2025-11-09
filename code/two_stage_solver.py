import numpy as np
from dual_solve.stochastic_dual_subgradient import StochasticDualSubgradient
from frank_wolfe_algorithms.bcfw import BlockCoordinateFrankWolfe
from separable_opt_problem import SeparableOptProblem
from separable_opt_problem import NonConvexSeparableOptProblem
from caratheodory.mnp import MetaCaratheodoryMNPSolve

class TwoStageStochasticDualSubgradientBlockFrankWolfe:
    
    def __init__(self, problem:SeparableOptProblem):
        """
        Args:
            oracles: list of functions, each oracle_i(g) returns argmin_{x in D_i} <g, x>
        """
        self.problem = problem
        self.n_components = problem.n
        if not problem.is_convex:
            if not isinstance(problem, NonConvexSeparableOptProblem):
                raise ValueError("For nonconvex problems, problem must be an instance of NonConvexSeparableOptProblem")


    def optimize(self, lbd_0, max_iter_stochastic_dual_subgradient, alpha_bar, 
                 max_iter_block_FW, 
                 freq_compute_dual_cost=100,
                 freq_compute_primal_cost=100,
                 initialize_block_coordinate_FW_with_stochastic_dual_subgradient_output=True,
                 stepsize_strategy_block_fw="fixed"):
        
        stochastic_dual_subgradient_solver = StochasticDualSubgradient(problem=self.problem)

        history_stoch_dual_sub, X = stochastic_dual_subgradient_solver.optimize(lbd_0=lbd_0, 
                                                                             max_iter=max_iter_stochastic_dual_subgradient, 
                                                                             freq_compute_dual=freq_compute_dual_cost, 
                                                                             alpha_bar=alpha_bar)
        
        d_star = np.max(np.array(history_stoch_dual_sub["dual_value"]))

        block_coordinate_FW_solver = BlockCoordinateFrankWolfe(problem=self.problem, d_star=d_star)

        if initialize_block_coordinate_FW_with_stochastic_dual_subgradient_output:
            X0 = X
        else:
            #create matrix of primal candidates
            X0 = np.zeros((self.problem.A_list[0].shape[1], self.n_components))
            for i in range(self.n_components):
                #initialize with a feasible primal point
                X0[:, i] = self.problem.oracle(i, 1, (np.random.randn(self.A_list[i].shape[1])))[0]
        

        history_block_FW, X_sol, w_K, beta_z_dic, x_dic, weights_dic = block_coordinate_FW_solver.optimize(X_0=X0, 
                                                               max_iter=max_iter_block_FW, 
                                                               freq_compute_cost=freq_compute_primal_cost,
                                                               stepsize_strategy=stepsize_strategy_block_fw)
        #add the linear oracle calls from the previous algorithm
        history_block_FW['nb_oracle_calls'] = (np.array(history_block_FW['nb_oracle_calls']) + history_stoch_dual_sub['total_nb_oracle_calls']).tolist()
        history_block_FW['iteration'] = (np.array(history_block_FW['iteration']) + max_iter_stochastic_dual_subgradient).tolist()

        if not self.problem.is_convex:
            #call Caratheodory algorithm
            carathodory_mnp_solver = MetaCaratheodoryMNPSolve(y_dic=beta_z_dic, 
                                                              x_dic=x_dic,
                                                              weights_dic=weights_dic, verbose=True)

            caratheodory_output = carathodory_mnp_solver.solve(z=w_K, 
                                                                nb_indices_considered=2*self.problem.b.shape[0],
                                                                T=self.n_components * 10)
            
            X_final = self.problem.build_final_solution_from_caratheodory_output(caratheodory_output)

            rec = np.zeros_like(w_K)
            for i in range(self.n_components):
                x_i = caratheodory_output.y_dic[i][0] @ caratheodory_output.y_dic[i][1]
                rec += 1/self.n_components * np.concatenate(([self.problem.h_i(i, X_final[:, i])],self.problem.compute_Ai_dot_x(i, X_final[:, i])))
            print(np.linalg.norm(rec - w_K))
            
            cost_final = self.problem.h(X_final)
            infeasibility_final = np.linalg.norm(self.problem.compute_infeasibility(X_final))
            print(f"After Caratheodory MNP: final cost = {cost_final}, infeasibility = {infeasibility_final}")

            #TODO: build final solution from Caratheodory output

        return history_stoch_dual_sub, history_block_FW, X_sol