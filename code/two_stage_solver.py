import numpy as np
from dual_solve.stochastic_dual_subgradient import StochasticDualSubgradient
from frank_wolfe_algorithms.bcfw import BlockCoordinateFrankWolfe
from separable_opt_problem import SeparableOptProblem

class TwoStageStochasticDualSubgradientBlockFrankWolfe:
    
    def __init__(self, separable_opt_problem:SeparableOptProblem):
        """
        Args:
            oracles: list of functions, each oracle_i(g) returns argmin_{x in D_i} <g, x>
        """
        self.separable_opt_problem = separable_opt_problem
        self.n_components = separable_opt_problem.n


    def optimize(self, lbd_0, max_iter_stochastic_dual_subgradient, alpha_bar, 
                 max_iter_block_FW, 
                 freq_compute_dual_cost=100,
                 freq_compute_primal_cost=100,
                 initialize_block_coordinate_FW_with_stochastic_dual_subgradient_output=True,
                 stepsize_strategy_block_fw="fixed"):
        
        stochastic_dual_subgradient_solver = StochasticDualSubgradient(separable_opt_problem=self.separable_opt_problem)

        history_stoch_dual_sub, X = stochastic_dual_subgradient_solver.optimize(lbd_0=lbd_0, 
                                                                             max_iter=max_iter_stochastic_dual_subgradient, 
                                                                             freq_compute_dual=freq_compute_dual_cost, 
                                                                             alpha_bar=alpha_bar)
        
        d_star = np.max(np.array(history_stoch_dual_sub["dual_value"]))

        block_coordinate_FW_solver = BlockCoordinateFrankWolfe(separable_opt_problem=self.separable_opt_problem, d_star=d_star)

        if initialize_block_coordinate_FW_with_stochastic_dual_subgradient_output:
            X0 = X
        else:
            #create matrix of primal candidates
            X0 = np.zeros((self.separable_opt_problem.A_list[0].shape[1], self.n_components))
            for i in range(self.n_components):
                #initialize with a feasible primal point
                X0[:, i] = self.separable_opt_problem.oracle(i, 1, (np.random.randn(self.A_list[i].shape[1])))[0]
        

        history_block_FW, X_sol = block_coordinate_FW_solver.optimize(X_0=X0, 
                                                               max_iter=max_iter_block_FW, 
                                                               freq_compute_cost=freq_compute_primal_cost,
                                                               stepsize_strategy=stepsize_strategy_block_fw)
        #add the linear oracle calls from the previous algorithm
        history_block_FW['nb_oracle_calls'] = (np.array(history_block_FW['nb_oracle_calls']) + history_stoch_dual_sub['total_nb_oracle_calls']).tolist()
        history_block_FW['iteration'] = (np.array(history_block_FW['iteration']) + max_iter_stochastic_dual_subgradient).tolist()

        return history_stoch_dual_sub, history_block_FW