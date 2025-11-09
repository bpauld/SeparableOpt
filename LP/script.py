import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os

sys.path.insert(1, os.path.join(os.path.dirname(__file__), '..', 'code'))
from two_stage_solver import TwoStageStochasticDualSubgradientBlockFrankWolfe
from lp_problem import LPProblem
import pulp
import random_polytope as rp
from highs_solver import HIGHSSolver


#we compute an approximate bound on G, the Lipschitz constant of the dual
def get_approximate_dual_lipschitz_constant(A_list, b, polytope_list):
    #build the Q
    max_Gi = 0
    for i in range(len(A_list)):
        n_vertices = polytope_list[i].vertices.shape[0]
        for k in range(n_vertices):
            candidate_G_i = np.linalg.norm(A_list[i] @ polytope_list[i].vertices[k] - b)
            if candidate_G_i > max_Gi:
                max_Gi = candidate_G_i
    return max_Gi

def test_function(seed=None):
    equality_constrained_problem = True

    if seed is not None:
        np.random.seed(seed)
    n = 6000
    di = 5
    m = 10
    n_vertices = 10
    oracle_list = []
    A_list = []
    c_list = []
    h_list = []
    polytope_list = []
    for i in range(n):
        oracle, polytope = rp.generate_random_polytope_oracle(
                di, method='random_vertices', n_vertices=n_vertices, 
                scale=1.0, seed=None
            )
        c_i = 100 * np.random.randn(di)
        Ai = 100 * np.random.randn(m, di)
        def h_i(x, c_i=c_i):
            return c_i @ x
        def oracle_i(gamma, v, oracle=oracle, c_i=c_i, Ai=Ai):
            return oracle(gamma * c_i + Ai.T @ v)

        c_list.append(c_i)
        h_list.append(h_i)
        A_list.append(Ai)
        polytope_list.append(polytope)

        oracle_list.append(oracle_i)
    #ensure feasibility
    b = np.zeros(m)
    for i in range(n):
        rand_xi = 0.5 * polytope_list[i].vertices[0] + 0.5 * polytope_list[i].vertices[1]
        b += 1/n * A_list[i] @ rand_xi
        
    lp_problem = LPProblem(n=n, c_list=c_list, A_list=A_list, b=b, oracle_list=oracle_list)

    two_stage_solver = TwoStageStochasticDualSubgradientBlockFrankWolfe(problem=lp_problem)
    G = get_approximate_dual_lipschitz_constant(A_list, b, polytope_list)
    print(f"Approximate value of G = {G}")


    max_iter_two_stage = n * 100
    ratio_iter_stoch_dual_subgradient = 0.5
    max_iter_stoch_dual_subgradient = int(max_iter_two_stage * ratio_iter_stoch_dual_subgradient)
    max_iter_block_FW = int(max_iter_two_stage * (1 - ratio_iter_stoch_dual_subgradient))
    freq_compute_dual_stoch = n * 10 # freq_compute_dual * n
    alpha_bar_stoch = n / G
    lbd_0 = np.random.randn(m)
    start_time_2_stage = time.time()
    history_stoch_dual_sub, history_block_FW, X_sol = two_stage_solver.optimize(lbd_0=lbd_0,
                                                                        max_iter_stochastic_dual_subgradient=max_iter_stoch_dual_subgradient,
                                                                        alpha_bar=alpha_bar_stoch,
                                                                        max_iter_block_FW=max_iter_block_FW, 
                                                                        freq_compute_dual_cost=freq_compute_dual_stoch,
                                                                        freq_compute_primal_cost=freq_compute_dual_stoch,
                                                                        stepsize_strategy_block_fw='linesearch')

    total_time_2_stage = time.time() - start_time_2_stage
    opt_primal_value_2_stage = lp_problem.h(X_sol)
    infeasibility_2_stage = np.linalg.norm(lp_problem.compute_infeasibility(X_sol))
    print(f"Optimal value found using 2-stage algorithm = {opt_primal_value_2_stage} with infeasibility = {infeasibility_2_stage:.4e} in {total_time_2_stage:.4f} seconds")

    """

    highs_solver = HIGHSSolver(c_list, A_list, b, polytope_list,
                                equality_constrained_problem=equality_constrained_problem)
    primal_feasibility_tol = 1e-7
    dual_feasibility_tol = 1e-7
    internal_solver = 'simplex'
    optimal_value_lp, optimal_solution, solve_time_lp = highs_solver.solve(primal_feasibility_tol=primal_feasibility_tol,
                                                                     dual_feasibility_tol=dual_feasibility_tol,
                                                                     solver=internal_solver)



    two_stage_solver = TwoStageStochasticDualSubgradientBlockFrankWolfe(h_list=h_list, A_list=A_list, b=b, oracle_list=oracle_list)
    G = get_approximate_dual_lipschitz_constant(A_list, b, polytope_list)
    print(f"Approximate value of G = {G}")


    max_iter_two_stage = n * 100
    ratio_iter_stoch_dual_subgradient = 0.5
    max_iter_stoch_dual_subgradient = int(max_iter_two_stage * ratio_iter_stoch_dual_subgradient)
    max_iter_block_FW = int(max_iter_two_stage * (1 - ratio_iter_stoch_dual_subgradient))
    freq_compute_dual_stoch = n * 10 # freq_compute_dual * n
    alpha_bar_stoch = n / G
    lbd_0 = np.random.randn(m)
    start_time_2_stage = time.time()
    history_stoch_dual_sub, history_block_FW = two_stage_solver.optimize(lbd_0=lbd_0,
                                                                        max_iter_stochastic_dual_subgradient=max_iter_stoch_dual_subgradient,
                                                                        alpha_bar=alpha_bar_stoch,
                                                                        max_iter_block_FW=max_iter_block_FW, 
                                                                        freq_compute_dual_cost=freq_compute_dual_stoch,
                                                                        freq_compute_primal_cost=freq_compute_dual_stoch,
                                                                        stepsize_strategy_block_fw='linesearch')
    
    total_time_2_stage = time.time() - start_time_2_stage
    opt_value_2_stage = history_block_FW['primal_value'][-1]
    infeasibility_2_stage = history_block_FW['infeasibility'][-1]

    print(f"Optimal value found using HIGHS solver = {optimal_value_lp} in {solve_time_lp:.4f} seconds")
    print(f"Optimal value found using 2-stage algorithm = {opt_value_2_stage} with infeasibility = {infeasibility_2_stage:.4e} in {total_time_2_stage:.4f} seconds")

    # Compute stats to compare LP and 2-stage algorithm
    diff_opt_value = opt_value_2_stage - optimal_value_lp
    print(f"Difference in optimal value between 2-stage algorithm and LP solver = {diff_opt_value}, relative difference = {np.abs(diff_opt_value/optimal_value_lp)}")



    """
    


if __name__ == "__main__":
    test_function(seed=123)