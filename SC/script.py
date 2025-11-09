from sc_problem import SCProblem
import numpy as np
import sys
import os
import time

sys.path.insert(1, os.path.join(os.path.dirname(__file__), '..', 'code'))
from two_stage_solver import TwoStageStochasticDualSubgradientBlockFrankWolfe

def create_random_sc_instance(n, m, seed=0):
    #following the setup in Large Scale Mixed-Integer Optimization: a Solution Method with Supply Chain Applications
    #Robin Vujanic, Peyman Mohajerin Esfahani, Paul Goulart and Manfred Morari
    K = np.random.uniform(1, 100, size=n)
    r = np.random.uniform(1, 15, size=(n, m))
    D = np.random.uniform(1, 100, size=(n, m))
    beta = 0.6

    if n == 100 and m == 25:
        I = np.random.uniform(1000, 2500, size=m)
    elif n == 300 and m == 75:
        I = np.random.uniform(4000, 5000, size=m)
    elif n== 500 and m==50:
        I = np.random.uniform(7000, 85000, size=m)
    elif n==600 and m==50:
        I = np.random.uniform(8500, 9500, size=m)
    elif n==1000 and m==100:
        I = np.random.uniform(14500, 15500, size=n)
    
    sc_problem = SCProblem(K=K, r=r, D=D, I=I, beta=beta)
    return sc_problem

def get_approximate_dual_lipschitz_constant(sc_problem: SCProblem):
    #build the Q
    max_Gi = np.linalg.norm(sc_problem.b)
    for i in range(sc_problem.n):
        x_ik = sc_problem.D[i, :]
        candidate_G_i = np.linalg.norm(x_ik - 1/sc_problem.n * sc_problem.I)
        if candidate_G_i > max_Gi:
            max_Gi = candidate_G_i
    return max_Gi





def test_function():
    n = 100
    m = 25

    sc_problem = create_random_sc_instance(n, m)

    two_stage_solver = TwoStageStochasticDualSubgradientBlockFrankWolfe(problem=sc_problem)
    G = get_approximate_dual_lipschitz_constant(sc_problem)
    print(f"Approximate value of G = {G}")


    max_iter_two_stage = n * 300
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
    opt_primal_value_2_stage = sc_problem.h(X_sol)
    infeasibility_2_stage = np.linalg.norm(sc_problem.compute_infeasibility(X_sol))
    print(f"Optimal value found using 2-stage algorithm = {opt_primal_value_2_stage} with infeasibility = {infeasibility_2_stage:.4e} in {total_time_2_stage:.4f} seconds")

if __name__ == "__main__":
    test_function()