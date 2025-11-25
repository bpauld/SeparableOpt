import numpy as np
from pev_problem import PEVProblem
from utils import create_pevs_problem
import sys
import os
import time

sys.path.insert(1, os.path.join(os.path.dirname(__file__), '..', 'code'))
from two_stage_solver import TwoStageStochasticDualSubgradientBlockFrankWolfe
from dual_solve.dual_subgradient import DualSubgradient
from pev_problem import PEVProblem


def create_pevs_problem(n, m, random_seed=0):
    np.random.seed(random_seed)
    delta_T = 1/3
    xi = np.random.uniform(low=0.015, high=0.075, size=n)
    xi_u = 1 - xi
    xi_v = 1 + xi #useless for charging only scenario
    P = np.random.uniform(low=3, high=5, size=n)
    E_min = np.ones(n)
    E_max = np.random.uniform(low=8, high=16, size=n)
    E_init = np.random.uniform(low=0.2, high=0.5) * E_max
    E_ref = np.random.uniform(low=0.55, high=0.8) * E_max
    P_max = 3 * n * np.ones(m)
    P_min = - P_max  # no need to consider P_min in charging only scenario if P_min < 0
    Cu = np.random.uniform(low=19, high=35, size=m)
    #Cv = 1.1 * Cu

    delta_u = np.random.uniform(low=-0.3, high=0.3, size=n)
    delta_v = np.random.uniform(low=-0.3, high=0.3, size=n)

    #compute rho
    rho = m * np.max(P)
    P_max_bar = P_max - rho
    P_min_bar = P_min + rho

    return PEVProblem(n, m, delta_T, xi_u, xi_v, P, E_min, E_max, E_init, E_ref, P_max, Cu, rho, P_max_bar, delta_u)

def get_approximate_dual_lipschitz_constant(pev_problem: PEVProblem):
    #build the Q
    max_Gi = 0 
    for i in range(pev_problem.n):
        x_ik = pev_problem.oracle(i, 1, np.zeros(0), np.random.randn(pev_problem.m))[0]
        candidate_G_i = np.linalg.norm(x_ik - 1/pev_problem.n * pev_problem.P_max_bar)
        if candidate_G_i > max_Gi:
            max_Gi = candidate_G_i
    return max_Gi





def test_function(n, m, max_number_oracle_calls, 
                  alpha_bar_dual_sub=1,
                  alpha_bar_two_stage=1):
    dimension_eq = 0
    dimension_ineq = m

    pev_problem = create_pevs_problem(n, m, random_seed=0)


    dual_subgradient_solver = DualSubgradient(problem=pev_problem)

    two_stage_solver = TwoStageStochasticDualSubgradientBlockFrankWolfe(problem=pev_problem)
    G = get_approximate_dual_lipschitz_constant(pev_problem)
    print(f"Approximate value of G = {G}")

    max_iter_two_stage = max_number_oracle_calls
    ratio_iter_stoch_dual_subgradient = 0.5
    max_iter_stoch_dual_subgradient = int(max_iter_two_stage * ratio_iter_stoch_dual_subgradient)
    max_iter_block_FW = int(max_iter_two_stage * (1 - ratio_iter_stoch_dual_subgradient))
    freq_compute_dual_stoch = max_iter_stoch_dual_subgradient // 20
    lbd_0 = np.random.randn(dimension_eq)
    mu_0 = np.zeros(dimension_ineq)
    lbd_0 = np.zeros(dimension_eq)

    max_iter_dual_subgradient = max_number_oracle_calls // n
    freq_compute_dual = max_iter_dual_subgradient // 100
    alpha_bar_dual_sub = alpha_bar_dual_sub * (1 / G)
    history_dual_sub, X_sol_dual_sub = dual_subgradient_solver.optimize(lbd_0=lbd_0, mu_0=mu_0,
                                                                 max_iter=max_iter_dual_subgradient,
                                                                 freq_compute_dual=freq_compute_dual,
                                                                 alpha_bar=alpha_bar_dual_sub)




    start_time_2_stage = time.time()
    alpha_bar_stoch_dual_sub = alpha_bar_two_stage * (1 / G)
    history_stoch_dual_sub, history_block_FW, X_sol_two_stage = two_stage_solver.optimize(lbd_0=lbd_0,
                                                                                mu_0=mu_0,
                                                                        max_iter_stochastic_dual_subgradient=max_iter_stoch_dual_subgradient,
                                                                        alpha_bar=alpha_bar_stoch_dual_sub,
                                                                        max_iter_block_FW=max_iter_block_FW, 
                                                                        freq_compute_dual_cost=freq_compute_dual_stoch,
                                                                        freq_compute_primal_cost=freq_compute_dual_stoch,
                                                                        stepsize_strategy_block_fw='linesearch')

    total_time_2_stage = time.time() - start_time_2_stage
    opt_primal_value_2_stage = pev_problem.h(X_sol_two_stage)
    infeasibility_2_stage = np.linalg.norm(pev_problem.compute_infeasibility(X_sol_two_stage))
    print(f"Optimal value found using 2-stage algorithm = {opt_primal_value_2_stage} with infeasibility = {infeasibility_2_stage:.4e} in {total_time_2_stage:.4f} seconds")


    return pev_problem, history_dual_sub, history_stoch_dual_sub, history_block_FW, X_sol_dual_sub, X_sol_two_stage

if __name__ == "__main__":
    n = 1000
    m = 10
    max_number_oracle_calls = n * 1000
    test_function(n, m, max_number_oracle_calls, alpha_bar_dual_sub=1)