import numpy as np
from pev_problem import PEVProblem
import sys
import os
import time

sys.path.insert(1, os.path.join(os.path.dirname(__file__), '..', 'code'))
from two_stage_solver import TwoStageStochasticDualSubgradientBlockFrankWolfe
from dual_solve.dual_subgradient import DualSubgradient
from dual_solve.stochastic_dual_subgradient import StochasticDualSubgradient
from pev_problem import PEVProblem
import json
from datetime import datetime
from milp_solver import PevMILPSolver


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
    Cv = 1.1 * Cu

    delta_u = np.random.uniform(low=-0.3, high=0.3, size=n)
    delta_v = np.random.uniform(low=-0.3, high=0.3, size=n)

    #compute rho
    rho = m * np.max(P)
    rho = 0
    P_max_bar = P_max - rho
    P_min_bar = P_min + rho

    return PEVProblem(n, m, delta_T, xi_u, xi_v, P, E_min, E_max, E_init, E_ref, P_max, P_min, Cu, Cv, rho, P_max_bar, delta_u, delta_v)

def get_approximate_dual_lipschitz_constant(pev_problem: PEVProblem):
    #build the Q
    max_Gi = 0 
    for i in range(pev_problem.n):
        x_ik = pev_problem.oracle(i, 1, np.zeros(0), np.random.randn(pev_problem.m))[0]
        candidate_G_i = np.linalg.norm(x_ik - 1/pev_problem.n * pev_problem.P_max_bar)
        if candidate_G_i > max_Gi:
            max_Gi = candidate_G_i
    return max_Gi

def save_experiment(path, n, m, random_seed, 
                    nb_rounds,
                    rho,
                    d_star,
                    alpha_bar_dual_sub_list, alpha_bar_two_stage_list, 
                    history_dual_sub_list, history_stoch_dual_sub_list, 
                    history_block_FW_list):
    data = {
        'n':n,
        'm':m,
        'rho':rho,
        'random_seed': random_seed,
        'nb_rounds': nb_rounds,
        'd_star': d_star,
        'alpha_bar_dual_sub_list': alpha_bar_dual_sub_list,
        'alpha_bar_two_stage_list': alpha_bar_two_stage_list,
        'history_dual_sub_list': history_dual_sub_list,
        'history_stoch_dual_sub_list': history_stoch_dual_sub_list,
        'history_block_FW_list': history_block_FW_list
    }
    with open(path, 'w') as f:
        json.dump(data, f)


def load_experiment(path):
    with open(path, 'r') as f:
        data = json.load(f)
    
    return (
        data['n'],
        data['m'],
        data['random_seed'],
        data['nb_rounds'],
        data['d_star'],
        data['alpha_bar_dual_sub_list'],
        data['alpha_bar_two_stage_list'],
        data['history_dual_sub_list'],
        data['history_stoch_dual_sub_list'],
        data['history_block_FW_list']
    )



def test_function(n, m, max_number_oracle_calls, 
                  alpha_bar_dual_sub_list=[1],
                  alpha_bar_two_stage_list=[1],
                  random_seed=0,
                  nb_rounds=1):
    dimension_eq = 0
    dimension_ineq = m

    pev_problem = create_pevs_problem(n, m, random_seed=random_seed)

    #milp_solver = PevMILPSolver(pev_problem=pev_problem)
    #solver='GUROBI'
    #u, res = milp_solver.build_and_solve(solver)
    #print(res, u.shape)
    #print(pev_problem.h(u), pev_problem.compute_A_ineq_dot_x(u) - 1/n * pev_problem.P_max)


    #dual subgradient solver
    dual_subgradient_solver = DualSubgradient(problem=pev_problem)

    #two-stage solver
    two_stage_solver = TwoStageStochasticDualSubgradientBlockFrankWolfe(problem=pev_problem)

    #compute value of G
    G = get_approximate_dual_lipschitz_constant(pev_problem)
    print(f"Approximate value of G = {G}")

    max_iter_two_stage = max_number_oracle_calls
    ratio_iter_stoch_dual_subgradient = 0.5
    max_iter_stoch_dual_subgradient = int(max_iter_two_stage * ratio_iter_stoch_dual_subgradient)
    max_iter_block_FW = int(max_iter_two_stage * (1 - ratio_iter_stoch_dual_subgradient))
    freq_compute_dual_stoch = max_iter_stoch_dual_subgradient // 100
    lbd_0 = np.random.randn(dimension_eq)
    mu_0 = np.zeros(dimension_ineq)
    lbd_0 = np.zeros(dimension_eq)

    max_iter_dual_subgradient = max_number_oracle_calls // n
    freq_compute_dual = max_iter_dual_subgradient // 100


    #start with stochastic dual subgradient to obtain best dual value
    stoch_dual_subgradient_solver = StochasticDualSubgradient(problem=pev_problem)
    history_stoch_dual_sub, X_stoch_dual_sub, beta_z_dic, x_dic, weights_dic = stoch_dual_subgradient_solver.optimize(lbd_0=lbd_0, mu_0=mu_0,
                                                                             max_iter=max_number_oracle_calls*2, 
                                                                             freq_compute_dual=freq_compute_dual_stoch, 
                                                                             alpha_bar=10/G)
    d_star = np.max(np.array(history_stoch_dual_sub["dual_value"]))
    print(f"Best dual value found with stochastic dual subgradient = {d_star}")


    #run dual subgradient
    history_dual_sub_list = []
    for alpha_bar_dual_sub in alpha_bar_dual_sub_list:
        alpha_bar_dual_sub = alpha_bar_dual_sub * (1 / G)
        history_dual_sub, X_sol_dual_sub = dual_subgradient_solver.optimize(lbd_0=lbd_0, mu_0=mu_0,
                                                                    max_iter=max_iter_dual_subgradient,
                                                                    freq_compute_dual=freq_compute_dual,
                                                                    alpha_bar=alpha_bar_dual_sub)
        history_dual_sub_list.append(history_dual_sub)



    #run two-stage method
    history_stoch_dual_sub_list = []
    history_block_FW_list = []
    start_time_2_stage = time.time()
    for alpha_bar_two_stage in alpha_bar_two_stage_list:
        alpha_bar_stoch_dual_sub = alpha_bar_two_stage * (1 / G)
        history_stoch_dual_sub_dic = {}
        history_block_FW_dic = {}
        for seed_round in range(nb_rounds):
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
            history_stoch_dual_sub_dic[seed_round] = history_stoch_dual_sub
            history_block_FW_dic[seed_round] = history_block_FW

        history_stoch_dual_sub_list.append(history_stoch_dual_sub_dic)
        history_block_FW_list.append(history_block_FW_dic)


    return pev_problem, history_dual_sub_list, history_stoch_dual_sub_list, history_block_FW_list, X_sol_dual_sub, X_sol_two_stage, d_star

if __name__ == "__main__":
    n = 10000
    m = 24
    max_number_oracle_calls = n * 1000
    random_seed = 1
    alpha_bar_dual_sub_list = [ 10]
    alpha_bar_two_stage_list = [ 10]

    nb_rounds = 5
    
    pev_problem, history_dual_sub_list, history_stoch_dual_sub_list, history_block_FW_list, X_sol_dual_sub, X_sol_two_stage, d_star = test_function(n, m, 
                                                                                                                                            max_number_oracle_calls, 
                                                                                                                                            alpha_bar_dual_sub_list=alpha_bar_dual_sub_list, 
                                                                                                                                            alpha_bar_two_stage_list=alpha_bar_two_stage_list, 
                                                                                                                                            random_seed=random_seed,
                                                                                                                                            nb_rounds=nb_rounds)
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = f"results/experiment-{date_str}.json"
    save_experiment(path=path, n=n, m=m, nb_rounds=nb_rounds, rho=pev_problem.rho, d_star=d_star,
                    alpha_bar_dual_sub_list=alpha_bar_dual_sub_list, alpha_bar_two_stage_list=alpha_bar_two_stage_list,
                    history_dual_sub_list=history_dual_sub_list, history_stoch_dual_sub_list=history_stoch_dual_sub_list,
                    history_block_FW_list=history_block_FW_list, random_seed=random_seed)
    