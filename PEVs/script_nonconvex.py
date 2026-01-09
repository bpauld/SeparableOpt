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
                    max_number_oracle_calls_list,
                    cost_primal_dual_sub_list,
                    infeasibility_primal_dual_sub_list,
                    cost_bidual_dual_sub_list,
                    infeasibility_bidual_dual_sub_list,
                    cost_primal_two_stage_list,
                    infeasibility_primal_two_stage_list,
                    cost_bidual_two_stage_list,
                    infeasibility_bidual_two_stage_list
                    ):
    data = {
        'n':n,
        'm':m,
        'rho':rho,
        'random_seed': random_seed,
        'nb_rounds': nb_rounds,
        'd_star': d_star,
        'max_number_oracle_calls_list': max_number_oracle_calls_list,
        'cost_primal_dual_sub_list': cost_primal_dual_sub_list,
        'infeasibility_primal_dual_sub_list': infeasibility_primal_dual_sub_list,
        'cost_bidual_dual_sub_list': cost_bidual_dual_sub_list,
        'infeasibility_bidual_dual_sub_list': infeasibility_bidual_dual_sub_list,
        'cost_primal_two_stage_list': cost_primal_two_stage_list,
        'infeasibility_primal_two_stage_list': infeasibility_primal_two_stage_list,
        'cost_bidual_two_stage_list': cost_bidual_two_stage_list,
        'infeasibility_bidual_two_stage_list': infeasibility_bidual_two_stage_list
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
        data['max_number_oracle_calls_list'],
        data['cost_primal_dual_sub_list'],
        data['infeasibility_primal_dual_sub_list'],
        data['cost_bidual_dual_sub_list'],
        data['infeasibility_bidual_dual_sub_list'],
        data['cost_primal_two_stage_list'],
        data['infeasibility_primal_two_stage_list'],
        data['cost_bidual_two_stage_list'],
        data['infeasibility_bidual_two_stage_list']
    )



def test_function(n, m, max_number_oracle_calls_list, 
                  random_seed=0,
                  alpha_bar_dual_sub=10,
                  alpha_bar_two_stage=10,
                  nb_rounds=1):
    dimension_eq = 0
    dimension_ineq = m

    pev_problem = create_pevs_problem(n, m, random_seed=random_seed)


    #dual subgradient solver
    dual_subgradient_solver = DualSubgradient(problem=pev_problem)

    #two-stage solver
    two_stage_solver = TwoStageStochasticDualSubgradientBlockFrankWolfe(problem=pev_problem)

    #compute value of G
    G = get_approximate_dual_lipschitz_constant(pev_problem)
    print(f"Approximate value of G = {G}")

   
    lbd_0 = np.random.randn(dimension_eq)
    mu_0 = np.zeros(dimension_ineq)
    lbd_0 = np.zeros(dimension_eq)

    freq_compute_dual = max_number_oracle_calls_list[-1] * 2 // 100


    #start with stochastic dual subgradient to obtain best dual value
    stoch_dual_subgradient_solver = StochasticDualSubgradient(problem=pev_problem)
    history_stoch_dual_sub, _, _,_,_ = stoch_dual_subgradient_solver.optimize(lbd_0=lbd_0, mu_0=mu_0,
                                                                             max_iter=max_number_oracle_calls_list[-1] * 2, 
                                                                             freq_compute_dual=freq_compute_dual, 
                                                                             alpha_bar=10/G)
    d_star = np.max(np.array(history_stoch_dual_sub["dual_value"]))
    print(f"Best dual value found with stochastic dual subgradient = {d_star}")


    #run dual subgradient
    cost_primal_dual_sub_list = []
    infeasibility_primal_dual_sub_list = []
    cost_bidual_dual_sub_list = []
    infeasibility_bidual_dual_sub_list = []
    alpha_bar_dual_sub = alpha_bar_dual_sub * (1 / G)
    for max_iter in max_number_oracle_calls_list:
        max_iter_dual_subgradient = max_iter // n
        _, X_sol_dual_sub_primal, X_sol_dual_sub_bidual = dual_subgradient_solver.optimize(lbd_0=lbd_0, mu_0=mu_0,
                                                            max_iter=max_iter_dual_subgradient,
                                                            freq_compute_dual=max_iter_dual_subgradient // 10,
                                                            alpha_bar=alpha_bar_dual_sub)
        
        cost_primal_dual_sub = pev_problem.h(X_sol_dual_sub_primal)
        infeasibility_primal_dual_sub = np.linalg.norm(pev_problem.compute_infeasibility(X_sol_dual_sub_primal))
        cost_primal_dual_sub_list.append(cost_primal_dual_sub)
        infeasibility_primal_dual_sub_list.append(infeasibility_primal_dual_sub)

        cost_bidual_dual_sub = pev_problem.h(X_sol_dual_sub_bidual)
        infeasibility_bidual_dual_sub = np.linalg.norm(pev_problem.compute_infeasibility(X_sol_dual_sub_bidual))
        cost_bidual_dual_sub_list.append(cost_bidual_dual_sub)
        infeasibility_bidual_dual_sub_list.append(infeasibility_bidual_dual_sub)




    #run two-stage method
    cost_primal_two_stage_list = []
    infeasibility_primal_two_stage_list = []
    cost_bidual_two_stage_list = []
    infeasibility_bidual_two_stage_list = []
    alpha_bar_stoch_dual_sub = alpha_bar_two_stage * (1 / G)

    for max_iter in max_number_oracle_calls_list:
        max_iter_two_stage = max_iter
        ratio_iter_stoch_dual_subgradient = 0.5
        max_iter_stoch_dual_subgradient = int(max_iter_two_stage * ratio_iter_stoch_dual_subgradient)
        max_iter_block_FW = int(max_iter_two_stage * (1 - ratio_iter_stoch_dual_subgradient))
        freq_compute_dual_stoch = max_iter_stoch_dual_subgradient // 10

        cost_primal_two_stage_dic = {}
        infeasibility_primal_two_stage_dic = {}
        cost_bidual_two_stage_dic = {}
        infeasibility_bidual_two_stage_dic = {}
        for seed_round in range(nb_rounds):
            _, history_block_FW, X_sol_two_stage = two_stage_solver.optimize(lbd_0=lbd_0,
                                                                mu_0=mu_0,
                                                                max_iter_stochastic_dual_subgradient=max_iter_stoch_dual_subgradient,
                                                                alpha_bar=alpha_bar_stoch_dual_sub,
                                                                max_iter_block_FW=max_iter_block_FW, 
                                                                freq_compute_dual_cost=freq_compute_dual_stoch,
                                                                freq_compute_primal_cost=freq_compute_dual_stoch,
                                                                stepsize_strategy_block_fw='linesearch')

            cost_primal_value_2_stage = pev_problem.h(X_sol_two_stage)
            infeasibility_primal_2_stage = np.linalg.norm(pev_problem.compute_infeasibility(X_sol_two_stage))
            cost_primal_two_stage_dic[str(seed_round)] = cost_primal_value_2_stage
            infeasibility_primal_two_stage_dic[str(seed_round)] = infeasibility_primal_2_stage

            cost_bidual_two_stage = history_block_FW['primal_value'][-1]
            infeasibility_bidual_two_stage = history_block_FW['infeasibility'][-1]
            cost_bidual_two_stage_dic[str(seed_round)] = cost_bidual_two_stage
            infeasibility_bidual_two_stage_dic[str(seed_round)] = infeasibility_bidual_two_stage

        cost_primal_two_stage_list.append(cost_primal_two_stage_dic)
        infeasibility_primal_two_stage_list.append(infeasibility_primal_two_stage_dic)
        cost_bidual_two_stage_list.append(cost_bidual_two_stage_dic)
        infeasibility_bidual_two_stage_list.append(infeasibility_bidual_two_stage_dic)


    return (pev_problem, cost_primal_dual_sub_list, 
            infeasibility_primal_dual_sub_list,
            cost_bidual_dual_sub_list,
            infeasibility_bidual_dual_sub_list,
            cost_primal_two_stage_list, 
            infeasibility_primal_two_stage_list,
            cost_bidual_two_stage_list,
            infeasibility_bidual_two_stage_list,
            d_star)

if __name__ == "__main__":
    n = 10000
    m = 24
    max_number_oracle_calls_list = [n*10, n*50, n*100, n*250, n*500, n*750, n*1000]
    #max_number_oracle_calls_list = [ n*25, n*100]
    random_seed = 1

    nb_rounds = 5
    
    (pev_problem, cost_primal_dual_sub_list, 
            infeasibility_primal_dual_sub_list,
            cost_bidual_dual_sub_list,
            infeasibility_bidual_dual_sub_list,
            cost_primal_two_stage_list, 
            infeasibility_primal_two_stage_list,
            cost_bidual_two_stage_list,
            infeasibility_bidual_two_stage_list,
            d_star) = test_function(n, m, 
                                    max_number_oracle_calls_list, 
                                    random_seed=random_seed,
                                    nb_rounds=nb_rounds)
    
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = f"results_nonconvex/experiment-{date_str}.json"
    save_experiment(path=path, n=n, m=m, nb_rounds=nb_rounds, rho=pev_problem.rho, d_star=d_star,
                    random_seed=random_seed,
                    max_number_oracle_calls_list=max_number_oracle_calls_list,
                    cost_primal_dual_sub_list=cost_primal_dual_sub_list,
                    infeasibility_primal_dual_sub_list=infeasibility_primal_dual_sub_list,
                    cost_bidual_dual_sub_list=cost_bidual_dual_sub_list,
                    infeasibility_bidual_dual_sub_list=infeasibility_bidual_dual_sub_list,
                    cost_primal_two_stage_list=cost_primal_two_stage_list,
                    infeasibility_primal_two_stage_list=infeasibility_primal_two_stage_list,
                    cost_bidual_two_stage_list=cost_bidual_two_stage_list,
                    infeasibility_bidual_two_stage_list=infeasibility_bidual_two_stage_list)
    