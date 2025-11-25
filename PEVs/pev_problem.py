import sys
import os
sys.path.insert(1, os.path.join(os.path.dirname(__file__), '..', 'code'))
from separable_opt_problem import NonConvexSeparableOptProblem
import numpy as np
import scipy as sp

class PEVProblem(NonConvexSeparableOptProblem):
    """
    PEV problem as a special case of NonConvexSeparableOptProblem.
    """

    def __init__(self, n, m, delta_T, xi_u, xi_v, P, E_min, E_max, E_init, E_ref, P_max, Cu, rho, P_max_bar, delta_u):
        """
        Initialize the PEV problem.

        Parameters
        ----------
        n : int
            Number of electric vehicles.
        m : int
            Number of time slots.
        delta_T : float
            Duration of each time slot.
        xi_u : array-like
            Charging efficiency for each vehicle.
        xi_v : array-like
            Discharging efficiency for each vehicle.
        P : array-like
            Power rating for each vehicle.
        E_min : array-like
            Minimum energy level for each vehicle.
        E_max : array-like
            Maximum energy level for each vehicle.
        E_init : array-like
            Initial energy level for each vehicle.
        E_ref : array-like
            Reference energy level for each vehicle.
        P_max : array-like
            Maximum power constraint for each vehicle.
        Cu : float
            Unit cost of charging.
        rho : float
            Penalty parameter.
        P_max_bar : float
            Aggregate power constraint.
        delta_u : array-like
            Additional cost parameters for each vehicle.
        """
        self.n = n
        self.m = m
        self.delta_T = delta_T
        self.xi_u = xi_u
        self.xi_v = xi_v
        self.P = P
        self.E_min = E_min
        self.E_max = E_max
        self.E_init = E_init
        self.E_ref = E_ref
        self.P_max = P_max
        self.Cu = Cu
        self.rho = rho
        self.P_max_bar = P_max_bar
        self.delta_u = delta_u
        self.b_ineq_bar = P_max_bar / self.n

        h_list = []
        for i in range(n):
            c_i = self.P[i] * (self.Cu + self.delta_u[i])
            def h_i(x, c=c_i):
                return c @ x
            h_list.append(h_i)

        super().__init__(n=n, h_list=h_list, b_ineq=self.b_ineq_bar)


    def compute_Ai_ineq_dot_x(self, i, x):
        """
        Compute A_i^ineq * x for the PEV problem.

        Parameters
        ----------
        i : int
            Index of the vehicle.
        x : array-like
            Decision variable for vehicle i.

        Returns
        -------
        array-like
            Result of A_i^ineq * x.
        """
        return self.P[i] * x
    
    def compute_Ai_eq_dot_x(self, i, x):
        return np.zeros(0)
    
    def oracle(self, i, gamma, g, v):
        """
        Oracle for the PEV problem.

        Parameters
        ----------
        i : int
            Index of the vehicle.
        gamma : float
            Dual variable associated with the objective.
        g : array-like
            Dual variable associated with equality constraints.
        v : array-like
            Dual variable associated with inequality constraints.

        Returns
        -------
        tuple
            Optimal solution and its cost.
        """
        C = gamma * self.P[i] * (self.Cu + self.delta_u[i]) + self.P[i] * v

        nb_of_ones_needed = int(np.ceil((self.E_ref[i] - self.E_init[i]) /  (self.P[i] * self.delta_T * self.xi_u[i]))) #to ensure we reach E_ref

        sorted_indices = np.argsort(C)
        res = np.zeros(self.m)
        res[sorted_indices[:nb_of_ones_needed]] = 1
    
        current_energy_consumed = self.E_init[i] + nb_of_ones_needed * self.P[i] * self.delta_T * self.xi_u[i]
        added_indices_pointer = nb_of_ones_needed
        #check if we can add more 1's while it decreases cost and respects E_max
        while added_indices_pointer < self.m and C[sorted_indices[added_indices_pointer]] < 0 and current_energy_consumed < self.E_max[i] - self.P[i] * self.delta_T * self.xi_u[i]:
            res[sorted_indices[added_indices_pointer]] = 1
            added_indices_pointer += 1
            current_energy_consumed += self.P[i] * self.delta_T * self.xi_u[i]
        return res, res @ C   
    
    def get_di(self, i):
        return self.m
    
    def build_final_solution_from_caratheodory_output(self, caratheodory_output):
        X_final = np.zeros((self.m, self.n))
        y_dic = caratheodory_output.y_dic
        nb_trivial_cvx_comb = 0
        for i in range(self.n):
            if y_dic[i][0].shape[1] == 1:
                X_final[:, i] = y_dic[i][0][:, 0]
                nb_trivial_cvx_comb += 1
            else:
                index_alpha_max = np.argmax(y_dic[i][1])
                X_final[:, i] = y_dic[i][0][:, index_alpha_max]
        print(f"Number of trivial convex combinations after Caratheodory: {nb_trivial_cvx_comb} out of {self.n}")
        return X_final
