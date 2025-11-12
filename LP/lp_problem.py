import sys
import os
import numpy as np

# Add parent directory to path to import SeparableOptProblem
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code'))
from separable_opt_problem import ConvexSeparableOptProblem, NonConvexSeparableOptProblem  # type: ignore


class LPProblem(ConvexSeparableOptProblem):
    """
    Linear Programming problem as a special case of SeparableOptProblem.

    The problem has the form:
        min  1/n * sum_{i=1}^n c_i^T x_i
        s.t. 1/n * sum_{i=1}^n A_i x_i = b
             x_i in X_i for all i

    where each X_i is a polytope and h_i(x_i) = c_i^T x_i is linear.
    """

    def __init__(self, n, c_list, A_eq_list, b_eq, oracle_list):
        """
        Initialize the LP problem.

        Parameters
        ----------
        n : int
            Number of blocks
        c_list : list of ndarray
            List of cost vectors [c_1, ..., c_n] where c_i is the linear cost for block i
        A_list : list of ndarray
            List of constraint matrices [A_1, ..., A_n]
        b : ndarray
            Right-hand side of the constraint
        oracle_list : list of callables
            List of oracle functions. Each oracle_list[i](gamma, g) returns:
                (x_i^*, value) where x_i^* = argmin_{x_i in X_i} gamma * c_i^T x_i + g^T A_i x_i
                and value is the optimal objective value
        """
        # Create h_list from c_list (linear functions)
        h_list = []
        for i in range(n):
            c_i = c_list[i]
            def h_i(x, c=c_i):
                return c @ x
            h_list.append(h_i)

        # Call parent constructor (is_convex=True for LP)
        super().__init__(n=n, h_list=h_list, A_eq_list=A_eq_list, b_eq=b_eq, A_ineq_list=None, b_ineq=None)

        # Store LP-specific data
        self.c_list = c_list
        self.oracle_list = oracle_list

        # Validate oracle_list
        if len(oracle_list) != n:
            raise ValueError(f"oracle_list must have exactly n={n} elements, got {len(oracle_list)}")
        
    def get_di(self, i):
        return self.A_eq_list[i].shape[1]

    def oracle(self, i, gamma, g, v):
        """
        Oracle for the i-th block:

            argmin_{x_i in X_i} gamma * c_i^T x_i + g^T A_i x_i

        Parameters
        ----------
        i : int
            Block index
        gamma : float
            Weight on the objective function
        g : ndarray
            Dual variable for the constraint

        Returns
        -------
        tuple (ndarray, float)
            (x_i^*, value) where x_i^* is the minimizer and value is the optimal value
        """
        return self.oracle_list[i](gamma, g)