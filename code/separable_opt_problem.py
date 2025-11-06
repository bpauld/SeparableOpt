from abc import ABC, abstractmethod
import numpy as np


class SeparableOptProblem(ABC):
    """
    Base class for separable optimization problems of the form:

        min  1/n * sum_{i=1}^n h_i(x_i)
        s.t. 1/n * sum_{i=1}^n A_i x_i = b
             x_i in X_i for all i

    where h_i can be either convex or nonconvex.

    When creating a child of SeparableOptProblem, one must implement:
    - oracle(i, gamma, g): returns argmin_{x_i in X_i} gamma * h_i(x_i) + g^T A_i x_i
    - h_i(i, x_i): evaluates the objective function h_i at x_i
    - Other abstract methods as needed
    """

    def __init__(self, n, h_list, A_list, b, oracle_list):
        """
        Initialize the separable optimization problem.

        Parameters
        ----------
        n : int
            Number of blocks in the separable problem
        A_list : list of ndarray
            List of matrices [A_1, A_2, ..., A_n] where A_i is the constraint matrix
            for block i. Each A_i should be of shape (m, d_i) where m is the number
            of constraints and d_i is the dimension of x_i.
        """
        self.n = n
        self.A_list = A_list
        self.h_list = h_list
        self.oracle_list = oracle_list
        self.b = b

        # Validate that A_list has n elements
        if len(A_list) != n:
            raise ValueError(f"A_list must have exactly n={n} elements, got {len(A_list)}")
        # Validate that h_list has n elements
        if len(h_list) != n:
            raise ValueError(f"h_list must have exactly n={n} elements, got {len(h_list)}")
        

    def h_i(self, i, x_i):
        """
        Evaluate the objective function h_i at x_i.

        Parameters
        ----------
        i : int
            Block index
        x_i : ndarray
            Block variable

        Returns
        -------
        float
            Value of h_i(x_i)
        """
        return self.h_list[i](x_i)
    
    def h(self, x):
        """
        Evaluate the total objective function sum_{i=1}^n h_i(x_i).

        Parameters
        ----------
        x : ndarray
            Full variable (di x n) or dictionary with keys 0, ..., n-1

        Returns
        -------
        float
            Value of sum_{i=1}^n h_i(x_i)
        """
        result = 0.0
        if isinstance(x, dict):
            for i in range(self.n):
                result += self.h_list[i](x[i])
        else:
            # Assume x is a matrix where column i is x_i
            for i in range(self.n):
                result += self.h_list[i](x[:, i])
        return 1/self.n * result

    def oracle(self, i, gamma, g):
        """
        Oracle for the i-th block:

            argmin_{x_i in X_i} gamma * h_i(x_i) + g^T A_i x_i

        Parameters
        ----------
        i : int
            Block index
        gamma : float
            Weight on the objective function h_i
        g : ndarray
            Dual variable for the constraint

        Returns
        -------
        ndarray
            Minimizer x_i^* in X_i
        """
        return self.oracle_list[i](gamma, g)
    
    def compute_dual(self, lbd):
        """
        Compute the dual function value at lambda.

        Parameters
        ----------
        lbd : ndarray
            Dual variable

        Returns
        -------
        float
            Dual function value
        """
        dual_value = 0.0
        for i in range(self.n):
            dual_value += 1/self.n * self.oracle(i, 1, lbd)[1]
            #dual_value += (lbd @ (self.A_list[i] @ x_i_star)) + self.h_list[i](x_i_star)
        dual_value -=  lbd @ self.b
        return dual_value

    def compute_Ai_dot_x(self, i, x_i):
        """
        Compute A_i @ x_i.

        Parameters
        ----------
        i : int
            Block index
        x_i : ndarray
            Block variable

        Returns
        -------
        ndarray
            A_i @ x_i
        """
        return self.A_list[i] @ x_i

    def compute_A_dot_x(self, x):
        """
        Compute sum_{i=1}^n A_i @ x_i.

        Parameters
        ----------
        x : ndarray or dict
            Full variable

        Returns
        -------
        ndarray
            1/n * sum_{i=1}^n A_i @ x_i
        """
        result = 0.0
        if isinstance(x, dict):
            for i in range(self.n):
                Ai_xi = self.compute_Ai_dot_x(i, x[i])
                result += Ai_xi
        else:
            # Assume x is a matrix where column i is x_i
            for i in range(self.n):
                Ai_xi = self.compute_Ai_dot_x(i, x[:, i])
                result += Ai_xi
        return 1/self.n * result

    def compute_AiT_dot_g(self, i, g):
        """
        Compute A_i^T @ g.

        Parameters
        ----------
        i : int
            Block index
        g : ndarray
            Dual variable

        Returns
        -------
        ndarray
            A_i^T @ g
        """
        return self.A_list[i].T @ g

    def get_constraint_rhs(self):
        """
        Get the right-hand side b of the constraint 1/n * sum_{i=1}^n A_i x_i = b.

        Returns
        -------
        ndarray
            Right-hand side vector b
        """
        return self.b

    def get_feasible_point(self):
        """
        Return a feasible point for the problem (if available).
        Can be overridden in subclasses.

        Returns
        -------
        ndarray or dict
            A feasible point x
        """
        raise NotImplementedError("get_feasible_point must be implemented in subclass")

    def check_feasibility(self, x, tol=1e-6):
        """
        Check if x is feasible (satisfies the constraint within tolerance).

        Parameters
        ----------
        x : ndarray or dict
            Point to check
        tol : float, optional
            Tolerance for constraint satisfaction

        Returns
        -------
        bool
            True if feasible, False otherwise
        """
        Ax = self.compute_A_dot_x(x)
        return np.linalg.norm(Ax - self.b) <= tol
    
    def compute_infeasibility(self, x):
        """
        Compute the infeasibility norm ||sum_{i=1}^n A_i x_i - b||.

        Parameters
        ----------
        x : ndarray or dict
            Point to evaluate

        Returns
        -------
        float
            Infeasibility norm
        """
        Ax = self.compute_A_dot_x(x)
        return Ax - self.b

    def get_y_ik(self, i, y_k):
        """
        Extract the i-th block from y_k.
        Can be overridden for different internal representations.

        Parameters
        ----------
        i : int
            Block index
        y_k : ndarray or dict
            Full variable y_k

        Returns
        -------
        ndarray
            Block y_k[i]
        """
        if isinstance(y_k, dict):
            return y_k[i]
        else:
            return y_k[:, i]
