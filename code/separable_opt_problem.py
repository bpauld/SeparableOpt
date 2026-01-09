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

    def __init__(self, n, 
                 h_list, 
                 A_eq_list=None, b_eq=None, 
                 A_ineq_list=None, b_ineq=None, 
                 is_convex=True):
        """
        Initialize the separable optimization problem.

        Parameters
        ----------
        n : int
            Number of blocks in the separable problem
        h_list : list of callables
            List of objective functions [h_1, h_2, ..., h_n]
        A_eq_list : list of ndarray, optional
            List of equality constraint matrices [A_eq_1, A_eq_2, ..., A_eq_n]
        b_eq : ndarray, optional
            Right-hand side of the equality constraints
        A_ineq_list : list of ndarray, optional
            List of inequality constraint matrices [A_ineq_1, A_ineq_2, ..., A_ineq_n]
        b_ineq : ndarray, optional
            Right-hand side of the inequality constraints
        is_convex : bool, optional
            Indicates if the problem is convex (default is True)
        """
        self.n = n
        self.h_list = h_list
        self.A_eq_list = A_eq_list
        self.b_eq = b_eq
        self.A_ineq_list = A_ineq_list
        self.b_ineq = b_ineq
        self.is_convex = is_convex

        if self.b_eq is None:
            self.b_eq = np.zeros(0)
        if self.b_ineq is None:
            self.b_ineq = np.zeros(0)

        self.m = self.b_eq.shape[0] + self.b_ineq.shape[0]

        # Validate that A_list has n elements
        if A_eq_list is not None and len(A_eq_list) != n:
            raise ValueError(f"A_list must have exactly n={n} elements, got {len(A_eq_list)}")
        if A_ineq_list is not None and len(A_ineq_list) != n:
            raise ValueError(f"A_list must have exactly n={n} elements, got {len(A_ineq_list)}")
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

    @abstractmethod
    def oracle(self, i, gamma, g, v):
        """
        Oracle for the i-th block:

            argmin_{x_i in X_i} gamma * h_i(x_i) + g^T A_eq_i x_i + v^T A_ineq_i x_i

        Parameters
        ----------
        i : int
            Block index
        gamma : float
            Weight on the objective function h_i
        g : ndarray
            Dual variable for the equality constraints
        v : ndarray
            Dual variable for the inequality constraints

        Returns
        -------
        ndarray
            Minimizer x_i^* in X_i
        float
            Minimum value of the objective
        """
        pass

    @abstractmethod
    def get_di(self, i):
        """
        Returns the dimension di of the vector x_i
        """
        pass
    
    def compute_dual(self, lbd, mu):
        """
        Compute the dual function value at lambda.

        Parameters
        ----------
        lbd : ndarray
            Dual variable for equality constraints
        mu : ndarray
            Dual variable for inequality constraints

        Returns
        -------
        float
            Dual function value
        """
        dual_value = 0.0
        for i in range(self.n):
            dual_value += 1/self.n * self.oracle(i, 1, lbd, mu)[1]
        dual_value -=  lbd @ self.b_eq + mu @ self.b_ineq
        return dual_value

    def compute_Ai_eq_dot_x(self, i, x_i):
        """
        Compute A_i_eq @ x_i.

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
        if self.A_eq_list is None:
            return np.zeros(0)
        return self.A_eq_list[i] @ x_i
    
    def compute_Ai_ineq_dot_x(self, i, x_i):
        """
        Compute A_i_ineq @ x_i.

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
        if self.A_ineq_list is None:
            return np.zeros(0)
        return self.A_ineq_list[i] @ x_i

    def compute_A_eq_dot_x(self, x):
        """
        Compute 1/n * sum_{i=1}^n A_eq_i @ x_i.

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
                Ai_xi = self.compute_Ai_eq_dot_x(i, x[i])
                result += Ai_xi
        else:
            # Assume x is a matrix where column i is x_i
            for i in range(self.n):
                Ai_xi = self.compute_Ai_eq_dot_x(i, x[:, i])
                result += Ai_xi
        return 1/self.n * result
    
    def compute_A_ineq_dot_x(self, x):
        """
        Compute 1/n * sum_{i=1}^n A_ineq_i @ x_i.

        Parameters
        ----------
        x : ndarray or dict
            Full variable

        Returns
        -------
        ndarray
            1/n * sum_{i=1}^n A_ineq_i @ x_i
        """
        result = 0.0
        if isinstance(x, dict):
            for i in range(self.n):
                Ai_xi = self.compute_Ai_ineq_dot_x(i, x[i])
                result += Ai_xi
        else:
            # Assume x is a matrix where column i is x_i
            for i in range(self.n):
                Ai_xi = self.compute_Ai_ineq_dot_x(i, x[:, i])
                result += Ai_xi
        return 1/self.n * result

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
        Aeq_x = self.compute_A_eq_dot_x(x)
        Aineq_x = self.compute_A_ineq_dot_x(x)
        infeas_eq = Aeq_x - self.b_eq
        infeas_ineq = np.clip(Aineq_x - self.b_ineq, 0, None)
        return np.concatenate((infeas_eq, infeas_ineq))

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


class ConvexSeparableOptProblem(SeparableOptProblem):
    """
    Convex separable optimization problem where each h_i is convex.

    The problem has the form:
        min  1/n * sum_{i=1}^n h_i(x_i)
        s.t. 1/n * sum_{i=1}^n A_i x_i = b
             x_i in X_i for all i

    where each h_i is convex.
    """

    def __init__(self, n, h_list, A_eq_list=None, b_eq=None, A_ineq_list=None, b_ineq=None):
        """
        Initialize the convex separable optimization problem.

        Parameters
        ----------
        n : int
            Number of blocks in the separable problem
        h_list : list of callables
            List of objective functions [h_1, h_2, ..., h_n]
        A_eq_list : list of ndarray, optional
            List of equality constraint matrices [A_eq_1, A_eq_2, ..., A_eq_n]
        b_eq : ndarray, optional
            Right-hand side of the equality constraints
        A_ineq_list : list of ndarray, optional
            List of inequality constraint matrices [A_ineq_1, A_ineq_2, ..., A_ineq_n]
        b_ineq : ndarray, optional
            Right-hand side of the inequality constraints
        is_convex : bool, optional
            Indicates if the problem is convex (default is True)
        """
        super().__init__(n=n, h_list=h_list, A_eq_list=A_eq_list, b_eq=b_eq, A_ineq_list=A_ineq_list, b_ineq=b_ineq, is_convex=True)


class NonConvexSeparableOptProblem(SeparableOptProblem):
    """
    Nonconvex separable optimization problem where h_i and X_i may be nonconvex.

    The problem has the form:
        min  1/n * sum_{i=1}^n h_i(x_i)
        s.t. 1/n * sum_{i=1}^n A_i x_i = b
             x_i in X_i for all i

    where h_i and X_i can be nonconvex.
    """

    def __init__(self, n, h_list, A_eq_list=None, b_eq=None, A_ineq_list=None, b_ineq=None):
        """
        Initialize the nonconvex separable optimization problem.

        Parameters
        ----------
        n : int
            Number of blocks in the separable problem
        h_list : list of callables
            List of objective functions [h_1, h_2, ..., h_n]
        A_eq_list : list of ndarray, optional
            List of equality constraint matrices [A_eq_1, A_eq_2, ..., A_eq_n]
        b_eq : ndarray, optional
            Right-hand side of the equality constraints
        A_ineq_list : list of ndarray, optional
            List of inequality constraint matrices [A_ineq_1, A_ineq_2, ..., A_ineq_n]
        b_ineq : ndarray, optional
            Right-hand side of the inequality constraints
        is_convex : bool, optional
            Indicates if the problem is convex (default is True)
        """
        super().__init__(n=n, h_list=h_list, A_eq_list=A_eq_list, b_eq=b_eq, A_ineq_list=A_ineq_list, b_ineq=b_ineq, is_convex=False)

    @abstractmethod
    def build_final_solution_from_caratheodory_output(self, caratheodory_output):
        """
        Build the final solution from Caratheodory output.

        This method is used to construct a final solution after applying
        Caratheodory's theorem to sparsify the solution.

        Parameters
        ----------
        y_dic : dict
            Dictionary containing the points from the Frank-Wolfe algorithm
        eta_vector : ndarray
            Coefficients from Caratheodory
        corresponding_indices : ndarray or list
            Indices corresponding to the selected points

        Returns
        -------
        ndarray or dict
            Final solution constructed from the output
        """
        pass
