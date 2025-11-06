import numpy as np
import time
import highspy

class HIGHSSolver:
    """Solve the polytope-constrained LP using standard form conversion with HIGHS."""

    def __init__(self, c_list, A_list, b, polytope_list, equality_constrained_problem=True):
        """
        Args:
            c_list: list of n cost vectors, each of dimension d_i
            A_list: list of n matrices A_i of shape (m, d_i)
            b: constraint vector of dimension m
            polytope_list: list of n RandomPolytope objects (each has .A and .b for inequality constraints)
        """
        self.c_list = c_list
        self.A_list = A_list
        self.b = b
        self.polytope_list = polytope_list
        self.n_components = len(c_list)
        self.equality_constrained_problem = equality_constrained_problem

        # Validate dimensions
        self.m = b.shape[0]  # number of coupling constraints
        self.d_list = [c_i.shape[0] for c_i in c_list]  # dimensions of each x_i

        self.h_problem, self.objective = self.create_problem()

    def create_problem(self):
        # Create HiGHS instance
        h = highspy.Highs()

        # Add variables for each component
        x_vars = []
        for i in range(self.n_components):
            d_i = self.d_list[i]
            x_i = [h.addVariable(lb=-highspy.kHighsInf, ub=highspy.kHighsInf) for _ in range(d_i)]
            x_vars.append(x_i)

        # Add polytope constraints for each component
        for i in range(self.n_components):
            A_poly = self.polytope_list[i].A
            b_poly = self.polytope_list[i].b

            # Add each polytope constraint: A_poly[row] @ x_i <= b_poly[row]
            for row in range(A_poly.shape[0]):
                expr = sum(A_poly[row, j] * x_vars[i][j] for j in range(self.d_list[i]))
                h.addConstr(expr <= b_poly[row])

        # Add coupling constraints: (1/n) * sum_i A_i @ x_i = b
        for row in range(self.m):
            expr = sum(
                (self.A_list[i][row, j]/self.n_components) * x_vars[i][j]
                for i in range(self.n_components)
                for j in range(self.d_list[i])
            )
            if self.equality_constrained_problem:
                h.addConstr(expr == self.b[row])
            else:
                h.addConstr(expr <= self.b[row])

        # Set objective: minimize (1/n) * sum_i c_i @ x_i
        objective = sum(
            (self.c_list[i][j] /self.n_components) * x_vars[i][j]
            for i in range(self.n_components)
            for j in range(self.d_list[i])
        )

        return h, objective


    def solve(self, primal_feasibility_tol=None, dual_feasibility_tol=None, solver=None):
        """
        Solve the LP and return the optimal value and solution.

        Args:
            primal_feasibility_tol: tolerance for primal feasibility
            dual_feasibility_tol: tolerance for dual feasibility
            solver: which solver to use. Options:
                   - None or 'choose': let HiGHS choose automatically (default)
                   - 'simplex': use simplex method
                   - 'ipm': use interior point method
                   - 'pdlp': use primal-dual LP method

        Returns:
            optimal_value: the optimal objective value
            optimal_solution: dual solution object
            solve_time: time taken to solve
        """

        print("Solving with HIGHS...")
        start_time = time.time()

        # Set solver method
        if solver is not None:
            solver_map = {
                'choose': 'choose',
                'simplex': 'simplex',
                'ipm': 'ipm',
                'pdlp': 'pdlp'
            }
            if solver.lower() not in solver_map:
                raise ValueError(f"Unknown solver '{solver}'. Options: {list(solver_map.keys())}")
            self.h_problem.setOptionValue("solver", solver_map[solver.lower()])

        # Set tolerances
        if primal_feasibility_tol is not None:
            self.h_problem.setOptionValue("primal_feasibility_tolerance", primal_feasibility_tol)
        if dual_feasibility_tol is not None:
            self.h_problem.setOptionValue("dual_feasibility_tolerance", dual_feasibility_tol)

        self.h_problem.minimize(self.objective)

        solve_time = time.time() - start_time

        # Extract solution
        optimal_value = self.h_problem.getObjectiveValue()
        optimal_dual_value = self.h_problem.getSolution()

        print(self.h_problem.getInfo().max_dual_infeasibility, self.h_problem.getInfo().max_primal_infeasibility)
        print(self.h_problem.getOptions().solver)


        #optimal_solution = []
        #for i in range(self.n_components):
        #    x_i = np.array([x_vars[i][j].value() for j in range(self.d_list[i])])
        #    optimal_solution.append(x_i)

        return optimal_value, optimal_dual_value, solve_time