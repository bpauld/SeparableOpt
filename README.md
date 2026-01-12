# Separable Optimization via Frank-Wolfe and Caratheodory.

This repository contains code for solving nonconvex separable optimization problems of the form

$$
\begin{align}
\text{minimize } \ &\sum_{i=1}^n f_i(x_i) \\
\text{subject to } \ &\sum_{i=1}^n A_i x_i \leq b \\
&x_i \in \text{dom}(f_i), \quad i=1, \dots, n
\end{align}
$$

where the functions $f_i$ and their domains need not be convex.

The full theoretical background can be found in our paper:

Benjamin Dubois-Taine, Laurent Pfeiffer, Nadia Oudjane, Adrien Seguret, Francis Bach. "". In: arXiv preprint.

## Use the code

If your problem is convex, you should define the problem by creating a child of the `ConvexSeparableOptProblem` class defined in `separable_opt_problem.py` and implementing the required functions.

If your problem is nonconvex, you should define a child of the `NonConvexSeparableOptProblem` class defined in `separable_opt_problem.py` and implementing the required functions. You can use the file `PEVs/pev_problem.py` as template.

You can then run the two-stage solver implemented in `two_stage_solver.py`

## Reproduce experiments

To reproduce the experiment from the first plot on the charging of electric vehicles problem, run the following in `PEVs` folder:
```
$ python script.py
```

To reproduce the experiment from the second plot on the charging of electric vehicles problem, run the following in `PEVs` folder:
```
$ python script_nonconvex.py
```

You can use the jupyter notebook to reproduce the plots.

## Citing our work

To cite our work please use:

```
@article{dubois2024frank,
}
```
