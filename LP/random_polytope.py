import numpy as np
from scipy.spatial import ConvexHull

class RandomPolytope:
    """Generate and work with random polytopes"""
    
    def __init__(self, n_dim, method='random_vertices', n_vertices=None, 
                 center=None, scale=1.0, seed=None):
        """
        Generate a random polytope
    
        Args:
            n_dim: dimension
            method: 'random_vertices', 'hypercube_perturbation', 'random_halfspaces'
            n_vertices: number of vertices (if None, use 2*n_dim)
            center: center point (if None, use origin)
            scale: scale of the polytope
            seed: random seed
        """
        if seed is not None:
            np.random.seed(seed)
        
        self.n_dim = n_dim
        self.center = np.zeros(n_dim) if center is None else np.array(center)
        self.scale = scale
        
        if n_vertices is None:
            n_vertices = max(n_dim + 1, 2 * n_dim)
        
        if method == 'random_vertices':
            self._generate_from_random_vertices(n_vertices)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _generate_from_random_vertices(self, n_vertices):
        """Generate polytope as convex hull of random points"""
        # Generate random points on unit sphere, then scale
        vertices = np.random.randn(n_vertices, self.n_dim)
        vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
        vertices = vertices * np.random.uniform(0.3, 1.0, (n_vertices, 1))
        vertices = self.center + self.scale * vertices
        
        # Compute convex hull
        hull = ConvexHull(vertices)
        self.vertices = vertices[hull.vertices]
        self.hull = hull
        
        # Store H-representation: A x <= b
        self.A = hull.equations[:, :-1]
        self.b = -hull.equations[:, -1] + 1e-8

    
    def linear_oracle(self, c):
        """
        Solve min_{x in P} <c, x> using linear programming
        
        This returns the vertex that minimizes the dot product with gradient
        """
        dots = self.vertices @ c
        return self.vertices[np.argmin(dots)], c @ self.vertices[np.argmin(dots)]
    
    def contains(self, x):
        """Check if point x is in polytope"""
        return np.all(self.A @ x <= self.b + 1e-8)


def generate_random_polytope_oracle(n_dim, method='random_vertices', **kwargs):
    """Factory function to create a polytope oracle"""
    polytope = RandomPolytope(n_dim, method=method, **kwargs)
    return polytope.linear_oracle, polytope