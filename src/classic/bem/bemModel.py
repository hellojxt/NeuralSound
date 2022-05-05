import bempp.api.linalg
from bempp.api.operators import potential, boundary
from bempp.api import GridFunction, export
import numpy as np

class boundary_mesh():
    def __init__(self, grid):
        self.grid = grid
        self.dp0_space = bempp.api.function_space(self.grid, "DP", 0)
        self.p1_space = bempp.api.function_space(self.grid, "P", 1)
        self.dirichlet_fun = self.neumann_fun = None

    def set_wave_number(self,k):
        self.k = k
    
    def set_neumann_fun(self, neumann_fun):
        self.neumann_fun = neumann_fun
    
    def set_dirichlet_fun(self, dirichlet_fun):
        self.dirichlet_fun = dirichlet_fun

    def preprocess_layer(self):
        self.adjoint_double = boundary.helmholtz.adjoint_double_layer(
                        self.dp0_space, self.p1_space, self.p1_space, self.k, precision="single",device_interface="opencl")
        self.hyper_single   = boundary.helmholtz.hypersingular(
                        self.p1_space, self.p1_space, self.p1_space, self.k, precision="single",device_interface="opencl")
        self.identity       = boundary.sparse.identity(
                        self.dp0_space, self.p1_space, self.p1_space, precision="single",device_interface="opencl")

    def ext_neumann2dirichlet(self):
        left_side = self.hyper_single
        right_side = (-0.5 * self.identity - self.adjoint_double) * self.neumann_fun
        dirichlet_fun, info, res = bempp.api.linalg.gmres(left_side, right_side, tol=1e-10*self.neumann_fun.coefficients.max(), maxiter=500, return_residuals=True)
        self.dirichlet_fun = dirichlet_fun

    def points_dirichlet(self, points):
        potential_single = potential.helmholtz.single_layer(self.dp0_space, points.T, self.k, precision="single",device_interface="opencl")
        potential_double = potential.helmholtz.double_layer(self.p1_space, points.T, self.k, precision="single",device_interface="opencl")
        dirichlet = - potential_single * self.neumann_fun + potential_double * self.dirichlet_fun 
        return dirichlet.reshape(-1)

    def export_neumann(self, filename):
        bempp.api.export(filename, grid_function=self.neumann_fun)

    def export_dirichlet(self, filename):
        bempp.api.export(filename, grid_function=self.dirichlet_fun)


