# coding=utf-8
import numpy as np
import scipy.sparse as sp

from pymg.problem_base import ProblemBase


class Poisson1D(ProblemBase):
    """Implementation of the 1D Poission problem.

    Here we define the 1D Poisson problem :math:`-\Delta u = 0` with
    Dirichlet-Zero boundary conditions. This is the homogeneous problem,
    derive from this class if you want to play around with different RHS.

    Attributes:
        dx (float): mesh size
    """

    def __init__(self, ndofs, *args, **kwargs):
        """Initialization routine for the Poisson1D problem

        Args:
            ndofs (int): number of degrees of freedom (see
                :attr:`pymg.problem_base.ProblemBase.ndofs`)
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments
        """
        self.dx = 1.0 / (ndofs + 1)
        # compute system matrix A, scale by 1/dx^2
        A = 1.0 / (self.dx ** 2) * self.__get_system_matrix(ndofs)
        rhs = self.__get_rhs(ndofs)

        super(Poisson1D, self).__init__(ndofs, A, rhs, *args, **kwargs)

    @staticmethod
    def __get_system_matrix(ndofs):
        """Helper routine to get the system matrix discretizing :math:`-Delta` with second order FD

        Args:
            ndofs (int): number of inner grid points (no boundaries!)
        Returns:
            scipy.sparse.csc_matrix: sparse system matrix A
                of size :attr:`ndofs` x :attr:`ndofs`
        """
        data = np.array([[2] * ndofs, [-1] * ndofs, [-1] * ndofs])
        diags = np.array([0, -1, 1])
        return sp.spdiags(data, diags, ndofs, ndofs, format='csc')

    @staticmethod
    def __get_rhs(ndofs):
        """Helper routine to set the right-hand side

        Args:
            ndofs (int): number of inner grid points (no boundaries!)
        Returns:
            numpy.ndarray: the right-hand side vector of size :attr:`ndofs`
        """
        return np.zeros(ndofs)

    @property
    def u_exact(self):
        """Routine to compute the exact solution

        Returns:
            numpy.ndarray: exact solution array of size :attr:`ndofs`
        """
        return np.zeros(self.ndofs)

    @property
    def domain(self):
        return np.array([(i + 1) * self.dx for i in range(self.ndofs)])

    @ProblemBase.ndofs.setter
    def ndofs(self, val):

        ProblemBase.ndofs.fset(self, val)
        self.dx = 1.0 / (val + 1)
        # compute system matrix A, scale by 1/dx^2
        self.A = 1.0 / (self.dx ** 2) * self.__get_system_matrix(val)
        self.rhs = self.__get_rhs(self._ndofs)


class Poisson1DPeriodic(ProblemBase):
    """Implementation of the 1D Poission problem.

    Here we define the 1D Poisson problem :math:`-\Delta u = 0` with
    Dirichlet-Zero boundary conditions. This is the homogeneous problem,
    derive from this class if you want to play around with different RHS.

    Attributes:
        dx (float): mesh size
    """

    def __init__(self, ndofs, sigma, *args, **kwargs):
        """Initialization routine for the Poisson1D problem

        Args:
            ndofs (int): number of degrees of freedom (see
                :attr:`pymg.problem_base.ProblemBase.ndofs`)
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments
        """
        self.dx = 1.0 / ndofs
        self.sigma = sigma
        # compute system matrix A, scale by 1/dx^2
        A = self.__get_system_matrix(ndofs)
        A[0, -1] = A[0, 1]
        A[-1, 0] = A[1, 0]
        A = -sigma * 1.0 / (self.dx ** 2) * A
        rhs = self.__get_rhs(ndofs)

        super(Poisson1DPeriodic, self).__init__(ndofs, A, rhs, *args, **kwargs)

    @staticmethod
    def __get_system_matrix(ndofs):
        """Helper routine to get the system matrix discretizing :math:`-Delta` with second order FD

        Args:
            ndofs (int): number of inner grid points (no boundaries!)
        Returns:
            scipy.sparse.csc_matrix: sparse system matrix A
                of size :attr:`ndofs` x :attr:`ndofs`
        """
        data = np.array([[2] * ndofs, [-1] * ndofs, [-1] * ndofs])
        diags = np.array([0, -1, 1])
        return sp.spdiags(data, diags, ndofs, ndofs, format='csc')

    @staticmethod
    def __get_rhs(ndofs):
        """Helper routine to set the right-hand side

        Args:
            ndofs (int): number of inner grid points (no boundaries!)
        Returns:
            numpy.ndarray: the right-hand side vector of size :attr:`ndofs`
        """
        return np.zeros(ndofs)

    @property
    def u_exact(self):
        """Routine to compute the exact solution

        Returns:
            numpy.ndarray: exact solution array of size :attr:`ndofs`
        """
        return np.zeros(self.ndofs)

    @property
    def domain(self):
        return np.array([(i) * self.dx for i in range(self.ndofs)])

    @ProblemBase.ndofs.setter
    def ndofs(self, val):
        ProblemBase.ndofs.fset(self, val)
        self.dx = 1.0 / val
        # compute system matrix A, scale by 1/dx^2
        self.A = -self.sigma * 1.0 / (self.dx ** 2) * self.__get_system_matrix(val)
        self.A[0, -1] = self.A[0, 1]
        self.A[-1, 0] = self.A[1, 0]

        self.rhs = self.__get_rhs(self._ndofs)
