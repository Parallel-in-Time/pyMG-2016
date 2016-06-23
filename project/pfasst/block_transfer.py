import numpy as np
import scipy.sparse as sp
from pymg.transfer_base import TransferBase
from project.pfasst.transfer_tools import interpolation_matrix_1d
from pymg.problem_base import BlockProblemBase


class EquidistantBlockTransfer(TransferBase):
    """ Implementation of prolongation and restriction between BlockProblems
    In this case we assume that in each direction the meshes are equidistant.

    Attributes:
        I_2htoh (scipy.sparse.csc_matrix): prolongation matrix
        I_hto2h (scipy.sparse.csc_matrix): restriction matrix
    """

    def __init__(self, fine_bp, coarse_bp, int_orders, rstr_orders, *args, **kwargs):
        """Initialization routine for transfer operators

        Args:
            fine_bp : BlockProblem on fine level
            coarse_bp : Blockproblem on coarse level
            int_orders (list of int):
            coarse_orders (list of int):
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments
        """

        assert isinstance(fine_bp, BlockProblemBase)
        assert isinstance(coarse_bp, BlockProblemBase)
        assert fine_bp.n_layer == len(int_orders) and coarse_bp.n_layer == len(rstr_orders)

        f_grids = map(lambda x: np.linspace(0, 1, x.shape[0]), fine_bp.As[0])
        c_grids = map(lambda x: np.linspace(0, 1, x.shape[0]), coarse_bp.As[0])

        intpl_matrices = map(lambda fg, cg, x: interpolation_matrix_1d(fg, cg, x),
                             f_grids, c_grids, int_orders)
        rstr_matrices = map(lambda fg, cg, x: 0.5 * interpolation_matrix_1d(fg, cg, x).transpose(),
                            f_grids, c_grids, rstr_orders)

        self.I_2htoh = reduce(lambda a, b: sp.kron(a, b, format='csc'), intpl_matrices)
        self.I_hto2h = reduce(lambda a, b: sp.kron(a, b, format='csc'), rstr_matrices)
        super(EquidistantBlockTransfer, self).__init__(fine_bp.ndofs, coarse_bp.ndofs, *args, **kwargs)

    def restrict(self, u_coarse):
        """Routine to apply restriction

        Args:
            u_coarse (numpy.ndarray): vector on coarse grid, size `ndofs_coarse`
        Returns:
            numpy.ndarray: vector on fine grid, size `ndofs_fine`
        """
        return self.I_hto2h.dot(u_coarse)

    def prolong(self, u_fine):
        """Routine to apply prolongation

        Args:
            u_fine (numpy.ndarray): vector on fine grid, size `ndofs_fine`
        Returns:
            numpy.ndarray: vector on coarse grid, size `ndofs_coarse`
        """
        return self.I_2htoh.dot(u_fine)


class CombinedBlockTransfer(TransferBase):
    def __init__(self, transfer_list, *args, **kwargs):
        for t in transfer_list:
            assert hasattr(t, 'I_2htoh') and hasattr(t, 'I_hto2h')
        self.I_2htoh = reduce(lambda a, b: sp.kron(a.I_2htoh, b.I_2htoh, format='csc'), transfer_list)
        self.I_hto2h = reduce(lambda a, b: sp.kron(a.I_hto2h, b.I_hto2h, format='csc'), transfer_list)
        n_f = reduce(lambda a, b: a.ndofs_fine * b.ndofs_fine, transfer_list)
        n_c = reduce(lambda a, b: a.ndofs_coarse * b.ndofs_coarse, transfer_list)
        super(CombinedBlockTransfer, self).__init__(n_f, n_c)

    def restrict(self, u_coarse):
        """Routine to apply restriction

        Args:
            u_coarse (numpy.ndarray): vector on coarse grid, size `ndofs_coarse`
        Returns:
            numpy.ndarray: vector on fine grid, size `ndofs_fine`
        """
        return self.I_hto2h.dot(u_coarse)

    def prolong(self, u_fine):
        """Routine to apply prolongation

        Args:
            u_fine (numpy.ndarray): vector on fine grid, size `ndofs_fine`
        Returns:
            numpy.ndarray: vector on coarse grid, size `ndofs_coarse`
        """
        return self.I_2htoh.dot(u_fine)


class PolynomialTransfer(TransferBase):
    """ Class which uses a lagrange-polynomial basis to construct the interpolation
        and restriction matrices.

    """

    def __init__(self, fine_nodes, coarse_nodes, periodic=False, T=1.0, i_ord=2, r_ord=2, *args, **kwargs):
        """ Initialization routine for transfer operators in 1d

        :param fine_nodes: nodes on the fine level
        :param coarse_nodes: nodes on the coarse level
        :param periodic: wether or not the boundaries are
        :param args: Variable length argument list
        :param kwargs: Arbitrary keyword arguments
        :return:
        """
        assert isinstance(fine_nodes, np.ndarray), 'fine_nodes should be a numpy array'
        assert isinstance(coarse_nodes, np.ndarray), 'coarse_nodes should be a numpy array'
        assert isinstance(periodic, bool)
        assert isinstance(i_ord, int) and i_ord > 0 and i_ord < fine_nodes.size
        assert isinstance(r_ord, int) and r_ord > 0 and r_ord < fine_nodes.size

        self._periodic = periodic
        self._fine_nodes = fine_nodes
        self._coarse_nodes = coarse_nodes
        self.I_2htoh = interpolation_matrix_1d(fine_nodes, coarse_nodes, i_ord,
                                               'csc', periodic, T)
        self.I_hto2h = 0.5 * interpolation_matrix_1d(fine_nodes, coarse_nodes, r_ord,
                                                     'csc', periodic, T).T

        super(PolynomialTransfer, self).__init__(fine_nodes.size, coarse_nodes.size)

    @property
    def periodic(self):
        return self._periodic

    def restrict(self, u_coarse):
        """Routine to apply restriction

        Args:
            u_coarse (numpy.ndarray): vector on coarse grid, size `ndofs_coarse`
        Returns:
            numpy.ndarray: vector on fine grid, size `ndofs_fine`
        """
        return self.I_hto2h.dot(u_coarse)

    def prolong(self, u_fine):
        """Routine to apply prolongation

        Args:
            u_fine (numpy.ndarray): vector on fine grid, size `ndofs_fine`
        Returns:
            numpy.ndarray: vector on coarse grid, size `ndofs_coarse`
        """
        return self.I_2htoh.dot(u_fine)
