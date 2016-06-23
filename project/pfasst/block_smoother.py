# coding=utf-8
import numpy as np
import scipy.sparse as sp
from pymg.smoother_base import BlockSmootherBase
from pymg.problem_base import BlockProblemBase


class WeightedBlockJacobi(BlockSmootherBase):
    """Implementation of a BlockJacobi Smoother
       by taking only the diagonal elements in the first layer.

    Attributes:
        P (scipy.sparse.csc_matrix): Preconditioner
        Pinv (scipy.sparse.csc_matrix): inverse of the Preconditioner
    """

    def __init__(self, block_problem, omega, set_diagonal=[0], *args, **kwargs):
        """Initialization routine for the smoother

        Args:
            block_problem (scipy.sparse.csc_matrix): A BlockProblem object
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments
        """

        super(WeightedBlockJacobi, self).__init__(block_problem, *args, **kwargs)

        assert len(
            set_diagonal) <= block_problem.n_layer, "not enough layers, choose another dimensions to set diagonal"
        assert min(set_diagonal) >= 0 and max(set_diagonal) < block_problem.n_layer, "this diagonal, does not exist"

        As = block_problem.As
        for a_row in As:
            for i in set_diagonal:
                a_row[i] = sp.spdiags(a_row[i].diagonal(), 0, a_row[i].shape[0], a_row[i].shape[1], format='csc')
        self.P = BlockProblemBase.generate_A(As)
        self.Pinv = omega * sp.linalg.inv(self.P)

    def smooth(self, rhs, u_old):
        """
        Routine to perform a smoothing step

        Args:
            rhs (numpy.ndarray): the right-hand side vector, size
                :attr:`pymg.problem_base.ProblemBase.ndofs`
            u_old (numpy.ndarray): the initial value for this step, size
                :attr:`pymg.problem_base.ProblemBase.ndofs`

        Returns:
            numpy.ndarray: the smoothed solution u_new of size
                :attr:`pymg.problem_base.ProblemBase.ndofs`
        """
        u_new = u_old + self.Pinv.dot(rhs - self.A.dot(u_old))
        return u_new


class BlockGaussSeidel(BlockSmootherBase):
    """Implementation of a BlockGaussSeidel Smoother
       by taking only the diagonal elements in the first layer.

    Attributes:
        P (scipy.sparse.csc_matrix): Preconditioner
        Pinv (scipy.sparse.csc_matrix): inverse of the Preconditioner
    """

    def __init__(self, block_problem, set_tril=[0], *args, **kwargs):
        """Initialization routine for the smoother

        Args:
            block_problem (scipy.sparse.csc_matrix): A BlockProblem object
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments
        """

        super(BlockGaussSeidel, self).__init__(block_problem, *args, **kwargs)

        assert len(set_tril) <= block_problem.n_layer, "not enough layers, choose another dimensions to set diagonal"
        assert min(set_tril) >= 0 and max(set_tril) < block_problem.n_layer, "set_tril is wrong"

        As = block_problem.As
        for a_row in As:
            for i in set_tril:
                a_row[i] = sp.tril(a_row[i], format='csc')
        self.P = BlockProblemBase.generate_A(As)
        self.Pinv = sp.linalg.inv(self.P)

    def smooth(self, rhs, u_old):
        """
        Routine to perform a smoothing step

        Args:
            rhs (numpy.ndarray): the right-hand side vector, size
                :attr:`pymg.problem_base.ProblemBase.ndofs`
            u_old (numpy.ndarray): the initial value for this step, size
                :attr:`pymg.problem_base.ProblemBase.ndofs`

        Returns:
            numpy.ndarray: the smoothed solution u_new of size
                :attr:`pymg.problem_base.ProblemBase.ndofs`
        """
        u_new = u_old + self.Pinv.dot(rhs - self.A.dot(u_old))
        return u_new


class BlockSolverSmoother(BlockSmootherBase):
    """Implementation of a BlockJacobi Smoother
       by taking only the diagonal elements in the first layer.

    Attributes:
        P (scipy.sparse.csc_matrix): Preconditioner
        Pinv (scipy.sparse.csc_matrix): inverse of the Preconditioner
    """

    def __init__(self, block_problem, set_tril=[0], *args, **kwargs):
        """Initialization routine for the smoother

        Args:
            block_problem (scipy.sparse.csc_matrix): A BlockProblem object
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments
        """

        super(BlockSolverSmoother, self).__init__(block_problem, *args, **kwargs)

        # assert len(set_tril) <= block_problem.n_layer, "not enough layers, choose another dimensions to set diagonal"
        # assert min(set_tril) >= 0 and max(set_tril) < block_problem.n_layer, "set_tril is wrong"
        #
        As = block_problem.As
        # for a_row in As:
        #     for i in set_tril:
        #         a_row[i] = sp.tril(a_row[i], format='csc')
        self.P = BlockProblemBase.generate_A(As)
        self.Pinv = sp.linalg.inv(self.P)

    def smooth(self, rhs, u_old):
        """
        Routine to perform a smoothing step

        Args:
            rhs (numpy.ndarray): the right-hand side vector, size
                :attr:`pymg.problem_base.ProblemBase.ndofs`
            u_old (numpy.ndarray): the initial value for this step, size
                :attr:`pymg.problem_base.ProblemBase.ndofs`

        Returns:
            numpy.ndarray: the smoothed solution u_new of size
                :attr:`pymg.problem_base.ProblemBase.ndofs`
        """
        u_new = u_old + self.Pinv.dot(rhs - self.A.dot(u_old))
        return u_new
