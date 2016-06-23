# coding=utf-8
import numpy as np
import scipy.sparse as sp
import scipy.interpolate as intpl
import block_transfer as bt

from pymg.transfer_base import TransferBase
from pymg.problem_base import BlockProblemBase, ProblemBase
from pymg.smoother_base import SmootherBase
from pymg.space_time_base import SpaceTimeProblemBase, CollocationTimeStepBase, SpaceTimeMultigridBase
from project.solversmoother import SolverSmoother
from copy import deepcopy


class SimplePFASSTCollocationProblem(BlockProblemBase, SpaceTimeProblemBase):
    """ A problem class especially suited for the PFASST solver in multigrid form.

        E.g. the decomposition of the time domain is accessible from this class.
        This class is a simple version, because the same time_stepping method and the
        same space_problem is used on each subinterval.

    """

    def __init__(self, N_subintervals, coll_step, space_problem, u_init_gen, *args, **kwargs):
        """Initialization routine for a problem

        Args:
            N_subintervals (int): Number of subintervals
            space_problem (pymg.ProblemBase): Space-Problem object
            coll_base (pymg.CollocationTimeStepBase): timestepping method
            u_init (numpy.ndarray): initial value
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments
        """

        # we use private attributes

        assert isinstance(coll_step, CollocationTimeStepBase), 'Please provide a CollocationTimeStepBase object'
        self._coll_step = coll_step
        self._t_step = coll_step

        assert isinstance(space_problem, ProblemBase), 'Please provide a ProblemBase object'
        self._s_prob = space_problem

        assert callable(u_init_gen), 'u_init_generator should be at least callable'
        self._u_init_gen = u_init_gen

        # this works only for classes
        u_init = u_init_gen(space_problem.domain)

        assert u_init.size == space_problem.ndofs, 'U_init does not match dimension of space_time problem'
        self._u_init = u_init

        self._N_sub = None
        self._t_step = None

        # set the attributes
        self.N_sub = N_subintervals

        # use BlockProblemBase
        As = self.__get_As()
        rhs = self.__get_rhs(self._coll_step, self._N_sub, self._u_init)
        # print rhs.shape
        # print map(lambda arow: map(lambda x: x.shape, arow), As)
        super(SimplePFASSTCollocationProblem, self).__init__(As=As, block_rhs=rhs, space_problem=space_problem,
                                                             time_stepping=coll_step, *args, **kwargs)

    @staticmethod
    def __get_rhs(coll_step, N_sub, u_init):
        M = coll_step.ndofs
        L = N_sub
        return np.kron(np.asarray([1] * M + [0] * M * (L - 1)), u_init)

    @staticmethod
    def matrix_N(tau, rows=-1, last_value=1.0):
        n = tau.shape[0]
        if rows == -1:
            rows = n
        N = np.zeros((rows, n))
        # construct the lagrange polynomials
        circulating_one = np.asarray([1.0] + [0.0] * (n - 1))
        lag_pol = []
        for i in range(n):
            lag_pol.append(intpl.lagrange(tau, np.roll(circulating_one, i)))
            N[:, i] = -np.ones(rows) * lag_pol[-1](last_value)
        return sp.csc_matrix(N)

    @staticmethod
    def matrix_E(dim):
        if dim > 1:
            return sp.diags([1], [-1], shape=(dim, dim)).tocsc()
        else:
            return sp.csc_matrix([1])

    @property
    def ndofs_time(self):
        return self.coll_step.ndofs * self.N_sub

    @property
    def u_init_gen(self):
        return self._u_init_gen

    @SpaceTimeProblemBase.ndofs_time.setter
    def ndofs_time(self, ndofs):
        SpaceTimeProblemBase.ndofs_time.fset(self, self.N_sub * ndofs)

    @property
    def space_problem(self):
        return self._s_prob

    def __get_As(self):
        I_L = sp.eye(self._N_sub).tocsc()
        I_N = sp.eye(self._s_prob.ndofs).tocsc()
        I_M = sp.eye(self._coll_step.ndofs).tocsc()
        Q = sp.csc_matrix(self._coll_step.Q)
        A = -self._s_prob.A
        E = self.matrix_E(self._N_sub)
        N = self.matrix_N(self._coll_step.nodes, rows=-1, last_value=self._coll_step.nodes[-1])

        return [[I_L, I_M, I_N], [-I_L, Q * self.coll_step.dt, A], [E, N, I_N]]

    @property
    def N_sub(self):
        return self._N_sub

    @N_sub.setter
    def N_sub(self, N_subintervals):
        assert isinstance(N_subintervals, int), 'Please use only integer values for ndofs'
        assert N_subintervals > 0, 'Please use at least one subinterval'
        self._N_sub = N_subintervals

    @property
    def coll_step(self):
        return self._coll_step

    @coll_step.setter
    def coll_step(self, coll_step):
        assert isinstance(coll_step, CollocationTimeStepBase), 'Please use only CollocationTimeStepBase objects'
        self._coll_step = coll_step

    @property
    def u_init(self):
        return self._u_init

    @u_init.setter
    def u_init(self, u_init):
        assert u_init.size == self._s_prob.ndofs, 'Initial value does not fit the space problem'
        self._u_init = u_init
        self.rhs = self.get_rhs

    def get_all_nodes(self):
        return np.concatenate(map(lambda j: self.coll_step.t_0 + self.coll_step.nodes + self.coll_step.dt * j,
                                  range(self._N_sub)))


class SimplePFASSTSmoother(SmootherBase):
    """ A simple PFASST smoother, which is an approximate BlockGaussSeidel on the coarsest level
        and a approximative BlockJacobi solver the other levels.
    """

    def __init__(self, simple_pfasst_problem, space_smoother_class=SolverSmoother, is_coarsest=False, *args, **kwargs):
        """ Initialisation routine

        :param simple_pfasst_problem:
        :param is_coarsest:
        :param args:
        :param kwargs:
        :return:
        """
        assert isinstance(simple_pfasst_problem, SimplePFASSTCollocationProblem)
        self._sp_prob = simple_pfasst_problem
        I_L = sp.eye(self._sp_prob.N_sub).tocsc()
        I_N = sp.eye(self._sp_prob.space_problem.ndofs).tocsc()
        I_M = sp.eye(self._sp_prob.coll_step.ndofs).tocsc()
        QD = self.__get_QD(self._sp_prob.coll_step.nodes)
        A = self._sp_prob.space_problem.A

        self._space_smoother = space_smoother_class(A, *args, **kwargs)
        assert hasattr(self._space_smoother, 'P')
        A_precond = self._space_smoother.P
        # A_precond_inv = self._space_smoother.Pinv
        E = self._sp_prob.matrix_E(self._sp_prob.N_sub)
        N = self._sp_prob.matrix_N(self._sp_prob.coll_step.nodes, rows=-1,
                                   last_value=self._sp_prob._coll_step.nodes[-1])

        super(SimplePFASSTSmoother, self).__init__(simple_pfasst_problem.A, *args, **kwargs)

        if is_coarsest:
            # make an approximative Gauss-Seidel smoother
            As = [[I_L, I_M, I_N], [I_L, QD * self._sp_prob.coll_step.dt, A], [E, N, I_N]]
            self.P = BlockProblemBase.generate_A(As)
            self.Pinv = sp.linalg.inv(self.P)
        else:
            # make an approximative BlockJacobi smoother
            As = [[I_L, I_M, I_N], [I_L, QD * self._sp_prob.coll_step.dt, A_precond]]
            self.P = BlockProblemBase.generate_A(As)
            self.Pinv = sp.kron(I_L, sp.linalg.inv(
                BlockProblemBase.generate_A([[I_M, I_N], [QD * self._sp_prob.coll_step.dt, A_precond]])), format='csc')

    def __get_QD(self, nodes):
        # self.left_is_node = False
        # self.right_is_node = True
        # e.g. GaussRadau
        tau = np.concatenate([nodes[0:1], nodes[1:] - nodes[:-1]])
        n = tau.shape[0]
        Q_delta = np.zeros((n, n))
        i = 0
        for t in tau:
            Q_delta[i:, i] = np.ones(n - i) * t
            i += 1
        return sp.csc_matrix(Q_delta)

    def smooth(self, rhs, u_old):
        """
        Routine to perform a smoothing step

        Args:
            rhs (numpy.ndarray): the right-hand side vector, size
                :attr:`pymg.problem_base.ProblemBase.ndofs`
            u_old (numpy.ndarray): the initial value for this step,
                size :attr:`pymg.problem_base.ProblemBase.ndofs`

        Returns:
            numpy.ndarray: the smoothed solution u_new of size
                :attr:`pymg.problem_base.ProblemBase.ndofs`
        """
        u_new = u_old + self.Pinv.dot(rhs - self.A.dot(u_old))
        return u_new


class SimplePFASSTTransfer(TransferBase):
    """ Depending on two SimplePFASST problems

    Attributes:
        I_2htoh (scipy.sparse.csc_matrix): prolongation matrix
        I_hto2h (scipy.sparse.csc_matrix): restriction matrix
    """

    def __init__(self, simple_pfasst_problem, coarse_strat_tuple, opts=None, *args, **kwargs):
        """Initialisation routine for transferoperators

        :param simple_pfasst_problem:
        :param coarse_strat: tuple which contains 3 bools, first is for coarsening in space second for coarsening in
                             the subintervals and third for the coarsening in the nodes.
        :param opts: contains the order for interpolation and restriction in space, time and nodes
        :param args:
        :param kwargs:
        :return:
        """
        self._sp_prob = simple_pfasst_problem
        if opts is None:
            # take default values
            self._i_ord_space = 2
            self._r_ord_space = 2
            self._i_ord_time = 2
            self._r_ord_time = 2
            self._periodic = False
        else:
            self._sp_prob = simple_pfasst_problem
            self._i_ord_time = opts['time']['i_ord']
            self._r_ord_time = opts['time']['r_ord']
            self._periodic = opts['space']['periodic']
            self._i_ord_space = opts['space']['i_ord']
            self._r_ord_space = opts['space']['r_ord']

        # space
        if coarse_strat_tuple[0] is True:

            f_grid = self._sp_prob.space_problem.domain
            c_grid = f_grid[0::2]
            # TODO : find a better idea to get the coarse grid
            space_trans = bt.PolynomialTransfer(f_grid, c_grid, periodic=self._periodic, T=1.0,
                                                i_ord=self._i_ord_space, r_ord=self._r_ord_space)
            self.I_2htoh_space = space_trans.I_2htoh
            self.I_hto2h_space = space_trans.I_hto2h
        else:
            self.I_2htoh_space = sp.eye(self._sp_prob.space_problem.ndofs, format='csc')
            self.I_hto2h_space = self.I_2htoh_space

        # same number of subintervals but new number of nodes
        if not coarse_strat_tuple[1] and coarse_strat_tuple[2]:
            self._f_nodes = self._sp_prob.coll_step.nodes
            self._c_nodes = opts['time']['new_nodes']
            node_trans = bt.PolynomialTransfer(self._f_nodes, self._c_nodes, periodic=self._periodic, T=1.0,
                                               i_ord=self._i_ord_time, r_ord=self._r_ord_time)

            self.I_2htoh_time = sp.kron(sp.eye(self._sp_prob.N_sub, format='csc'),
                                        node_trans.I_2htoh, format='csc')
            self.I_hto2h_time = sp.kron(sp.eye(self._sp_prob.N_sub, format='csc'),
                                        node_trans.I_hto2h, format='csc')

        # no coarsening in time, the really simple PFASST
        elif not coarse_strat_tuple[1] and not coarse_strat_tuple[2]:
            self.I_2htoh_time = sp.eye(self._sp_prob.ndofs_time, format='csc')
            self.I_hto2h_time = sp.eye(self._sp_prob.ndofs_time, format='csc')

        # coarsening in number of subintervals
        elif coarse_strat_tuple[1]:
            assert self._sp_prob.N_sub % 2 == 0, "I can't divide those intervals"
            # new nodes on new coarse subinterval
            if coarse_strat_tuple[2]:
                self._c_nodes = opts['time']['new_nodes']
                n_nodes = self._sp_prob.coll_step.nodes
                self._f_nodes = np.concatenate((n_nodes * 0.5, n_nodes * 0.5 + 0.5))

            # old nodes on new coarse subinterval
            else:
                self._c_nodes = self._sp_prob.coll_step.nodes
                self._f_nodes = np.concatenate((self._c_nodes * 0.5, self._c_nodes * 0.5 + 0.5))

            node_time_trans = bt.PolynomialTransfer(self._f_nodes, self._c_nodes, periodic=False, T=1.0,
                                                    i_ord=self._i_ord_time, r_ord=self._r_ord_time)

            self.I_2htoh_time = sp.kron(sp.eye(self._sp_prob.N_sub / 2, format='csc'),
                                        node_time_trans.I_2htoh, format='csc')
            self.I_hto2h_time = sp.kron(sp.eye(self._sp_prob.N_sub / 2, format='csc'),
                                        node_time_trans.I_hto2h, format='csc')

        # now space and time interpolation and restriction matrices are constructed, we compose them
        self.I_2htoh = sp.kron(self.I_2htoh_time, self.I_2htoh_space, format='csc')
        self.I_hto2h = sp.kron(self.I_hto2h_time, self.I_hto2h_space, format='csc')

        super(SimplePFASSTTransfer, self).__init__(self._sp_prob.ndofs, self.I_hto2h.shape[0])

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


class SimplePFASSTMultigrid(SpaceTimeMultigridBase):
    """ The Simple PFASST algorithm in Multigrid form
    """

    def __init__(self, simple_pfasst_problem, space_solver, nlevels, *args, **kwargs):
        """Initialisation routine

        :param simple_pfasst_problem:
        :param n_levels:
        :param args:
        :param kwargs:
        :return:
        """

        assert isinstance(simple_pfasst_problem,
                          SimplePFASSTCollocationProblem), \
            'This class is a special snowflake and takes only a special ProblemClass'
        assert issubclass(space_solver, SmootherBase)

        self._sp_prob = simple_pfasst_problem
        self._space_solver = space_solver
        # first try to achieve a well-balanced coarsening strategy
        if 'c_strat' not in kwargs.keys():
            def c_strat(nx, nt):
                cfl = self._sp_prob.coll_step.dt * self._sp_prob.N_sub * nx ** 2 / nt
                if cfl > 1:
                    return nx / 2, nt
                else:
                    return nx, nt / 2
        else:
            c_strat = kwargs['c_strat']

        super(SimplePFASSTMultigrid, self).__init__(nlevels, self._sp_prob.space_problem.ndofs,
                                                    self._sp_prob.ndofs_time, c_strat)

        self._sp_prob_list = self.__generate_problem_hierarchy()

    @property
    def problem_list(self):
        return self._sp_prob_list

    @property
    def problem(self):
        return self._sp_prob_list[0]

    def __generate_problem_hierarchy(self):
        """
        Generates a list of SimplePFASSTProblems based on ndofs list and the initial SimplePFASSTProblem
        :return:
        """
        # N_subintervals, coll_step, space_problem, u_init
        sp_problem_list = [self._sp_prob]

        for i in range(1, self.nlevels):
            print 'ndofs_list, when problem hierarchy is generated', self.ndofs_list
            new_space_problem = deepcopy(sp_problem_list[-1].space_problem)
            new_space_problem.ndofs = self._ndofs_list[i][0]
            new_N_sub = self._ndofs_list[i][1] / self._sp_prob.coll_step.ndofs
            sp_problem_list.append(
                SimplePFASSTCollocationProblem(new_N_sub, sp_problem_list[-1].coll_step, new_space_problem,
                                               sp_problem_list[-1].u_init_gen))
        return sp_problem_list

    def attach_smoother(self, space_time_smoother_class, *args, **kwargs):
        """Routine to attach a smoother to each level

        Args:
            smoother_class (see :class:`pymg.smoother_base.SmootherBase`): the class of smoother
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments
        """
        self.smoo = map(lambda x: space_time_smoother_class(x, self._space_solver, False, *args, **kwargs),
                        self._sp_prob_list[:-1])
        self.smoo.append(space_time_smoother_class(self._sp_prob_list[-1], SolverSmoother, True, *args, **kwargs))

    def attach_transfer(self, st_transfer_class, opt_list, *args, **kwargs):
        """Routine to attach transfer operators to each level (except for the coarsest)

        Args:
            transfer_class (see :class:`pymg.transfer_base.TransferBase`): the class of transfer ops
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments
        """

        for l in range(self.nlevels - 1):
            coarse_strat = (self._ndofs_list[l][0] / self._ndofs_list[l + 1][0] == 2,  # coarsening in space
                            self._ndofs_list[l][1] / self._ndofs_list[l + 1][1] == 2,
                            # coarsening by merging subintervals
                            False)  # use always the same nodes
            self.trans.append(st_transfer_class(self._sp_prob_list[l], coarse_strat, opt_list[l], *args, **kwargs))

