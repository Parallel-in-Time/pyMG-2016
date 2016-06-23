# coding=utf-8

import numpy as np
import scipy as sp
import scipy.linalg as la
import scipy.sparse.linalg as sprsla
from project.pfasst.plot_tools import eigvalue_plot_list, eigvalue_plot_nested_list
from project.pfasst.matrix_method_tools import prolongation_restriction_fourier_diagonals, get_iteration_matrix_v_cycle
from project.poisson1d import Poisson1D
from pymg.smoother_base import SmootherBase
from pymg.problem_base import ProblemBase
from pymg.transfer_base import TransferBase
from pymg.space_time_base import CollocationTimeStepBase, SpaceTimeMultigridBase
from project.pfasst.pfasst import SimplePFASSTSmoother, SimplePFASSTCollocationProblem, SimplePFASSTTransfer, \
    SimplePFASSTMultigrid
from project.solversmoother import SolverSmoother

from project.pfasst.plot_tools import MatrixAnimation


# This classe will all be taylored to work with the PFASST classes,
# if it works with other classes it is just coincidence ...
class SimplePFASSTProblemSetup(object):
    """ This class will contain all information needed to
        setup an testing environment for the multigrid at hand
    """

    def __init__(self, *args, **kwargs):
        # Setup of the SimplePFASSTCollocationProblem
        self.init_value_gen = kwargs['init_value_gen']
        self.num_nodes = kwargs['num_nodes']
        self.num_subintervals = kwargs['num_subintervals']
        self.colloc_class = kwargs['CollocationClass']
        self.colloc = self.colloc_class(self.num_nodes, 0.0, 1.0)
        self.space_problem = kwargs['space_problem']
        self.num_space = self.space_problem.ndofs

        # Setup of the SimplePFASSTMultigridSolver
        self.c_strat = kwargs['c_strat']
        self.nlevels = kwargs['nlevels']
        self.transfer_opts_space = kwargs['transfer_opts_space']
        self.transfer_opts_time = kwargs['transfer_opts_time']
        self.transfer_opts_time['new_nodes'] = self.colloc_class(self.transfer_opts_time['num_new_nodes'], 0.0,
                                                                 1.0).nodes

    def PFASST_problem(self, dt):
        time_step = CollocationTimeStepBase(0.0, dt, self.colloc.Qmat[1:, 1:], self.colloc.nodes)
        return SimplePFASSTCollocationProblem(self.num_subintervals, time_step, self.space_problem, self.init_value_gen)

    def transfer_opts(self):

        return [{'space': self.transfer_opts_space, 'time': self.transfer_opts_time}] * (self.nlevels - 1)

    def solution(self, dt):
        PFASSTCollocProb = self.PFASST_problem(dt)
        return sprsla.spsolve(PFASSTCollocProb.A, PFASSTCollocProb.rhs)

    def __str__(self):
        string = ""
        string += "Number of nodes:\t\t\t" + str(self.num_nodes) + "\nNumber of subintervals\t\t" + str(
            self.num_subintervals) + "\nDegree of freedom in space:\t" + str(self.num_space)
        string += "\n\nUsing " + str(self.nlevels) + " levels with \nTransfer options in space:" + str(
            self.transfer_opts_space) + "\nTransferoptions in time:" + str(self.transfer_opts_time)
        # string += "\n\nFor the collocation problem:" + str(self.colloc)
        # string += "\n\nAnd the space_problem" + str(self.space_problem)
        return string



class SimplePFASSTMultiGridAnalyser(object):
    """ Takes a Multigrid Class and the associated Problem and gives a easy way to
        access the error plots, spectral radii or a simple kind of lfa.
    """

    def __init__(self, problem_setup, pfasst_multigrid_class=SimplePFASSTMultigrid, space_smoother_class=SolverSmoother,
                 smoother_class=SimplePFASSTSmoother,
                 transfer_class=SimplePFASSTTransfer, *args, **kwargs):
        assert issubclass(pfasst_multigrid_class, SpaceTimeMultigridBase)
        assert issubclass(smoother_class, SmootherBase)
        assert issubclass(transfer_class, TransferBase)
        assert isinstance(problem_setup, SimplePFASSTProblemSetup)

        self._smoother_class = smoother_class
        self._space_smoother_class = space_smoother_class
        self._pfasst_mg_class = pfasst_multigrid_class
        self._transfer_class = transfer_class
        self.prob_setup = problem_setup
        self.dt = None
        self._pfasst_multigrid = None

    def generate_pfasst_multigrid(self, dt, *args, **kwargs):
        self.dt = dt
        self._pfasst_multigrid = self._pfasst_mg_class(self.prob_setup.PFASST_problem(dt), self._space_smoother_class,
                                                       self.prob_setup.nlevels, c_strat=self.prob_setup.c_strat)
        self._pfasst_multigrid.attach_smoother(self._smoother_class, *args, **kwargs)
        self._pfasst_multigrid.attach_transfer(self._transfer_class, self.prob_setup.transfer_opts(), *args, **kwargs)

    def check_v_cycles(self, init, max_iter, nu1=0, nu2=1, norm_type=2):
        assert self._pfasst_multigrid is not None, 'the PFASSTMultigridClass is not instantiated'
        u = [init]
        sol = self.prob_setup.solution(self.dt)
        err = [np.linalg.norm(init - sol, norm_type)]
        rhs = self._pfasst_multigrid.problem.rhs
        res = [np.linalg.norm(self._pfasst_multigrid.problem.A.dot(init) - rhs, norm_type)]

        for i in range(max_iter):
            u.append(self._pfasst_multigrid.do_v_cycle_recursive(u[-1], rhs, nu1, nu2, 0))
            err.append(np.linalg.norm(u[-1] - sol, norm_type))
            res.append(np.linalg.norm(self._pfasst_multigrid.problem.A.dot(u[-1]) - rhs, norm_type))

        return err, res

    def check_fmg(self, nu0=1, nu1=0, nu2=1, norm_type=2):
        assert self._pfasst_multigrid is not None, 'the PFASSTMultigridClass is not instantiated'
        rhs = self._pfasst_multigrid.problem.rhs
        sol = self.prob_setup.solution(self.dt)
        u = self._pfasst_multigrid.do_fmg_cycle_recursive(rhs, nu0, nu1, nu2, norm_type)
        return np.linalg.norm(sol - u, norm_type), np.linalg.norm(rhs - self._pfasst_multigrid.problem.A.dot(u),
                                                                  norm_type)

    def check_w_cycles(self, init, max_iter, norm_type=2):
        pass

    def get_v_cycle_it_matrix(self, nu0=0, nu1=1):
        assert self._pfasst_multigrid is not None, 'the PFASSTMultigridClass is not instantiated'
        pre_smooth_list = self._pfasst_multigrid.smoo
        post_smooth_list = self._pfasst_multigrid.smoo
        problem_list = self._pfasst_multigrid.problem_list
        transfer_list = self._pfasst_multigrid.trans
        return get_iteration_matrix_v_cycle(problem_list, pre_smooth_list, post_smooth_list, transfer_list, nu0, nu1)

    def get_eigenvalues_v_cycle(self, nu0=0, nu1=1, all_eigvals=False, k_max=5, k_min=5):
        T, P_inv = self.get_v_cycle_it_matrix(nu0, nu1)
        if all_eigvals:
            return sp.linalg.eigvals(T.todense())
        else:
            eigval_list = sprsla.eigs(T, k=k_max, which='LM', return_eigenvectors=False)
            eigval_list.append(sprsla.eigs(T, k=k_min, which='SM', return_eigenvectors=False))
            return eigval_list

    def get_singularvalues_v_cycle(self, nu0=0, nu1=1, all_svdvals=False, k_max=5, k_min=5):
        T, P_inv = self.get_v_cycle_it_matrix(nu0, nu1)
        if all_svdvals:
            return sp.linalg.svdvals(T.todense())
        else:
            svdval_list = sprsla.svds(T, k=k_max, which='LM', return_eigenvectors=False)
            svdval_list.append(sprsla.svds(T, k=k_min, which='SM', return_eigenvectors=False))
            return svdval_list

    def animate_var_cfl_poisson1d(self, cfls_s, cycle_type='v_cycle', cycle_opts={'nu0': 0, 'nu1': 1},
                                  video_opts={'name': 'test', 'fps': 20, 'dpi': 200}):
        """
        Only works for the poisson1d problem.

        :return:
        """

        assert isinstance(self.prob_setup.space_problem, Poisson1D)
        if cycle_type == 'v_cycle':
            def matrix_generator(cfl):
                dt = cfl * self.prob_setup.space_problem.dx ** 2
                self.generate_pfasst_multigrid(dt)
                return self.get_v_cycle_it_matrix(nu0=cycle_opts['nu0'], nu1=cycle_opts['nu1'])[0]
        else:
            raise NotImplementedError("We only offer the V-cycle")

        matrix_anim = MatrixAnimation(matrix_generator, cfls_s, 'cfl')
        matrix_anim.save(video_opts['name'] + '.mp4', fps=video_opts['fps'], dpi=video_opts['dpi'],
                         extra_args=['-vcodec', 'libx264'])

    def animate_eigvalues_var_cfl_advection1d(self):
        pass


class SmootherAnalyser(object):
    """ Takes a Smoother and the associated Problem and gives an easy way to access
        the error plots and so on.
    """

    def __init__(self, smoother, problem):
        assert isinstance(smoother, SmootherBase), "Not a real Smoother"
        assert isinstance(problem, ProblemBase), "Not a real Problem"

        self.smoother = smoother
        self.problem = problem

        self.sol = sprsla.spsolve(self.problem.A, self.problem.rhs)

    def errors(self, init, max_iter, norm_type=2):
        u = [init]
        err = [np.linalg.norm(init - self.sol, norm_type)]
        for i in range(max_iter):
            u.append(self.smoother.smooth(self.problem.rhs, u[-1]))
            err.append(np.linalg.norm(u[-1] - self.sol, norm_type))

        return err

    def residuals(self, init, max_iter, norm_type=2):
        u = [init]
        res = [np.linalg.norm(self.problem.A.dot(init) - self.problem.rhs, norm_type)]
        for i in range(max_iter):
            u.append(self.smoother.smooth(self.problem.rhs, u[-1]))
            res.append(np.linalg.norm(self.problem.A.dot(u[-1]) - self.problem.rhs, norm_type))
        return res

    def eigvalues(self):
        T_it = sp.sparse.eye(self.problem.ndofs, format='csc') - self.problem.Pinv.dot(self.problem.A)
        return sp.linalg.eigvals(T_it.todense())

    def svdvals(self):
        T_it = sp.sparse.eye(self.problem.ndofs, format='csc') - self.problem.Pinv.dot(self.problem.A)
        return sp.linalg.svdvals(T_it.todense())

    def spec_rad(self):
        T_it = sp.sparse.eye(self.problem.ndofs, format='csc') - self.problem.Pinv.dot(self.problem.A)
        return sprsla.eigs(T_it, k=1, which='LM')


class SAMA_BlockLFA_Analyser(object):
    """ This tool constructs SAMA Blocks and BlockLFA Blocks
        for the postsmoother,system_matrix, coarse grid correction and iterationmatrix of PFASST.
        Additionaly it provides some methods to analyse those Blocks and give a estimate how well
        SAMA with and without BlockLFA works. Works only for periodic case and no coarsening in time.
    """

    def __init__(self, lin_pfasst, eigvalues_space_fine, eigvalues_space_coarse):

        # get everything needed from the lin_pfasst object
        self.A_I = lin_pfasst.block_diag_solver.it_list[0].A_I
        self.A_E = lin_pfasst.block_diag_solver.it_list[0].A_E
        self.Q = lin_pfasst.block_diag_solver.it_list[0].Q
        self.QD = lin_pfasst.block_diag_solver.it_list[0].QD
        self.dt = lin_pfasst.block_diag_solver.it_list[0].dt
        self.N = len(lin_pfasst.block_diag_solver.it_list)
        self.A = self.A_I + self.A_E
        self.N_x = self.A_I.shape[0]
        self.N_t = self.Q.shape[0]
        self.dx = 1.0 / self.N_x
        self.E = np.diag(np.ones(self.N - 1), -1)
        self.L = np.zeros((self.N_t, self.N_t))
        self.L[:, -1] = -1.0
        self.I_N_kron_Q = np.kron(np.eye(self.N), self.Q)
        self.I_N_kron_Q_delta = np.kron(np.eye(self.N), self.QD)
        self.E_kron_L = np.kron(self.E, self.L)
        self.eig_values_fine = map(lambda k: eigvalues_space_fine(self.N_x, k), range(self.N_x))
        self.eig_values_coarse = map(lambda k: eigvalues_space_coarse(self.N_x / 2, k), range(self.N_x / 2))
        # get interpolation and restriction matrix
        self.ipl_x = lin_pfasst.multi_step_solver.transfer_list[0].Pspace[:self.N_x, :self.N_x / 2]
        self.rst_x = lin_pfasst.multi_step_solver.transfer_list[0].Rspace[:self.N_x / 2, :self.N_x]

    def change_N(self, N, T=1.0):
        self.N = N
        self.dt = T / N
        self.E = np.diag(np.ones(self.N - 1), -1)
        self.I_N_kron_Q = np.kron(np.eye(self.N), self.Q)
        self.I_N_kron_Q_delta = np.kron(np.eye(self.N), self.QD)
        self.E_kron_L = np.kron(self.E, self.L)
        print "Warning: I've changed my N to " + str(N)

    def change_dt(self, dt, T=1.0):
        self.dt = dt
        self.T = self.N_t * dt
        print "Warning: I've changed my dt to " + str(dt)

    def change_eigvals(self, eig_f, eig_c):
        self.eig_values_fine = map(lambda k: eig_f(self.N_x, k), range(self.N_x))
        self.eig_values_coarse = map(lambda k: eig_c(self.N_x / 2, k), range(self.N_x / 2))
        print "Warning: I've changed my eigvals "

    def change_eigvals_list(self, eig_f, eig_c):
        self.eig_values_fine = eig_f
        self.eig_values_coarse = eig_c
        print "Warning: I've changed my eigvals "

    def sama_for_M(self):
        return map(lambda l: np.eye(self.N_t * self.N) - l * self.dt * (self.I_N_kron_Q) + self.E_kron_L,
                   self.eig_values_fine)

    def block_lfa_for_M(self):
        I = np.eye(self.N_t)
        return map(lambda j: map(lambda l: I - l * self.dt * (self.Q) + np.exp(-1j * (2 * np.pi * j) / self.N) * self.L,
                                 self.eig_values_fine), range(self.N))

    def compare_eigvalues(self, sama_blocks, sama_lfa_blocks, min_freq=0, max_freq=-1):
        eigvalue_plot_list(sama_blocks)
        eigvalue_plot_nested_list(map(lambda x: x[min_freq:max_freq], sama_lfa_blocks))

    def compare_norms(self, sama_blocks, sama_lfa_blocks, min_freq=0, max_freq=-1, norm_type=2):
        sama_lfa_blocks = map(lambda x: x[min_freq:max_freq], sama_lfa_blocks)
        sama_lfa_blocks = reduce(lambda a, b: a + b, sama_lfa_blocks)
        norm_sama = np.max(map(lambda bl: la.norm(bl, norm_type), sama_blocks))
        norm_block_lfa = np.max(map(lambda bl: la.norm(bl, norm_type), sama_lfa_blocks))
        return norm_sama, norm_block_lfa

    def sama_for_post_smoother(self, omega=1.0):
        l_fine = self.eig_values_fine
        half = self.N_x / 2
        I = np.eye(self.N_t * self.N)
        h_t = self.dt
        B_L_inv = map(lambda k: la.inv(I - l_fine[k] * h_t * (self.I_N_kron_Q_delta)), range(self.N_x))
        B_A = map(lambda k: I - l_fine[k] * h_t * (self.I_N_kron_Q) + self.E_kron_L, range(self.N_x))
        B_ps = map(lambda k: I - omega * np.dot(B_L_inv[k], B_A[k]), range(self.N_x))
        return B_ps

    def block_lfa_for_post_smoother(self, omega=1.0):
        l_fine = self.eig_values_fine
        h_t = self.dt
        I = np.eye(self.N_t)
        B_L_inv = map(lambda k: la.inv(I - h_t * l_fine[k] * self.QD), range(self.N_x))
        B_ps = map(lambda j: map(lambda k: I - omega * np.dot(B_L_inv[k],
                                                              (I - l_fine[k] * h_t * (self.Q) + np.exp(
                                                                  -1j * (2 * np.pi * j) / self.N) * self.L)),
                                 range(self.N_x)), range(self.N))
        return B_ps

    def sama_for_coarse_grid_correction(self, exact_solve_on_coarse_level=False):
        l_fine = self.eig_values_fine
        l_coarse = self.eig_values_coarse
        d, d_tilde, f, f_tilde = prolongation_restriction_fourier_diagonals(self.ipl_x, self.rst_x)
        h_t = self.dt
        I = np.eye(self.N_t * self.N)
        B_G = map(lambda k: I - l_fine[k] * h_t * (self.I_N_kron_Q) + self.E_kron_L, range(self.N_x))
        if exact_solve_on_coarse_level:
            B_K_inv = map(lambda k: la.inv(I - l_coarse[k] * h_t * (self.I_N_kron_Q) + self.E_kron_L),
                          range(self.N_x / 2))
        else:
            B_K_inv = map(lambda k: la.inv(I - l_coarse[k] * h_t * (self.I_N_kron_Q_delta) + self.E_kron_L),
                          range(self.N_x / 2))
        # B_K_tilde_inv = map(lambda B_k: np.dot(np.kron(np.eye(N),P_t),np.dot(B_k,np.kron(np.eye(N),R_t))),B_K_inv)
        # hence there is no coarsening in time :
        B_K_tilde_inv = B_K_inv
        B_CG = map(lambda k: np.vstack([np.hstack([I - f[k] * d[k] * np.dot(B_K_tilde_inv[k], B_G[k]),
                                                   -d[k] * f_tilde[k] * np.dot(B_K_tilde_inv[k],
                                                                               B_G[k + self.N_x / 2])]),
                                        np.hstack([-d_tilde[k] * f[k] * np.dot(B_K_tilde_inv[k], B_G[k]),
                                                   I - f_tilde[k] * d_tilde[k] * np.dot(B_K_tilde_inv[k],
                                                                                        B_G[k + self.N_x / 2])]
                                                  )]), range(self.N_x / 2))
        return B_CG

    def block_lfa_for_coarse_grid_correction(self, exact_solve_on_coarse_level=False):
        l_fine = self.eig_values_fine
        l_coarse = self.eig_values_coarse
        d, d_tilde, f, f_tilde = prolongation_restriction_fourier_diagonals(self.ipl_x, self.rst_x)
        I = np.eye(self.N_t)
        h_t = self.dt

        if exact_solve_on_coarse_level:
            B_K_inv = map(lambda j: map(
                lambda k: la.inv(I - l_coarse[k] * h_t * (self.Q) + np.exp(-1j * (2 * np.pi * j) / self.N) * self.L)
                , range(1, self.N_x / 2)), range(self.N))
        else:
            B_K_inv = map(lambda j: map(
                lambda k: la.inv(I - l_coarse[k] * h_t * (self.QD) + np.exp(-1j * (2 * np.pi * j) / self.N) * self.L)
                , range(1, self.N_x / 2)), range(self.N))

        B_G = map(
            lambda j: map(lambda k: (I - l_fine[k] * h_t * (self.Q) + np.exp(-1j * (2 * np.pi * j) / self.N) * self.L)
                          , range(self.N_x)), range(self.N))

        # under the assumption that the coarse grid correction kills all null modes
        B_K_inv_first = map(lambda j: la.inv(I + np.exp(-1j * (2 * np.pi * j) / self.N) * self.L), range(1, self.N))

        # what we assume for j = 0, k = 0, see afterthoughts
        Z = np.zeros((self.N_t, self.N_t))
        B_CG_very_first = [np.vstack([np.hstack([Z, Z]),
                                      np.hstack([Z, I])])]
        B_CG_first_space_modes = map(lambda j:
                                     np.vstack([
                                         np.hstack([I - f[0] * d[0] * np.dot(B_K_inv_first[j], B_G[j + 1][0]), Z]),
                                         np.hstack([Z, I])])
                                     , range(self.N - 1))

        B_CG = map(lambda j: map(lambda k:
                                 np.vstack([np.hstack(
                                     [I - f[k] * d[k] * np.dot(B_K_inv[j][k - 1],
                                                               B_G[j][k]),
                                      -d[k] * f_tilde[k] * np.dot(B_K_inv[j][k - 1],
                                                                  B_G[j][k + self.N_x / 2])]),
                                     np.hstack(
                                         [-f[k] * d_tilde[k] * np.dot(B_K_inv[j][k - 1], B_G[j][k]),
                                          I - f_tilde[k] * d_tilde[k] * np.dot(B_K_inv[j][k - 1],
                                                                               B_G[j][k + self.N_x / 2])]
                                     )])
                                 , range(1, self.N_x / 2)), range(self.N))

        return [B_CG_very_first + B_CG[0]] + map(lambda k: [B_CG_first_space_modes[k]] + B_CG[k + 1], range(self.N - 1))

    def sama_for_iteration_matrix(self, nu=1, mu=1, exact_solve_on_coarse_level=False):
        """
        Iterationmatrix with nu Post-Smoother steps and mu coarse-grid-correction steps
        """
        sama_ps = self.sama_for_post_smoother()
        sama_cg = self.sama_for_coarse_grid_correction(exact_solve_on_coarse_level)
        sama_ps_nu = map(lambda mat: np.linalg.matrix_power(mat, nu), sama_ps)
        sama_ps_harmonics = map(lambda low, high: la.block_diag(low, high), sama_ps_nu[:self.N_x / 2],
                                sama_ps_nu[self.N_x / 2:])
        sama_cg_mu = map(lambda mat: np.linalg.matrix_power(mat, mu), sama_cg)
        return map(lambda x, y: np.dot(x, y), sama_ps_harmonics, sama_cg_mu)

    def block_lfa_for_iteration_matrix(self, nu=1, mu=1, exact_solve_on_coarse_level=False):
        """
        Iterationmatrix with nu Post-Smoother steps and mu coarse-grid-correction steps
        """
        block_lfa_ps = self.block_lfa_for_post_smoother()
        block_lfa_cg = self.block_lfa_for_coarse_grid_correction(exact_solve_on_coarse_level)
        block_lfa_ps_nu = map(lambda j: map(lambda k: np.linalg.matrix_power(block_lfa_ps[j][k], nu), range(self.N_x)),
                              range(self.N))
        block_lfa_cg_mu = map(
            lambda j: map(lambda k: np.linalg.matrix_power(block_lfa_cg[j][k], nu), range(self.N_x / 2)), range(self.N))
        block_lfa_ps_harmonics = map(
            lambda j: map(lambda k: la.block_diag(block_lfa_ps_nu[j][k], block_lfa_ps_nu[j][self.N_x / 2 + k]),
                          range(self.N_x / 2)), range(self.N))
        return map(
            lambda j: map(lambda k: np.dot(block_lfa_ps_harmonics[j][k], block_lfa_cg[j][k]), range(self.N_x / 2)),
            range(self.N))
