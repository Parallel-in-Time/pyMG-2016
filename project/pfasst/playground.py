import numpy as np
import scipy.sparse.linalg as splinalg
from pymg.collocation_classes import CollGaussRadau_Right
from pymg.space_time_base import CollocationTimeStepBase
from project.pfasst.pfasst import SimplePFASSTCollocationProblem
from project.pfasst.plot_tools import matrix_plot, heat_map
from project.poisson1d import Poisson1D
from project.pfasst.block_smoother import BlockGaussSeidel, WeightedBlockJacobi
from project.pfasst.analyse_tools import SmootherAnalyser, SimplePFASSTProblemSetup, SimplePFASSTMultiGridAnalyser
from project.pfasst.pfasst import *

def show_call_order(cls, methname):
    'Utility to show the call chain'
    classes = [cls for cls in cls.__mro__ if methname in cls.__dict__]
    print '  ==>  '.join('%s.%s' % (cls.__name__, methname) for cls in classes)


if __name__ == "__main__":
    # build SimplePFASSTCollocationProblem
    num_nodes = 3
    num_subintervals = 2
    num_space = 16
    k = 1
    dt = 0.01
    GRC = CollGaussRadau_Right(num_nodes, 0.0, 1.0)
    Q = GRC.Qmat[1:, 1:]
    QD = GRC.QDmat
    # matrix_plot(QD)
    # print QD.shape, Q.shape
    nodes = GRC.nodes
    CTSB = CollocationTimeStepBase(0.0, dt, Q, nodes)
    CTSB_delta = CollocationTimeStepBase(0.0, dt, QD, nodes)
    SpaceProblem = Poisson1D(num_space)
    omega_h = np.linspace(1 / (num_space + 1), 1.0, num_space)
    u_init = np.sin(2 * np.pi * np.linspace(1 / (num_space + 1), 1.0, num_space))
    u_init_gen = lambda x: np.sin(2 * np.pi * x)
    # show_call_order(SimplePFASSTCollocationProblem, '__init__')
    PFASSTCollocProb = SimplePFASSTCollocationProblem(num_subintervals, CTSB, SpaceProblem, u_init_gen)
    PFASSTPrecondProb = SimplePFASSTCollocationProblem(num_subintervals, CTSB_delta, SpaceProblem, u_init_gen)

    sol = splinalg.spsolve(PFASSTCollocProb.A, PFASSTCollocProb.rhs)
    sol_precond = splinalg.spsolve(PFASSTPrecondProb.A, PFASSTPrecondProb.rhs)

    print '-' * 20
    print 'Error between the precond and colloc', np.linalg.norm(sol - sol_precond, 2)
    # heat_map(PFASSTCollocProb.rhs.reshape(-1, num_space))
    # heat_map(sol.reshape(-1, num_space))
    # heat_map(sol_precond.reshape(-1, num_space))
    # heat_map(sol_precond.reshape(-1, num_space)-sol.reshape(-1, num_space))

    # test the smoothers

    # jac_smoother = WeightedBlockJacobi(PFASSTCollocProb,2.0/3.0)
    # gs_smoother = BlockGaussSeidel(PFASSTCollocProb)
    approx_jac_smoother = WeightedBlockJacobi(PFASSTPrecondProb, 2.0 / 3.0)
    approx_gs_smoother = BlockGaussSeidel(PFASSTPrecondProb)

    analyser_approx_jac = SmootherAnalyser(approx_jac_smoother, PFASSTCollocProb)
    analyser_approx_gs = SmootherAnalyser(approx_gs_smoother, PFASSTCollocProb)

    init = np.kron(np.asarray([1] * num_nodes + [1] * num_nodes * (num_subintervals - 1)), u_init)
    print 'First 10 errors of approx block-Jacobi \n', analyser_approx_jac.errors(init, 10)
    print 'First 10 errors of approx block-Gauss-Seidel \n', analyser_approx_gs.errors(init, 10)
    print '-' * 20

    # print 'Next build a really simple PFASST-Solver with two levels, where the space problem is just solved'
    # c_strat = lambda nx, ny: (nx / 2, ny)
    # s_pfasst_mgrid = SimplePFASSTMultigrid(PFASSTCollocProb, SolverSmoother, 4, c_strat=c_strat)
    # print '...done, attach Smoother'
    # s_pfasst_mgrid.attach_smoother(SimplePFASSTSmoother)
    # print '...done, attach TransferOperator'
    # # first set the options
    # opts = {'space': {'i_ord': 4, 'r_ord': 2, 'periodic': False},
    #         'time': {'i_ord': 4, 'r_ord': 2, 'periodic': False, 'new_nodes': CollGaussRadau_Right(5, 0.0, 1.0).nodes}}
    # opt_list = [opts]*3
    # s_pfasst_mgrid.attach_transfer(SimplePFASSTTransfer, opt_list)
    # print '...done, check hierarchy chosen'
    # print 'ndofs_list', s_pfasst_mgrid.ndofs_list
    #
    # one_v_cycle = s_pfasst_mgrid.do_v_cycle_recursive(init, PFASSTCollocProb.rhs, 0, 1, 0)
    # print 'Error after 1 V-cycle', np.linalg.norm(sol - one_v_cycle, 2)


    print 'Start using the SimplePFASSTMultiGridAnalyser'
    # first the setup
    pfasst_setup = SimplePFASSTProblemSetup(init_value_gen=lambda x: np.sin(2 * np.pi * x),
                                            num_nodes=5, num_subintervals=4,
                                            CollocationClass=CollGaussRadau_Right,
                                            space_problem=Poisson1D(128),
                                            c_strat=lambda nx, ny: (nx / 2, ny),
                                            nlevels=2,
                                            transfer_opts_space={'i_ord': 8, 'r_ord': 2, 'periodic': False},
                                            transfer_opts_time={'i_ord': 4, 'r_ord': 2, 'periodic': False,
                                                                'num_new_nodes': 3})
    dt = 0.001
    print pfasst_setup
    # now the analyser with the canonical parts
    print "Assembling the analyser"
    pfasst_analyser = SimplePFASSTMultiGridAnalyser(pfasst_setup)
    pfasst_analyser.generate_pfasst_multigrid(dt)
    u_init = pfasst_setup.init_value_gen(pfasst_setup.space_problem.domain)
    init = np.kron(np.ones(pfasst_setup.num_nodes * pfasst_setup.num_subintervals), u_init)
    err, res = pfasst_analyser.check_v_cycles(init, 10, 1, 1)
    print "Checking the V-Cycle:"
    print 'V-Cycle - errors :\t\t', err
    print 'V-Cycle - residuals:\t', res

    print "Checking if iteration matrix may be constructed"
    # T_v_cycle, P_inv_v_cycle = pfasst_analyser.get_v_cycle_it_matrix()
    # sol_vec = pfasst_setup.solution(dt)
    # err_vec_list = [sol_vec - init]
    # for i in range(9):
    #     err_vec_list.append(T_v_cycle.dot(err_vec_list[-1]))
    # print "V-Cycle - errors:\t\t", map(lambda x: np.linalg.norm(x, 2), err_vec_list)
    # # ARGGGGGHHHHHHHH, passt nicht sofort
