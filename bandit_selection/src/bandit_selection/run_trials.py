import argparse, logging, numpy
from distributions.binomial import BinomialDistribution
from algorithms.equal_allocation import EqualAllocation
from algorithms.successive_rejects import SuccessiveRejects
from algorithms.ucb_e import UCB_E
from utils.data_logger import DataLogger

logger = logging.getLogger('bandit_test')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run a trial")

    parser.add_argument("--arm-dist", type=str,
                        choices=['uniform', 'gauss-left', 'gauss-right'],
                        default='uniform',
                        help="The distribution to draw the arms from")
    parser.add_argument("--num-arms", required=True, type=int,
                        help="The number of arms to select from")
    parser.add_argument("--budget", type=int, default=1000,
                        help="The total number of tests that can be performed")
    parser.add_argument("--algo", type=str, choices=['equal', 'ucbe', 'sr'], nargs='+',
                        default=['sr'],
                        help="The algorithm to use")
    args = parser.parse_args()

    num_arms = args.num_arms
    budget = args.budget

    logging.basicConfig(level = logging.DEBUG)

    logger.info('Selecting best of %d arms using %d tests' % (num_arms, budget))

    if args.arm_dist == 'uniform':
        pvals = numpy.random.random_sample(num_arms)
    elif args.arm_dist == 'gauss-left':
        pvals = numpy.random.normal(0.3, 0.1, num_arms)
    elif args.arm_dist == 'gauss-right':
        pvals = numpy.random.normal(0.7, 0.1, num_arms)
    else:
        logger.error('Unrecognized arm-dist parameter: %s' % args.arm_dist)
        exit(0)

    arms = {}
    for idx in range(num_arms):
        arms[idx] = BinomialDistribution(pvals[idx])
        logger.info('\t%d: %0.3f' % (idx, pvals[idx]))

    data_logger = DataLogger(arms)

    algos = []

    for name in args.algo:
        if name == 'equal':
            algos += [ EqualAllocation(arms, budget, data_logger = data_logger) ]
        elif name == 'ucbe':
            algos += [ UCB_E(arms, budget, a = 50, data_logger = data_logger) ]
        elif name == 'sr':
            algos += [ SuccessiveRejects(arms, budget, data_logger = data_logger) ]
        else:
            logger.error("Unrecognized algorithm: %s" % name)

    for algo in algos:
        selected_arm = algo.solve()

        logger.info('%s: Selected arm %d: p = %0.3f' % (algo.name, selected_arm, arms[selected_arm].p))

    data_logger.plot_time_selection()
