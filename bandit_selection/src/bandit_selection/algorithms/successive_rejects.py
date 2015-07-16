import copy, logging, math, numpy, random
from bandit import BanditAlgorithm

logger = logging.getLogger('bandit_test')

class SuccessiveRejects(BanditAlgorithm):

    def __init__(self, arm_set, budget, data_logger = None):
        BanditAlgorithm.__init__(self, arm_set, budget, data_logger = data_logger)
        self.name = 'Successive Rejects'

    def solve(self):
        logger.info('Running SuccessiveRejects')

        results = []
        
        K = len(self.arm_set)
        A = self.arm_set.keys()

        # There are K trajectories. The successive rejects algorithm rejects
        # exactly one trajectory in each phase, so there are K - 1 phases.
        all_results = { arm: [] for arm in A}
        for k in xrange(1, K):
            num_rounds = self._n(self.budget, k, K) - self._n(self.budget, k - 1, K)

            # Execute all rollouts in this phase
            for arm in A:
                rollout_results = self.arm_set[arm].generate(num_rounds)
                all_results[arm] += rollout_results

            # Log
            if self.data_logger:
                self.data_logger.log(self.name, copy.deepcopy(all_results))

            # Recompute X_hat for all arms
            X_hat = { arm: numpy.mean(all_results[arm]) for arm in A }

            # Find the worst trajectory. Randomize in the case of a tie.
            worst_X_hat = min(X_hat[arm] for arm in A)
            worst_candidates = [ arm for arm in A
                                 if abs(X_hat[arm] - worst_X_hat) < 0.0001 ]
            worst_arm = random.choice(worst_candidates)
            A.remove(worst_arm)

        assert len(A) == 1
        best_arm = A.pop()
        return best_arm

    
    @classmethod
    def _n(cls, n, k, K):
        '''
        @param n Total budget of tests
        @param k The phase
        @param K The total number of arms
        '''

        if k == 0:
            return 0
        else:
            return int(math.ceil((1. / cls._log_bar(K))
                                 * (n - K) / (K + 1. - k)))
                
    @staticmethod
    def _log_bar(k):
        return 0.5 + sum(1. / i for i in xrange(2, k + 1))
