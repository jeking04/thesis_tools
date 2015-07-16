import copy, logging, numpy
from bandit import BanditAlgorithm

logger = logging.getLogger('bandit_test')

class EqualAllocation(BanditAlgorithm):

    def __init__(self, arm_set, budget, data_logger = None):
        BanditAlgorithm.__init__(self, arm_set, budget, data_logger=data_logger)
        self.name = 'Equal Allocation'

    def solve(self):
        
        logger.info('Running EqualAllocation')

        A = self.arm_set.keys()
        
        num_rounds = int(numpy.floor(self.budget / float(len(A))))
        
        all_results = {arm: [] for arm in A}
        for r in xrange(num_rounds):
            for arm in A:
                all_results[arm] += self.arm_set[arm].generate(1)

            # Log
            if self.data_logger:
                self.data_logger.log(self.name, copy.deepcopy(all_results))

        # Compute X_hat for every arm
        X_hat = {}
        for arm in A:
            if len(all_results[arm]) == 0:
                X_hat[arm] = 0.
            else:
                X_hat[arm] = numpy.mean(all_results[arm])
        selected_arm = max(X_hat.keys(), key = lambda x: X_hat[x])

        return selected_arm
