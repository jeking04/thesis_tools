import copy, logging, numpy
from bandit import BanditAlgorithm

logger = logging.getLogger('bandit_test')

class UCB_E(BanditAlgorithm):
    
    def __init__(self, arm_set, budget, a, data_logger=None):
        BanditAlgorithm.__init__(self, arm_set, budget, data_logger=data_logger)
        self.a = a
        self.name = 'UCB-E'

    def solve(self):
        logger.info('Running UCB-E')

        A = self.arm_set.keys()
        all_results = { arm: [] for arm in A }
        B = { arm: float('inf') for arm in A }
        for i in range(1, self. budget + 1):
            
            # Select an arm
            selected_arm = max(B.keys(), key=lambda x: B[x])

            # Perform the rollout
            rollout_results = self.arm_set[selected_arm].generate(1)
            all_results[selected_arm] += rollout_results

            # Log
            if self.data_logger:
                self.data_logger.log(self.name, copy.deepcopy(all_results))

            # Update the B parameters for the selected arm
            m = numpy.mean(all_results[selected_arm])
            s = len(all_results[selected_arm])
            B[selected_arm] = self._B(m, s)

        # Compute X_hat for every arm
        X_hat = {}
        for arm in A:
            if len(all_results[arm]) == 0:
                X_hat[arm] = 0.
            else:
                X_hat[arm] = numpy.mean(all_results[arm])

        selected_arm = max(X_hat.keys(), key = lambda x: X_hat[x])

        return selected_arm
            

    def _B(self, m, s):
        '''
        @param cls The class
        @param m The empirical mean of the arm after s pulls
        @param s The number of pulls of the arm
        '''
        return m + numpy.sqrt(self.a / s)
