from __future__ import division
import bisect
import logging
import math
import numpy
import random

class Evaluator(object):
    
    def __init__(self, name):
        self.name = name

    def solve(self, traj_list, rollout_dict):
        pass


class FixedRolloutEvaluator(Evaluator):

    def __init__(self, num_rollouts):
        Evaluator.__init__(self, 'fixed_%d' % num_rollouts)
        self.num_rollouts = num_rollouts

    def solve(self, traj_list, rollout_dict):

        results = []

        k_best = -1
        for idx in range(len(traj_list)):
            
            traj = traj_list[idx]            
            rollouts = rollout_dict[traj][:self.num_rollouts]
            k = sum(rollouts)
            p = float(k)/float(self.num_rollouts)
            logging.debug('%s: k = %d, p = %0.3f' % (traj, k, p))
            
            if k > k_best:
                results.append((p, (idx+1)*self.num_rollouts, traj))
                k_best = k
            

        return results

        
class FailureCountEvaluator(Evaluator):
    
    def __init__(self, max_rollouts):
        Evaluator.__init__(self, 'failure_count_%d' % max_rollouts)
        self.max_rollouts = max_rollouts
        

    def solve(self, traj_list, rollout_dict):
        
        results = []
        
        min_failures = self.max_rollouts + 1 # pretend everything failed so far
        rollout_count = 0
        for idx in range(len(traj_list)):
            
            traj = traj_list[idx]
            rollouts = rollout_dict[traj][:self.max_rollouts]

            # now we want to compute cumsum
            rollout_failures = [not r for r in rollouts]
            rollout_failure_counts = numpy.cumsum(numpy.array(rollout_failures))

            v = bisect.bisect(rollout_failure_counts, min_failures)
            if v == len(rollout_failure_counts):
                min_failures = rollout_failure_counts[-1]
                results.append((float(sum(rollouts))/len(rollouts), rollout_count, traj))
                rollout_count += len(rollouts)
            else:
                rollout_count += v

        return results


class SuccessiveRejectsEvaluator(Evaluator):

    def __init__(self, rollout_budget, rollout_step=None):
        Evaluator.__init__(self, 'successive_rejects_%d' % rollout_budget)

        if rollout_step is None:
            rollout_step = int(math.ceil(rollout_budget / 10))

        self.rollout_budget = rollout_budget
        self.rollout_step = rollout_step

    def solve(self, traj_list, rollout_dict):
        results = []

        for n in xrange(2 * self.rollout_step,
                        self.rollout_budget + 1,
                        self.rollout_step):
            result = self.solve_with_budget(traj_list, rollout_dict, n)
            results.append(result)

        return results

    def solve_with_budget(self, traj_list, rollout_dict, n):
        results = []
        num_rollouts = 0

        K = len(traj_list)
        A = set(traj_list)
        rollout_offset = { traj: 0 for traj in traj_list }

        if n <= K:
            raise ValueError('Number of rollouts {:d} is less than the number'
                             ' of trajectories {:d}.'.format(n, K))

        # There are K trajectories. The successive rejects algorithm rejects
        # exactly one trajectory in each phase, so there are K - 1 phases.
        for k in xrange(1, K):
            num_rounds = self._n(n, k, K) - self._n(n, k - 1, K)

            num_rollouts += num_rounds * len(A)

            # Compute the number of rollouts to perform in this phase.
            X_hat = {}

            for traj in A:
                offset = rollout_offset[traj]
                rollouts = rollout_dict[traj][0:offset + num_rounds]
                rollout_offset[traj] += num_rounds
                X_hat[traj] = numpy.mean(rollouts)

            # Find the worst trajectory. Randomize in the case of a tie.
            worst_X_hat = min(X_hat[traj] for traj in A)
            worst_candidates = [ traj for traj in A
                                 if abs(X_hat[traj] - worst_X_hat) < 0.0001 ]
#                                 if numpy.isclose(X_hat[traj], worst_X_hat) ]
            worst_traj = random.choice(worst_candidates)
            A.remove(worst_traj)

        assert len(A) == 1
        best_traj = A.pop()
        return X_hat[best_traj], num_rollouts, best_traj

    @classmethod
    def _n(cls, n, k, K):
        if k == 0:
            return 0
        else:
            return int(math.ceil((1. / cls._log_bar(K))
                                 * (n - K) / (K + 1. - k)))

    @staticmethod
    def _log_bar(k):
        return 0.5 + sum(1. / i for i in xrange(2, k + 1))

