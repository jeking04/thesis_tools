#!/usr/bin/env python
import logging, numpy
from abc import abstractmethod
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)

def reward(state, goal):
    """
    @return The negative of the distance between state and goal
    """
    return -numpy.linalg.norm(state - goal)

def get_initial_state(mean, cov):
    """
    Draw an initial state from a 2-D guassian
    @param mean The nominal initial state
    @param cov The covariance of the guassian to draw from
    """
    return numpy.random.multivariate_normal(mean, cov)

def execute_action(state, action):
    """
    @param state The 2-D pose to start the action from
    @param action The direction to move
    @return The end state
    """
    return state + numpy.array([numpy.cos(action), numpy.sin(action)])

class Action(object):
    @abstractmethod
    def get_action(self, node):
        pass

class UCB1(Action):

    def __init__(self, min_val, max_val, num_bins, c):
        
        bin_size = (max_val - min_val) / num_bins
        self.bins = {}
        for idx in range(num_bins):
            self.bins[idx] = (min_val, min_val+bin_size)
            min_val += bin_size

        self.c = c

    def get_action(self, node):
        """
        Compute UCB1 score for each action
        and select the maximum
        """
        import operator, random
        
        values = { k: float('inf') for k in self.bins.keys() }
        visits = node.get_num_visits()

        for k in self.bins.keys():
            child_node = node.get_child(k)
            if child_node is not None and child_node.get_num_visits() > 0:
                child_visits = child_node.get_num_visits()
                values[k] = child_node.get_value() + self.c*numpy.sqrt(numpy.log(visits)/child_visits)
        aid = max(values.iteritems(), key=operator.itemgetter(1))[0]

        # Now uniformly draw an action from the bin
        b = self.bins[aid]
        a = random.uniform(b[0], b[1])
        return aid, a
            

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description="Example pomcp problem")
    parser.add_argument("--method", choices=['ucb1'], default='ucb1',
                        help="The action selection method to use")
    parser.add_argument("--c", type=float, default=0.,
                        help="The UCB constant")
    parser.add_argument("--iterations", type=int, default=20,
                        help="The number of times to iterate through tree building")
    parser.add_argument("--visualize", action="store_true",
                        help="Visualize the tree")

    args = parser.parse_args()

    if args.method == 'ucb1':
        action = UCB1(0., 2.*numpy.pi, 4, args.c)

    from pomcp import POMCP
    p = POMCP(get_initial_state, reward, execute_action, action.get_action,
              20, 0.95, 0.5)
    
    start = numpy.array([0., 0.])
    goal = numpy.array([5., 5.])

    p.run(start, goal, max_iterations=args.iterations)
    if args.visualize:
        p.visualize()


    import matplotlib.pyplot as plt
    plt.hold(True)

    cov = numpy.array([[0.1, 0.], [0., 0.1]])
    for _ in range(20):
        st = get_initial_state(start, cov)

        path = p.extract_path(st)
        xpoints = [pt[0] for pt in path]
        ypoints = [pt[1] for pt in path]
        plt.plot(xpoints, ypoints, marker='.', markersize=10, color='k')
        plt.plot(st[0], st[1], marker='o', color='blue', markersize=20)
        plt.plot(xpoints[-1], ypoints[-1], 'o', color='red', markersize=20)

    
    plt.plot(goal[0], goal[1], marker='o', color='green', markersize=20)
    plt.show()
