#!/usr/bin/env python
import logging, numpy

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
            

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description="Example pomcp problem")
    parser.add_argument("--method", choices=['ucb1', 'gps'], default='ucb1',
                        help="The action selection method to use")
    parser.add_argument("--c", type=float, default=0.,
                        help="The UCB constant")
    parser.add_argument("--iterations", type=int, default=20,
                        help="The number of times to iterate through tree building")
    parser.add_argument("--visualize", action="store_true",
                        help="Visualize the tree")

    args = parser.parse_args()

    if args.method == 'ucb1':
        from action import UCB1
        action = UCB1(0., 2.*numpy.pi, 4, args.c)
    elif args.method == 'gps':
        from action import GPS
        action = GPS(0., 2.*numpy.pi)

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
