#!/usr/bin/env python
import numpy, random
from sample_2d_actions import sample_random_trajectory
from graph_tools import RenderNode, PathGraph

def distance_goal(path, goal_center, goal_radius):
    
    last_node = path[-1]
    distance = numpy.linalg.norm(numpy.array([last_node.x, last_node.y]) - numpy.array(goal_center))
    
    return max(0, distance - goal_radius)

def sample_noisy_paths(path, noise_samples, sigma):
    # Generate a set of noisy samples of the path
    path_samples = []
    for idx in range(noise_samples.shape[0]):
        new_path = [ path[0] ]
        for pidx in range(1, len(path)):
            pt = path[pidx]
            nidx = pidx*2
            new_node = RenderNode(pt.id,
                                  pt.x + sigma * noise_samples[idx][nidx],
                                  pt.y + sigma * noise_samples[idx][nidx + 1],
                                  color = pt.color,
                                  parent_id = pt.parent_id)
            new_path.append(new_node)
        path_samples.append(new_path)
    return path_samples

def run_cma(path, goal_center, goal_radius, num_iterations=10, debug=False):
    
    num_points_in_path = len(path)
    num_samples = 10
    sigma = 0.05
    C = numpy.eye(num_points_in_path * 2.)
    ccov = 0.5 #(2/n^2)
    zero_mean = numpy.zeros(num_points_in_path * 2.)
    mean_path = path

    # Now compute weights
    mu = num_samples/2.
    w = numpy.log(mu + 0.5) - numpy.log(range(1, int(mu)+1))
    w = w / sum(w)
    mu_w = 1./ sum([w[idx]*w[idx] for idx in range(int(mu))])

    node_id = len(mean_path)
    for idx in range(num_iterations):
        # Sample y_i
        noise_vals = numpy.random.multivariate_normal(zero_mean, C, num_samples)

        # Sample a set of x candidates
        path_samples = sample_noisy_paths(mean_path, noise_vals, sigma)
        
        # Debug
        if args.debug:
            G = PathGraph()
            G.set_goal_region(goal_center, radius=goal_radius)
            G.add_path(mean_path, bold=True)
            for p in path_samples:
                for idx in range(1, len(p)):
                    p[idx].id = node_id
                    node_id += 1
                    p[idx].parent_id = p[idx-1].id
                G.add_path(p)

        # Evaluate each
        dists = [distance_goal(p, goal_center, goal_radius) for p in path_samples]
        dists = zip(range(len(dists)), dists)

        # Now sort in order of increasing f value
        dists = sorted(dists, key=lambda k: k[1])

        # Now move the mean
        kept_idx = [ d[0] for d in dists[:len(w)] ]
        yw = numpy.array([w[idx]*noise_vals[kept_idx[idx],:] for idx in range(len(w))])
        yw = numpy.sum(yw, axis=0)

        # Now update the mean
        path_vals = []
        for n in mean_path:
            path_vals = path_vals + [n.x, n.y] 
        path_vals = numpy.array(path_vals) + sigma*yw

        new_mean_path = [ mean_path[0] ] # same start node
        for idx in range(1, len(mean_path)):
            new_node = RenderNode(node_id,
                                  path_vals[idx*2],
                                  path_vals[idx*2+1],
                                  color = (0., 0., 0.5),
                                  parent_id = new_mean_path[-1].id)
            new_mean_path.append(new_node)
            node_id += 1

        if args.debug:
            G.add_path(new_mean_path, weight=2)
            G.render(edge_color=edge_color,
                     node_size=node_size,
                     savefile=args.savefile,
                     savefile_size=(2., 2.))

        mean_path = new_mean_path
        if distance_goal(mean_path, goal_center, goal_radius) == 0.0:
            break

        yw_tmp = numpy.array([yw])
        yyt = numpy.dot(yw_tmp.T, yw_tmp)
        #C = (1. - ccov) * C + ccov * mu_w * yyt
    
    return mean_path

if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser('Simple script to sample random trajectories and run gradient-free optimization to move them closer to the goal')
    parser.add_argument("--savefile", type=str, default=None,
                        help="The file to save the plot to")
    parser.add_argument("--num-paths", type=int, default=5,
                        help="The number of paths to sample")
    parser.add_argument("--num-actions", type=int, default=5,
                        help="The number of actions to sample per path")
    parser.add_argument("--iterations", type=int, default=5,
                        help="The number of optimization iterations to run")
    parser.add_argument("--debug", action='store_true',
                        help="If true, render the progress of the optimizations")
    args = parser.parse_args()

    bounds = numpy.array([[0.0, 0.5],
                          [-0.5, 0.5],
                          [0.1, 0.5]])

    
    num_actions = args.num_actions
    num_sampled_paths = args.num_paths

    node_color = (0., 0., 0.)
    edge_color = (0., 0., 0.)
    node_size = 3.

    start_pose = numpy.array([0., 0.])
    start_node = RenderNode(0, 0., 0., color = node_color)
    
    goal_center = [0.5, 0.]
    goal_radius = 0.1

    # Generate the random paths
    G = PathGraph()
    G.set_goal_region(goal_center, radius=goal_radius)

    node_id = 1
    path_samples = []
    for pidx in range(num_sampled_paths):
        path = sample_random_trajectory(start_node, num_actions, bounds, node_color = node_color)

        # Now run CMA
        path = run_cma(path, goal_center, goal_radius, num_iterations=args.iterations, debug=args.debug)

        for idx in range(1, len(path)):
            path[idx].id = node_id
            node_id += 1
            path[idx].parent_id = path[idx-1].id
            
        G.add_path(path)

    # Now plot
    G.render(edge_color=edge_color,
             node_size=node_size,
             savefile=args.savefile,
             savefile_size=(2., 2.))

