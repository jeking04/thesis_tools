#!/usr/bin/env python
import numpy, random
from graph_tools import RenderNode, PathGraph

def sample_random_trajectory(start_node, num_actions, bounds, node_color=(0., 0., 0.)):
    """
    @param start_node A RenderNode describing the start pose of the trajectory
    @param num_actions The number of actions to sample in the trajectory
    @param bounds A 3x2 matrix describing upper and lower bounds for dx, dy, dt
    """
    path = [start_node]

    dx_range = bounds[0,:]
    dy_range = bounds[1,:]
    dt_range = bounds[2,:]

    for aidx in range(num_actions):
        dx = random.uniform(dx_range[0], dx_range[1])
        dy = random.uniform(dy_range[0], dy_range[1])
        dt = random.uniform(dt_range[0], dt_range[1])
        parent_node = path[-1]
        new_node = RenderNode(aidx,
                              parent_node.x + dx*dt,
                              parent_node.y + dy*dt,
                              color=node_color,
                              parent_id=parent_node.id)
        path.append(new_node)
    return path

if __name__=='__main__':
    import argparse
    
    parser = argparse.ArgumentParser('Simple script to generate a plot of sampled 2d actions')
    parser.add_argument("--savefile", type=str, default=None,
                        help="The file to save the plot to")
    parser.add_argument("--num-paths", type=int, default=5,
                        help="The number of paths to sample")
    parser.add_argument("--num-actions", type=int, default=5,
                        help="The number of actions to sample per path")
    args = parser.parse_args()

    dx_range = [0.0, 0.5]
    dy_range = [-0.5, 0.5]
    dt_range = [0.1, 0.5]
    bounds = numpy.array([dx_range, dy_range, dt_range])

    num_actions = args.num_actions
    num_sampled_paths = args.num_paths

    node_color = (0., 0., 0.)
    edge_color = (0., 0., 0.)
    node_size = 3.

    start_pose = numpy.array([0., 0.])
    start_node = RenderNode(0, 0., 0., color = node_color)

    # Generate the random paths
    G = PathGraph()
    G.set_goal_region((.5, 0.), radius=0.1)
    node_id = 1
    for pidx in range(num_sampled_paths):
        path = sample_random_trajectory(start_node, num_actions, bounds)
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
