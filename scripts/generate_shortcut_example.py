#!/usr/bin/env python
import copy, random
from graph_tools import RenderNode, PathGraph

if __name__=='__main__':
    import argparse
    
    parser = argparse.ArgumentParser('Simple script to generate a set of shortcuts of a base path')
    parser.add_argument("--savefile", type=str, default=None,
                        help="The file to save the plot to")
    parser.add_argument("--num-paths", type=int, default=5,
                        help="The number of paths to sample")
    args = parser.parse_args()

    node_size=3.

    path_orig_pts = [(0., 0.), 
                     (.1, .1), 
                     (.2, -.1), 
                     (0.25, .0), 
                     (.32, .08), 
                     (.45, .02)]
    
    path_orig = []
    node_id = 0
    for pt in path_orig_pts:
        parent_id = None
        if(len(path_orig) > 0):
            parent_id = path_orig[-1].id
        node = RenderNode(node_id, pt[0], pt[1], 
                          parent_id = parent_id)
        path_orig.append(node)
        node_id += 1
        
    # Generate the random paths
    G = PathGraph()
    G.set_goal_region((.5, 0.), radius=0.1)

#    G.add_path(path_orig, bold=True)
    for sidx in range(args.num_paths):
        # Sample a random number of nodes to remove
        # Must remove something, can't remove either endpt
        num_kept = random.randint(2, len(path_orig)) 
        print num_kept

        # Now sample which to remove
        inds = range(1, len(path_orig)-1)
        random.shuffle(inds)

        kept = sorted([0, len(path_orig)-1] + inds[:num_kept-2])
        print kept
        new_path = [path_orig[0]]
        for k in kept[2:]:
            new_node = copy.copy(path_orig[k])
            new_node.id = node_id
            new_node.parent_id = new_path[-1].id
            new_node.x = path_orig[k].x + random.gauss(0., 0.02)
            new_node.y = path_orig[k].y + random.gauss(0., 0.02)
            new_path.append(new_node)
            node_id += 1
            print 'Adding node: ', new_node
        G.add_path(new_path)
        print 'Adding path'

    G.render(node_size=node_size, savefile=args.savefile, savefile_size=(2., 2.))
    

