#!/usr/bin/env python
from hoo import Range
class SingleDimension(Range):
    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val

    def __str__(self):
        return '%0.3f->%0.3f' % (self.min_val, self.max_val)
    
    def split(self):
        min_val = self.min_val
        max_val = self.max_val
        min_point = min_val + (max_val - min_val)*0.5
        R1 = SingleDimension(min_val, min_point)
        R2 = SingleDimension(min_point, max_val)
        return R1, R2

    def select_random(self):
        from random import uniform
        return uniform(self.min_val, self.max_val)

    def get_bins(self, num_bins):
        bin_size = (self.max_val - self.min_val) / num_bins
        min_v = self.min_val
        bins = []
        for _ in range(num_bins):
            bins.append(SingleDimension(min_v, min_v+bin_size))
            min_v += bin_size
        return bins

def rfunc(x):
    import scipy.stats
    return scipy.stats.norm(0.2, 0.1).pdf(x)
    
def visualize(root):
        import networkx as nx
        import matplotlib.pyplot as plt
        G = nx.DiGraph()
        d = dict()
        _, d = _recursive_visualize(root, G, d)
        nx.draw(G, pos=d, labels={k:'' for k in d.keys()})
        plt.axis('off')
        plt.gca().invert_yaxis()
        plt.hold(True)
        max_y = max([v[1] for v in d.values()])

        import numpy
        xvals = numpy.arange(0., 1., 0.05)
        rvals = [max_y - rfunc(x) for x in xvals]
        plt.fill_between(xvals, rvals, max_y, facecolor='blue', alpha=0.5)
        plt.show()

def _recursive_visualize(node, G, d):
    G.add_node(node)
    d[node] = [node.R.min_val + 0.5*(node.R.max_val - node.R.min_val), node.h]
    children = node.getChildren()
    for c in children:
        if c.N > 0:
            n, d = _recursive_visualize(c, G, d)
            G.add_edge(node, n)
    return node, d

if __name__ == '__main__':

    import math
    from hoo import HOO
    from ucb1 import UCB1

    R = SingleDimension(0., 1.)
    alpha = 2.
    row = pow(2., -alpha)
    v1 = pow(0.5, alpha)

    hoo = HOO(R, rfunc, row, v1)
    ucb = UCB1(R, rfunc, 10)
    
    reward_hoo = 0
    reward_ucb = 0
    for n in range(300):
        reward_hoo += hoo.run(n)
        reward_ucb += ucb.run(n)

    print 'HOO Total Reward: %0.3f' % reward_hoo
    print 'UCB Total Reward: %0.3f' % reward_ucb

    visualize(hoo.root)

