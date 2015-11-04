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
    return scipy.stats.norm(0.5, 0.2).pdf(x)
    
def visualize_hoo(root):
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

def visualize(xpoints, R, title=None):
    import matplotlib.pyplot as plt
    import numpy

    # Function value
    xvals = numpy.arange(0., 1., 0.005)
    rvals = [ rfunc(x) for x in xvals]
    plt.fill_between(xvals, 0, rvals, facecolor='gray', lw=0,  alpha=0.2)

    plt.hold(True)

    # Bins
    bins = R.get_bins(10)
    endpts = [b.min_val for b in bins]
    endpts += [bins[-1].max_val]
    plt.plot(endpts, [0. for _ in endpts], '+', color='blue', markersize=10)

    # Actual sampled points
    plt.plot(xpoints, [0. for _ in xpoints], 'o', markersize=10, color='gray')
    #plt.axis('off')
    if title is not None:
        plt.title(title)
    plt.ylim([-1., max(rvals)+0.1])

    plt.show()

if __name__ == '__main__':

    import math
    from hoo import HOO
    from ucb1 import UCB1
    from gps import GPS

    R = SingleDimension(0., 1.)
    alpha = 2.
    row = pow(2., -alpha)
    v1 = pow(0.5, alpha)

    hoo = HOO(R, rfunc, row, v1)
    ucb = UCB1(R, rfunc, 10)
    gps = GPS(R, rfunc)

    algos = [hoo, ucb, gps]

    for idx, a in enumerate(algos):
        reward = 0
        xvals = []
        for n in range(300):
            x, r = a.run(n)
            reward += r
            xvals += [x]

        print '%s Total Reward: %0.3f' % (str(a), reward)
        visualize(xvals, R, title=str(a))
        
    
