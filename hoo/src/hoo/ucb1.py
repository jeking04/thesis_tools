#!/usr/bin/env python

class UCBNode(object):
    def __init__(self, R):
        self.N = 0
        self.Y = list()
        self.R = R

    def getUCBVal(self, n):
        """
        @param n The current round
        """
        if self.N == 0:
            return float('inf')
        
        import numpy
        return numpy.mean(self.Y) + numpy.sqrt(2*numpy.log(n)/self.N)

class UCB1(object):

    def __init__(self, R, rfunc, num_bins):
        self.R = R
        self.rfunc = rfunc
        bins = R.get_bins(num_bins)
        self.nodes = [UCBNode(b) for b in bins]

    def __str__(self):
        return 'UCB1'

    def run(self, n):
        idx, val = max(enumerate([node.getUCBVal(n) for node in self.nodes]),
                       key=lambda v: v[1])
        b = self.nodes[idx]
        x = b.R.select_random()
        y = self.rfunc(x)
        
        b.N += 1
        b.Y += [y]

        return x, y
        
