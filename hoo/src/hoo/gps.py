#!/usr/bin/env python

class GPS(object):

    def __init__(self, R, rfunc):
        import numpy
        self.phi = (numpy.sqrt(5) - 1.)/2.
        self.a = self.phi*R.min_val + (1. - self.phi)*R.max_val
        self.b = (1. - self.phi)*R.min_val + self.phi*R.max_val
        self.rfunc = rfunc

    def __str__(self):
        return 'GPS'

    def run(self, n):
        
        aval = self.rfunc(self.a)
        bval = self.rfunc(self.b)
        if aval > bval:
            self.b = (1. + self.phi)*self.a - self.phi*self.b
            return self.a, aval
        else:
            self.a = (1. + self.phi)*self.b - self.phi*self.a
            return self.b, bval
