import numpy

class BinomialDistribution():

    def __init__(self, p):
        self.p = p

    def generate(self, n):
        return numpy.random.binomial(1, self.p, n).tolist()
        
