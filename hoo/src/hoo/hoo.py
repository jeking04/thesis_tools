from abc import abstractmethod

class Range(object):
    @abstractmethod
    def split(self):
        return

    @abstractmethod
    def select_random(self):
        return

    @abstractmethod
    def get_bins(self, num_bins):
        return

class HOONode(object):
    def __init__(self, h, i, R):
        """
        @param H The depth of this node
        @param I The index of this node at depth H
        @param R The range represented by this node
        """
        self.h = h
        self.i = i
        self.N = 0
        self.Y = list()
        
        self.R = R        
        self._children=None

    def __str__(self):
        return '%d_%d' % (self.h, self.i)

    def getBVal(self, n, row, v1):
        """
        @param n The round
        """
        import numpy

        if self.N == 0:
            return float('inf')
        U = numpy.mean(self.Y) + numpy.sqrt(2.*numpy.log(n)/self.N) + v1*pow(row,self.h)
        Bchild = max([c.getBVal(n, row, v1) for c in self.getChildren()])
        return min(U, Bchild)

    def getChildren(self):
        if self._children is None:
            R1, R2 = self.R.split()
            self._children = [HOONode(self.h+1, self.i*2-1, R1), HOONode(self.h+1, self.i*2, R2)]
        return self._children

class HOO(object):

    def __init__(self, R, rfunc, row, v1):
        self.root = None
        self.R = R
        self.rfunc = rfunc
        self.row = row
        self.v1 = v1

    def __str__(self):
        return 'HOO'

    def run(self, n):
        """
        @param n The round
        """
        if self.root is None:
            self.root = HOONode(0, 1, self.R)
        return self._recursive_search(self.root, n)

    def _recursive_search(self, node, n):
        if node.N == 0:
            x = node.R.select_random()
            y = self.rfunc(x)
            node.N = 1
            node.Y = [y]
            return x, y

        children = node.getChildren()        
        idx,val = max(enumerate([c.getBVal(n, self.row, self.v1) for c in children]), 
                      key=lambda v: v[1])
        x, y = self._recursive_search(children[idx], n)

        # Update the node parameters
        node.N += 1
        node.Y += [ y ]

        return x, y
