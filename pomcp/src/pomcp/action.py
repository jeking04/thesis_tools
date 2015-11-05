from abc import abstractmethod
import numpy

class Action(object):
    @abstractmethod
    def get_action(self, node):
        pass

class UCB1(Action):

    def __init__(self, min_val, max_val, num_bins, c):
        
        bin_size = (max_val - min_val) / num_bins
        self.bins = {}
        for idx in range(num_bins):
            self.bins[idx] = (min_val, min_val+bin_size)
            min_val += bin_size

        self.c = c

    def get_action(self, node):
        """
        Compute UCB1 score for each action
        and select the maximum
        """
        import operator, random
        
        values = { k: float('inf') for k in self.bins.keys() }
        visits = node.get_num_visits()

        for k in self.bins.keys():
            child_node = node.get_child(k)
            if child_node is not None and child_node.get_num_visits() > 0:
                child_visits = child_node.get_num_visits()
                values[k] = child_node.get_value() + self.c*numpy.sqrt(numpy.log(visits)/child_visits)
        aid = max(values.iteritems(), key=operator.itemgetter(1))[0]

        # Now uniformly draw an action from the bin
        b = self.bins[aid]
        a = random.uniform(b[0], b[1])
        return aid, a

class GPS(Action):
    
    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val

        self.trees = {}

    def get_action(self, node):
        """
        Build a GPS tree and return the best action
        """        
        if node.name not in self.trees:
            phi = (numpy.sqrt(5) - 1.)/2.
            a = phi*self.min_val + (1. - phi)*self.max_val
            b = (1. - phi)*self.min_val + phi*self.max_val
            self.trees[node.name] = GPSNode(a, b, node, 0)
        _, aid, a = self.trees[node.name].get_action()
        return aid, a

class GPSNode(object):

    def __init__(self, a, b, node, depth):
        self.a = a
        self.aname = '%d_a' % depth
        self.b = b
        self.bname = '%d_b' % depth
        self.node = node
        self.depth = depth

        self.phi = (numpy.sqrt(5) - 1.)/2.        
        self.children = None
        self.c_depth = 5

    def get_children(self):
        if abs(self.a - self.b) < 0.1:
            # Interval is too small
            return None

            
        if self.depth > numpy.log(self.node.get_num_visits())/numpy.log(self.c_depth):
            # Tree is deep enough
            return None

        if self.children is None:
            anew = (1. + self.phi)*self.b - self.phi*self.a
            bnew = (1. + self.phi)*self.a - self.phi*self.b
            self.children = [GPSNode(self.a, bnew, self.node, self.depth+1),
                             GPSNode(anew, self.b, self.node, self.depth+1)]
        return self.children
        
    def get_action(self):

        achild = self.node.get_child(self.aname)
        if achild is None:
            return float('inf'), self.aname, self.a

        bchild = self.node.get_child(self.bname)
        if bchild is None:
            return float('inf'), self.bname, self.b

        values = [(achild.get_value(), self.aname, self.a),
                  (bchild.get_value(), self.bname, self.b)]
        children = self.get_children()
        if children is not None:
            if achild.get_value() > bchild.get_value():
                values.append(children[0].get_action())
            else:
                values.append(children[1].get_action())

        return max(values, key=lambda v: v[0])

