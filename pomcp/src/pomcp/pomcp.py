#!/usr/bin/env python
import logging, numpy
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)

class POMCPNode(object):
    
    def __init__(self, B, name=None):
        """
        @param B The initial set of samples representing
          the initial belief state
        """
        self._B = B
        self._N = 0
        self._V = 0
        self._children = dict()
        self.visited = False
        self.name = name

    def draw_random(self):
        """
        Return a state from the belief uniformly at random
        """
        if len(self._B) == 0:
            raise Exception('No elements in belief')
        import random
        return random.choice(self._B)

    def add_state(self, s):
        """
        @param s The state to add to the belief represented by this node
        """
        self._B += [s]

    def get_num_visits(self):
        """
        @return The number of time this node has been visited
        """
        return self._N

    def add_visit(self):
        """
        Incriment the visit count for the node
        """
        self._N += 1

    def get_children(self):
        """
        @return All child nodes
        """
        return self._children.values()

    def get_child(self, a):
        """
        @param a The action to get the child for
        @return The child node, None if a child has not be created for this action
        """
        if a in self._children:
            return self._children[a]
        return None

    def add_child(self, aid, node):
        """
        @param aid The id of the action that created the child
        @param node The child node
        """
        self._children[aid] = node
        
    def get_value(self):
        """
        @return The V value of this node
        """
        return self._V

    def update_value(self, R):
        """
        @param R The reward achieved
        @return Update the value of this node
        """
        self._V += (R - self._V)/self._N

class POMCP(object):

    def __init__(self, init_fn, reward_fn, execute_fn, action_fn,
                 belief_size, gamma, epsilon):
        self.init_fn = init_fn
        self.reward_fn = reward_fn
        self.execute_fn = execute_fn
        self.action_fn = action_fn

        self.belief_size = belief_size
        self.epsilon = epsilon
        self.gamma = gamma

        self.root = None

    def run(self, start, goal, max_iterations=10):
        
        cov = numpy.array([[0.1, 0.], [0., 0.1]])
        B = [self.init_fn(start, cov) for _ in range(self.belief_size)]
        self.root = POMCPNode(B, 'root')

        for idx in range(max_iterations):
            s = self.root.draw_random()
            logger.debug('Executing iteration %d (start state: %s)', idx, str(s))
            self._simulate(s, self.root, 0, goal)

    def _simulate(self, s, node, depth, goal):
        
        # If we have reached maximum depth, just return
        if numpy.power(self.gamma, depth) < self.epsilon:
            return 0
         
        # Update the visit count of the node
        node.add_visit()

        # Update the belief state of the node
        node.add_state(s)

        # If this is the first visit to the node, we are at a leaf
        #  run a rollout
        if node.get_num_visits() == 1:
            r = self._rollout(s, node, depth, goal)
        else:

            # Select an action
            aid, a = self.action_fn(node)

            # Find the associated child node
            child_node = node.get_child(aid)

            if child_node is None:
                child_node = POMCPNode([], name='%s_%s' % (node.name, aid))
                node.add_child(aid, child_node)
                
            s_new = self.execute_fn(s, a)

            # Recursive search
            r = self.gamma *self._simulate(s_new, child_node, depth+1, goal)

        node.update_value(r)
        return r

    def _rollout(self, s, node, depth, goal):
        return self.reward_fn(s, goal)

        
    def visualize(self):
        if self.root is None:
            raise Exception('No tree to visualize.')

        import networkx as nx
        import matplotlib.pyplot as plt

        # Initialize a graph
        G = nx.DiGraph()
        pos = dict()
        labels = dict()

        # Compute the max depth of the tree - to be used to build the layout
        max_depth = self._compute_depth(self.root, 0)

        # Max x width
        xlim=[0., 10.]

        # Build the visualization
        self._recursive_visualize(self.root, G, 0, pos, labels, xlim[0], xlim[1], max_depth)

        # Visualize
        nx.draw(G, arrows=False, pos=pos, labels=labels, node_color=(1., 1., 1), node_size=50)
        plt.axis('off')
        plt.xlim([-1., 11])
        plt.gca().invert_yaxis()
        plt.show()
        
    def _compute_depth(self, node, depth):
        max_depth = 0
        for c in node.get_children():
            max_depth = self._compute_depth(c, depth+1)
        return max(depth, max_depth)

    def _recursive_visualize(self, node, G, depth, pos, labels, left, width, max_depth):
        """
        @param node The node
        @param G The current graph
        @param depth The depth of the node
        @param pos A dictionary being built to store node positions for rendering
        @param width The of the area this node is in, used to compute positions
        @param max_depth The max-depth of the tree, used to compute positions
        """

        G.add_node(node.name)
        pos[node.name] = (left + 0.5*width,depth)
        labels[node.name] = '\n\n\n\nN=%d\nV=%0.3f' % (node.get_num_visits(), node.get_value())

        children = node.get_children()
        for idx, c in enumerate(children):
            w = width / len(children)
            n = self._recursive_visualize(c, G, depth+1, pos, labels, left+idx*w, w, max_depth)
            G.add_edge(node.name, n.name)
        return node
    
    def extract_path(self, start):
        """
        Extract a path through the tree by recursively selecting
        the highest value action
        @param start The start state
        """
        path = self._recursive_extract(start, self.root)
        return [start] + path

    def _recursive_extract(self, s, node):
        aid, a = self.action_fn(node)
        s_new = self.execute_fn(s, a)
        child_node = node.get_child(aid)
        if child_node is None:
            return [s_new]
        path = self._recursive_extract(s_new, child_node)
        return [s_new] + path
