#!/usr/bin/env python
import networkx as nx
import matplotlib.pyplot as plt
from ss_plotting import plot_utils

class RenderNode(object):
    
    def __init__(self, node_id, x, y, color=(0., 0., 0.), parent_id=None):
        """
        @param node_id The unique id of this node
        @param x The x coordinate of the node
        @param y The y coordinate of the node
        @param color The color of the node
        @param parent_id The id of the parent node, None if this is a root node

        """
        self.id = node_id
        self.parent_id = parent_id
        self.x = x
        self.y = y
        self.color = color
        
    def __str__(self):
        return '{%d: (%0.2f, %0.2f) -> %d' % (self.id, self.x, self.y, self.parent_id if self.parent_id is not None else -1)

class PathGraph(object):

    def __init__(self):
        self.G = nx.DiGraph()
        self.node_list = {}
        self.goal_region = None

    def clear_nodes(self):
        self.node_list = {}

    def set_goal_region(self, center, radius=0.1):
        """
        @param center The center of the goal region (x,y)
        @param radius The radius of the goal region
        """
        self.goal_region = {'center': center,
                            'radius': radius}

    def get_edge_color(self, weight, default=(0., 0., 0.)):
        if weight == 3:
            return (0.5, 0., 0.)
        elif weight == 2:
            return (0., 0., 0.5)
        else:
            return default

    def add_path(self, path, bold=False, weight=1):
        """
        @param A list of RenderNode objects that defines a path
        """
        for rnode in path:
            self.G.add_node(rnode.id)
            if rnode.parent_id is not None:
                self.G.add_edge(rnode.parent_id, rnode.id,
                                weight=3 if bold else weight)
            self.node_list[rnode.id] = rnode
    
    def render(self, edge_color=(0., 0., 0.), node_size=50., goal_color=(.4, .4, .4), padding=0.05, savefile=None, savefile_size=(2., 2.), show_plot=True):
        """
        Render the graph
        @param edge_color The color for the edges in the graph
        @param node_size The size to render nodes (in pixels)
        @param padding The padding to add to the edge of the graph
        @param savefile The name of the file to save the graph image to
        @param savefile_size The size of the output image
        @param show_plot If true, show the plot via plt.show()
        """
        pos = nx.spring_layout(self.G)
        goal_region = self.goal_region
        node_colors = []
        node_list = []
        edge_list = []
        edge_colors = []
        for rnode in self.node_list.values():
            key = rnode.id
            pos[key][0] = rnode.x
            pos[key][1] = rnode.y
            
            node_colors.append(rnode.color)
            node_list.append(key)

        # Add the edges
        edge_list = self.G.edges()
        edge_colors = [self.get_edge_color(self.G[u][v]['weight'], default=edge_color) for u,v in edge_list]
        edge_weights = [self.G[u][v]['weight'] for u,v in edge_list]

        padding = 0.05
        xvals = [n.x for n in self.node_list.values()]
        yvals = [n.y for n in self.node_list.values()]
        if goal_region is not None:
            xvals += [goal_region['center'][0] - goal_region['radius'],
                      goal_region['center'][0] + goal_region['radius']]
            yvals += [goal_region['center'][1] - goal_region['radius'],
                      goal_region['center'][1] + goal_region['radius']]

        axis_bounds = [min(xvals) - padding,
                       max(xvals) + padding,
                       min(yvals) - padding,
                       max(yvals) + padding]

        fig, ax = plt.subplots()
        
        # Setup the axis limits - turn off ticks
        dx = goal_region['center'][0]+goal_region['radius'] + padding
        axis_bounds = ([-padding, dx, -0.5*dx, 0.5*dx])
        ax.set_xlim(axis_bounds[:2])
        ax.set_ylim(axis_bounds[2:])
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Draw the graph
        nx.draw_networkx_nodes(self.G,
                               pos, ax=ax,
                               with_labels=False,
                               nodelist = node_list,
                               node_color = node_colors,
                               node_size = node_size)

        nx.draw_networkx_edges(self.G,
                               pos,
                               ax = ax,
                               edgelist = edge_list,
                               edge_color = edge_colors,
                               width = edge_weights, 
                               arrows = False)
        
        # Draw teh goal region
        if goal_region is not None:
            gregion = plt.Circle(goal_region['center'],
                                 goal_region['radius'],
                                 linestyle='dashed',
                                 color=goal_color, fill=False)
            ax.add_artist(gregion)

        # Turn on plotting of all 4 axis 
        if savefile is not None or show_plot:
            plot_utils.simplify_axis(ax, xtop=True, yright=True)
            
        if savefile is not None:
            plot_utils.output(fig, savefile, savefile_size)

        if show_plot:
            plt.show()
