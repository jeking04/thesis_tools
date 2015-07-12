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

class PathGraph(object):

    def __init__(self):
        self.G = nx.DiGraph()
        self.node_list = {}
        self.goal_region = None

    def set_goal_region(self, center, radius=0.1):
        """
        @param center The center of the goal region (x,y)
        @param radius The radius of the goal region
        """
        self.goal_region = {'center': center,
                            'radius': radius}

    def add_path(self, path):
        """
        @param A list of RenderNode objects that defines a path
        """
        for rnode in path:
            self.G.add_node(rnode.id)
            if rnode.parent_id is not None:
                self.G.add_edge(rnode.parent_id, rnode.id)
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
            if rnode.parent_id is not None:
                edge_list.append((rnode.parent_id, key))
                edge_colors.append(edge_color)

        padding = 0.05
        xvals = [n.x for n in self.node_list.values()]
        yvals = [n.y for n in self.node_list.values()]
        if goal_region is not None:
            xvals += [goal_region['center'][0] - goal_region['radius'],
                      goal_region['center'][0] + goal_region['radius']]
            yvals += [goal_region['center'][1] - goal_region['radius'],
                      goal_region['center'][1] + goal_region['radius']]

        print xvals
        axis_bounds = [min(xvals) - padding,
                       max(xvals) + padding,
                       min(yvals) - padding,
                       max(yvals) + padding]
        print axis_bounds
        print axis_bounds[:2]
        print axis_bounds[2:]

        fig, ax = plt.subplots()
        
        # Setup the axis limits - turn off ticks
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
                               arrows = False)
        
        # Draw teh goal region
        if goal_region is not None:
            gregion = plt.Circle(goal_region['center'],
                                 goal_region['radius'],
                                 color=goal_color, fill=False)
            ax.add_artist(gregion)

        # Turn on plotting of all 4 axis 
        if savefile is not None or show_plot:
            plot_utils.simplify_axis(ax, xtop=True, yright=True)
            
        if savefile is not None:
            plot_utils.output(fig, savefile, savefile_size)

        if show_plot:
            plt.show()
