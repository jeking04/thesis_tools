#!/usr/bin/env python
import argparse, math, numpy
import scipy.special
from ss_plotting import make_plots
from scipy.stats import norm

distributions = {'gauss-left': [0.3, 0.3],
                 'gauss-right': [0.7, 0.1]}

def draw_arms(dist, num_arms):
    if dist == 'uniform':
        arms = numpy.random.random_sample(num_arms)
    elif dist == 'gauss-left':
        arms = []
        while len(arms) < num_arms:
            samples = numpy.random.normal(distributions[dist][0], distributions[dist][1], num_arms)
            arms += [s for s in samples if s >= 0.0 and s <= 1.0]
        arms = arms[:num_arms]
    elif dist == 'gauss-right':
        arms = []
        while len(arms) < num_arms:
            samples = numpy.random.normal(distributions[dist][0], distributions[dist][1], num_arms)
            arms += [s for s in samples if s >= 0.0 and s <= 1.0]
        arms = arms[:num_arms]
    return arms

def compute_expected_value(arms, max_selected_arms = None):
    arms = numpy.sort(arms)
    num_arms = len(arms)

    if max_selected_arms is None:
        max_selected_arms = num_arms

    counts = numpy.zeros((num_arms, max_selected_arms))
    for num_selected in range(0,max_selected_arms):
        total_possible = scipy.special.binom(num_arms, num_selected+1) * math.factorial(num_selected)
        counts[:,num_selected] = [scipy.special.binom(a, num_selected)*math.factorial(num_selected) for a in range(num_arms)]
        counts[:,num_selected] /= float(total_possible)

    expected_vals = numpy.dot(arms, counts)
    return expected_vals


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run a trial")

    parser.add_argument("--num-arms", required=True, type=int,
                        help="The number of arms to select from")    
    parser.add_argument("--max-selected", type=int, default=None,
                        help="The max number of selected arms to consider")
    parser.add_argument("--arm-dists", type=str,
                        choices=['uniform', 'gauss-left', 'gauss-right'],
                        default=['uniform'],
                        nargs='+',
                        help="The distribution to draw the arms from")
    args = parser.parse_args()

    num_arms = args.num_arms
    num_dists = len(args.arm_dists)
#    colors = ['blue', 'green', 'red', 'black', 'orange', 'pink']
    colors = [ (189, 189, 189), (99, 99, 99) ]

    # Generate ground truth pvalues for each arm
    all_data = []
    all_labels = []
    for dist in args.arm_dists:
        arms = draw_arms(dist, num_arms)
        expected_vals = compute_expected_value(arms, max_selected_arms = args.max_selected)
        all_data.append((range(1, num_arms+1), expected_vals))        
        all_labels.append(dist)

    # Plot expected value
    make_plots.plot(all_data, 
                    series_colors=colors[:num_dists], 
                    series_color_emphasis=[True for x in range(num_dists)],
             #       series_labels=all_labels,
                    plot_ylabel='Expected success probability',
             #       plot_xlabel='Size of $\Pi_{finite}$',
                    savefile='expected.pdf',
                    savefile_size=(2.0, 2.0))

    # Plot distribution
    all_data = []
    for dist in args.arm_dists:
        edges = numpy.arange(0, 1, 0.001)
        if dist == 'gauss-left':
            vals = norm.pdf(edges, distributions[dist][0], distributions[dist][1])
        elif dist == 'gauss-right':
            vals = norm.pdf(edges, distributions[dist][0], distributions[dist][1])
        else:
            vals = [1 for e in edges]

        all_data.append((edges, vals))

    make_plots.plot(all_data, 
                    series_colors=colors[:num_dists],
                    series_color_emphasis=[True for a in all_data],
                    plot_ylim=[0, max(vals)+0.01],
                    yaxis_on=False,
                    plot_xlabel='Success probability',
                    savefile='dists.pdf',
                    savefile_size=(2.0, 2.0))
