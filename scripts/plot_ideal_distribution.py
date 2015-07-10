#!/usr/bin/env
import numpy, ss_plotting
from ss_plotting.make_plots import plot
import matplotlib.pyplot as plt

def gauss(x, mu, sigma):
    return (1./2*numpy.sqrt(sigma*numpy.pi))*numpy.exp(-(x-mu)*(x-mu)/(2*sigma*sigma))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Plot a mixture of gaussians")
    parser.add_argument("--savefile", type=str, default=None,
                        help="The file to save the plot to")
    parser.add_argument("--color-region", action='store_true',
                        help="Color a small region of the plot green")
    args = parser.parse_args()

    mu = [0., -2.5, 2]
    sigma = [1., 0.8, 1.2]
    weights = [0.33, 0.33, 0.34]
    xvals = numpy.linspace(-5, 5, 1000)
    yvals = numpy.zeros(xvals.shape)

    for w, m, s in zip(weights, mu, sigma):
        yvals += w*gauss(xvals, m, s)
    yvals /= max(yvals)-0.01

    startidx = int(0.15*len(xvals))
    endidx = int(0.18*len(xvals))
    xvals2 = xvals[startidx:endidx]
    yvals2 = yvals[startidx:endidx]

    series = [(xvals, yvals)]
    series_colors = ['gray']
    series_color_emphasis = [True]
    if args.color_region:
        series.append((xvals2, yvals2))
        series_colors.append((116, 196, 118))
        series_color_emphasis.append(True)

    fig, ax = plot(series, series_colors=series_colors,
	 series_color_emphasis=series_color_emphasis, show_plot=False)
    ax.set_xticks([])
    ax.set_yticks([])
    ss_plotting.plot_utils.simplify_axis(ax)

    if args.savefile:
        ss_plotting.plot_utils.output(fig, args.savefile,
                                      (2., 2.),fontsize=8)
        print 'Saved plot to file %s' % args.savefile

    plt.show()
