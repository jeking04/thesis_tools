import numpy
import matplotlib.pyplot as plt
from ss_plotting import make_plots
colors = ['blue', 'green', 'red', 'purple', 'black', 'orange', 'pink']

class DataLogger(object):

    def __init__(self, arms):
        self.pts = {}
        self.arms = arms
    
    def log(self, algo, all_results):
        if algo not in self.pts:
            self.pts[algo] = []
        self.pts[algo].append(all_results)

    def plot_arm_dist(self, dist):
        num_samples = 100000
        if dist == 'uniform':
            pvals = numpy.random.random_sample(num_samples)
        elif dist == 'gauss-left':
            pvals = numpy.random.normal(0.3, 0.1, num_samples)
        elif dist == 'gauss-right':
            pvals = numpy.random.normal(0.7, 0.1, num_samples)

        edges = numpy.arange(0., 1., 0.01)
        vals, edges = numpy.histogram(pvals, edges, density=True)
        centers = [edges[i] + 0.5*(edges[i+1] - edges[i]) for i in range(len(edges)-1)]
        
        make_plots.plot([(centers, vals)], ['grey'], 
                        group_color_emphasis=[True],
                        plot_ylim=[0, max(vals)+0.01],
                        show_plot=True)
            
    def plot_time_selection(self):

        arms = [ (k, v.p) for k,v in self.arms.iteritems() ]
        sorted_arms = sorted(arms, key = lambda x: x[1])

        data_sets = []
        data_labels = []
        for algo in self.pts.keys():
            
            data = []
            
            for pt in self.pts[algo]:
                
                budget = sum([len(v) for v in pt.values()])
                best = self._get_best(pt)
                best_idx = [x[0] for x in sorted_arms].index(best)
                
                data.append((budget, best_idx))
                
            xvals = [pt[0] for pt in data]
            yvals = [pt[1] for pt in data]
            data_sets.append((xvals, yvals))
            data_labels.append(algo)
        
        make_plots.plot(data_sets, colors[:len(data_sets)], 
                        [True for x in data_sets],
                        group_labels=data_labels, 
                        plot_xlabel = 'Budget',
                        plot_ylabel = 'Selected Arm')

    @staticmethod
    def _get_best(all_results):
        
        X_hat = { arm: numpy.mean(all_results[arm]) for arm in all_results.keys()}
        
        return max(X_hat.keys(), key = lambda x: X_hat[x])
