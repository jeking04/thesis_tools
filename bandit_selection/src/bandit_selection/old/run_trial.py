#!/usr/bin/env python
import argparse, copy, csv, logging, os, random, re
from evaluator import *
#from plots.plot_stability_time_success import *

def parse_evalutors(evaluator_list):
    '''
    @param A list of text strings identifying evaluators to use
    @return A list of Evaluator objects
    '''
    evaluators = []

    for ename in evaluator_list:

        # Fixed
        m = re.search('([0-9]+)fixed', ename)
        if m is not None:
            num_rollouts = int(m.group(1))
            logging.info('Adding FixedRolloutEvaluator with %d rollouts', num_rollouts)
            evaluators.append(FixedRolloutEvaluator(num_rollouts))
            continue
            
        # Failure
        m = re.search('([0-9]+)failure', ename)
        if m is not None:
            max_rollouts = int(m.group(1))
            logging.info('Adding FailureCountEvaluator with %d rollouts', num_rollouts)
            evaluators.append(FailureCountEvaluator(max_rollouts))
            continue

        # Bandits
        m = re.search('([0-9]+)bandit', ename)
        if m is not None:
            rollout_budget = int(m.group(1))
            logging.info('Adding SuccessiveRejectsEvaluator with %d budget',
                         rollout_budget)
            evaluators.append(SuccessiveRejectsEvaluator(rollout_budget, 400))
        
        logging.error('Did not recognize evaluator: %s', ename)

    return evaluators

def load_traj_data(infiles):
    '''
    @param infiles A list of files containing rollout data for a set of trajectories
    @return rollout_dict A dictionary mapping traj_id to a list of True/False values
       indicating rollout results
    '''
    rollout_dict = {}
    for infile in infiles:
        logging.info('Loading data file %s' % infile)
        with open(infile, 'r') as f:
            rollout_data = csv.reader(f, delimiter=' ')
            next(rollout_data, None) #skip the header
            for row in rollout_data:
                test_id = row[0]
                rollout_dict[test_id] = [True if v == 'True' else False for v in row[1:]]
    return rollout_dict

def load_ground_truth(infiles):
    '''
    @param infiles The files containing ground truth data
    @return A dictionary mapping trajectory id to ground truth probability
    '''
    ground_truth_dict = {}
    for infile in infiles:
        logging.info('Loading ground truth file %s' % infile)
        with open(infile, 'r') as f:
            gt_data = csv.reader(f, delimiter=' ')
            next(gt_data, None) #skip the header
            for row in gt_data:
                test_id = row[0]
                pval = float(row[1])
                ground_truth_dict[test_id] = pval
    return ground_truth_dict


def run_trial(traj_list, rollout_dict, evaluator_list):
    '''
    @param traj_list The list of all trajectories to use during the trial
    @param rollout_dict A dictionary of rollouts 
        (key: traj  - should be in traj_list, 
         value: list of True/False values indicating success or failure of rollout)
    @return A dictionary that maps evaluator to results
    '''
    # First permute the list of trajectories
    random.shuffle(traj_list)

    # Next permute the rollouts for each trajectory
    rollout_order = {}
    for traj in traj_list:

        # grab the raw rollout list
        rollout_list = copy.copy(rollout_dict[traj])

        # permute it
        random.shuffle(rollout_list)
        rollout_order[traj] = rollout_list

    results = {}
    for evaluator in evaluator_list:
        logging.debug('Solving with evluator: %s' % evaluator.name)
        results[evaluator] = evaluator.solve(traj_list, rollout_order)

    return results
            

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Run a batch set of tests against the given trajectories")
    parser.add_argument("--infile", required=True, 
                        help="A set of files containing rollout data for a set of trajectories")
    parser.add_argument("--numtrials", default=10, type=int,
                        help="The number of trials to run")
    parser.add_argument("--evaluators", required=True, nargs='+',
                        help="The evaluators to use (ex: 30fixed, 150failure, 50bandit)")
    parser.add_argument("--savedir", default=None, 
                        help="The directory to save results to")
    parser.add_argument("--ground", required=True,
                        help="A set of files containing ground truth probabilities for every trajectory in infiles")
    parser.add_argument("--saveplot", dest="saveplot", default=None,
                        help="The filename to save the plot")
    parser.add_argument("--debug", action="store_true",
                       help="Pring debug info")

    args = parser.parse_args()
    
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    savedir = args.savedir
    if savedir is not None and os.path.exists(savedir):
        print 'Removing all data in %s' % savedir
        import shutil
        shutil.rmtree(savedir)
        os.makedirs(savedir)
    

    evaluators = parse_evalutors(args.evaluators)
    if len(evaluators) != len(args.evaluators):
        print 'Failed to parse evaluators'
        exit(0)
    
    rollout_dict = load_traj_data([args.infile])
    logging.info('Loaded %d trajectories with %d rollouts each' % (len(rollout_dict.keys()), len(rollout_dict.values()[0])))
    
    ground_truth = load_ground_truth([args.ground])

    for idx in range(args.numtrials):
        logging.info('Running trial %d of %d' % (idx, args.numtrials))
        
        trial_results = run_trial(rollout_dict.keys(), rollout_dict, evaluators)
        if savedir is None:
            continue

        # Save off the results
        trial_directory = os.path.join(savedir, 'test%d' % idx)
        if not os.path.exists(trial_directory):
            os.makedirs(trial_directory)

        for ev, result in trial_results.items():
            outfile = os.path.join(trial_directory, '%s.csv' % ev.name)
            with open(outfile, 'w') as f:
                header = ['traj_id', 'num_rollouts', 'p_est', 'p_act']
                f.write(' '.join(header))
                f.write('\n')
                for pt in result:
                    traj_id = str(pt[2])
                    p_est = pt[0]
                    p_act = ground_truth[traj_id]
                    num_rollouts = pt[1]
                    data_row = [traj_id, str(num_rollouts), str(p_est), str(p_act)]
                    f.write(' '.join(data_row))
                    f.write('\n')

    print 'Done'
    #if savedir is not None:
        #interval = get_probability_range(args.ground)
        #make_plot(savedir, interval, savefile=args.saveplot)
