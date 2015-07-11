#!/usr/bin/env python
import os, yaml

if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser(description="Break a belief state up into a set of files representing each state")
    parser.add_argument("--belief", type=str, required=True,
                        help="The belief state to break");
    parser.add_argument("--outdir", type=str, required=True,
                        help="The directory to write out the states to")
    args = parser.parse_args()

    # Load the belief state
    with open(args.belief, 'r') as f:
        bst = yaml.load(f.read())

    # Create the directory if it doesn't exist
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    for idx in range(len(bst)):
        outfile = os.path.join(args.outdir, '%03d.state' % idx)
        with open(outfile, 'w') as f:
            f.write(yaml.dump(bst[idx]))
        print 'Write state %d to file %s' % (idx, outfile)
        
