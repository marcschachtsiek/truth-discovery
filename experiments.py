import os
import json
import timeit
import argparse

from tqdm import tqdm
from voter import Optimal
from dataset import Dataset
from itertools import product

from voter import *
from dataset.distribution import *

#def get_parameters(n_sources, n_dataitems, n_distinct, sd_merge_func=zip, sd_scale=1):
#    params_raw = list(product(sd_merge_func(n_sources, n_dataitems), n_distinct))
#    params = []
#    for val in params_raw:
#        params.append((val[0][0] * sd_scale, val[0][1] * sd_scale, val[1]))
#    return params

def get_distributions(coverage, truth, distinct, spread):
    return list(product(coverage, truth, distinct, spread))

def run_experiments(filename, params, distribs, algorithms, save_interval, start_index, stop_index):
    iteration = 0
    
    filename = os.path.join("results", filename + ".json")

    if os.path.isfile(filename):
        raise FileExistsError
            
    records = {
            'n_sources': params['n_sources'],
            'n_dataitems': params['n_dataitems'],
            'n_distinct': params['n_distinct'],
            'algorithms_info': [algo._get_info() for algo in algorithms],
            'experiments': []
        }

    #options = list(product(get_parameters(**params), get_distributions(**distribs)))
    for distributions in (pbar := tqdm(get_distributions(**distribs))):

        if iteration < start_index: 
            pbar.set_postfix_str("Skipping")
            iteration += 1
            continue

        pbar.set_postfix_str("Dataset")

        ds = Dataset(*params.values(), *distributions, verbose=0)
        claims = ds.get_claims()
 
        infos = {
            'coverage_dist': distributions[0]._get_info(),
            'truth_dist': distributions[1]._get_info(),
            'distinct_dist': distributions[2]._get_info(),
            'spread_dist': distributions[3]._get_info(),
            'optimal': ds.compare(Optimal().run(claims, ds.truth)),
            'n_claims': len(claims), 
            'iteration_index': iteration,
            'results': {}
        }

        for algo in algorithms:
            pbar.set_postfix_str(algo.__class__.__name__)
            results = {}
                        
            start_time = timeit.default_timer()
            values = algo.run(claims)
            results['time'] = timeit.default_timer() - start_time
            
            results['scores'] = ds.compare(values)
            infos['results'][algo.__class__.__name__] = results
        
        pbar.set_postfix_str("Saving")

        records['experiments'].append(infos)
        if iteration % save_interval == 0:
            with open(filename, "w") as f:
                json.dump(records, f, indent = 4)

        iteration += 1
        
        if stop_index is not None and iteration >= stop_index: break
        #if iteration == 3: return

    with open(filename, "w") as f:
        json.dump(records, f, indent = 4)

def main():
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--filename', '-f', type=str, default='todo')
    parser.add_argument('--n_sources', '-s', type=int, default=1000)
    parser.add_argument('--n_dataitems', '-o', type=int, default=1000)
    parser.add_argument('--n_distinct', '-d', type=int, default=20)
    parser.add_argument('--index', '-x', type=int, default=None, choices=range(7))
    parser.add_argument('--save_interval', '-i', type=int, default=20)
    parser.add_argument('--start_index', type=int, default=0)
    parser.add_argument('--stop_index', type=int, default=None)

    args = vars(parser.parse_args())

    params = {
        'n_sources': args['n_sources'],
        'n_dataitems': args['n_dataitems'],
        'n_distinct': args['n_distinct']
    }

    distribs_list = [
        TruncPareto(), 
        TruncExponential(1), 
        TruncExponential(1/2), 
        Uniform(0, 1), 
        TruncPareto(flipped=True), 
        TruncExponential(1, flipped=True), 
        TruncExponential(1/2, flipped=True)
    ]

    if args['index'] is None:
        filename = args['filename']
        distribs = {'coverage': distribs_list, 'truth': distribs_list, 
                    'distinct': distribs_list, 'spread': distribs_list}
    else:
        distribs = {'coverage': distribs_list, 'truth': distribs_list, 
                    'distinct': distribs_list, 'spread': [distribs_list[args['index']]]}
        filename = args['filename'] + f"-part{args['index']}"

    if args['start_index'] != 0:
        filename += "_" + str(args['start_index'])
    if args['stop_index'] is not None:
        filename += "-" + str(args['start_index'])

    algorithms = [Majority(), TruthFinder(base_trust=0.001), 
                  TwoEstimates(base_trust=0.001), ThreeEstimates(base_trust=0.001)]

    run_experiments(filename, params, distribs, algorithms, args['save_interval'], args['start_index'], args['stop_index'])

if __name__ == "__main__":
    main()