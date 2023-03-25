import timeit

import numpy as np
import pandas as pd

from tqdm import tqdm

MASK_NA = 0
MASK_T = 1
MASK_F = 2

class Dataset():
    
    def __init__(self, n_sources, n_dataitems, n_distinct, coverage_dist, truth_dist, distinct_dist, spread_dist, verbose=0) -> None:
        
        # Values are taken in a batch from the Spread distribution and within that batch normalised to sum to 1.
        # For example: Uniform(x, x) is the same for any x, but if the low and high value differ, the relativ change will be passed forward.
        # The distribution will be essentially discretised to allow for integer value picking
        
        self.verbose = verbose
        self.len_claims = None
        
        self.n_sources = n_sources
        self.n_dataitems = n_dataitems
        self.n_distinct = n_distinct

        self.coverage_dist = coverage_dist.rvs()
        self.truth_dist = truth_dist.rvs()
        self.distinct_dist = distinct_dist.rvs()
        self.spread_dist = spread_dist.rvs()

        if verbose == 1:
            start_time = timeit.default_timer()

        # Generate Source Masks
        self.rows = []
        self.add_truth()
        self.add_claims()
        self.data = pd.concat(self.rows)
        
        # Generate DataItem Masks
        self.make_distinct()
        
        # Randomise Values
        self.randomise()
        
        # Separate Ground Truth and Data
        self.truth = self.data.loc[['Truth']]
        self.data = self.data.drop('Truth')

        if verbose == 1:
            print(f"Dataset Generated: {timeit.default_timer() - start_time:.4f} s")

        #self.data['F'] = self.data[self.data >= MASK_F].count(axis=1)
        #self.data['T'] = self.data[self.data == MASK_T].count(axis=1)
        #self.data['NA'] = self.data[self.data == MASK_NA].count(axis=1)

    def get_claims(self):
        res = []
        for col in self._verbose_iter(self.data.columns, "Claims"):
            for idx, val in enumerate(self.data[col]):
                if val == 0: continue
                res.append([idx, col, val])
        claims = pd.DataFrame(res, columns=['Source', 'DataItem', 'Value'])
        self.len_claims = len(claims)
        return claims


    def randomise(self):
        for idx in self._verbose_iter(range(self.n_dataitems), "Randomise"):
            temp_map = dict(enumerate(np.random.randint(10_000_000, 100_000_000, self.n_distinct + 1)))
            temp_map[0] = 0
            self.data[idx].replace(temp_map, inplace=True)
        

    def make_distinct(self):
        distinct_list = self.distinct_dist(self.n_dataitems) * self.n_distinct
        for idx, distinct in enumerate(self._verbose_iter(distinct_list, "DataItems")):
            
            if int(distinct) <= 1: continue
            vals = []
            probas = self.spread_dist(int(distinct) - 1)
            dist = np.random.multinomial((self.data[idx] == MASK_F).sum(), probas / probas.sum())
            
            for i, v in enumerate(dist):
                vals.extend([i] * v)
            vals = np.array(vals)
            np.random.shuffle(vals)
            
            self.data[idx][self.data[idx] == MASK_F] += vals
            

    def _add(self, index, values):
        self.rows.append(pd.DataFrame({index: values}, dtype=int).transpose())

    def add_truth(self):
        self._add('Truth', np.zeros(self.n_dataitems) + MASK_T)

    def add_claims(self):
        for source_id in self._verbose_iter(range(self.n_sources), "Sources"):
            
            n_zeroes = int(self.n_dataitems * (1 - self.coverage_dist(1)))    # Objects without value       (represented by 0)
            n_values = self.n_dataitems - n_zeroes                            # Split into truths and falsehoods
            n_false = int(n_values * (1 - self.truth_dist(1)))                # Objects with the true value (represented by 1)
            n_truth = n_values - n_false                                      # Objects with false values   (represented by 2)
                        
            zeroes = np.zeros(n_zeroes) + MASK_NA
            truths = np.zeros(n_truth) + MASK_T
            falses = np.zeros(n_false) + MASK_F
            
            claim = np.concatenate((zeroes, truths, falses))
            np.random.shuffle(claim)
            self._add(source_id, claim)

    def compare(self, results):
        np_results = results.to_numpy()
        np_truth = np.tile(self.truth.to_numpy(), (np_results.shape[0], 1))

        assert np_truth.shape == np_results.shape, (np_truth.shape, np_results.shape)
        sums = np.equal(np_truth, np_results).sum()
        assert sums <= self.n_dataitems
        return float(sums / self.n_dataitems), int(sums)
    
    def _verbose_iter(self, iter_list, title):
        if self.verbose > 1:
            return tqdm(iter_list, desc=f'{title:<12}')
        else:
            return iter_list
