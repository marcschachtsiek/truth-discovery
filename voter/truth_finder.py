import numpy as np
import pandas as pd

from ._common import _map_dict, _error, _get_highest_confidence, _get_top, _get_top_k

class TruthFinder():
    
    def __init__(self, base_trust, tolerance=0.001, dampening_factor=0.1) -> None:
        self.base_trust = base_trust
        self.tolerance = tolerance
        self.dampening_factor = dampening_factor
    
    def _get_info(self):
        return {'name': 'TruthFinder', 
                'base_trust': self.base_trust, 
                'tolerance': self.tolerance, 
                'dampening_factor': self.dampening_factor}
    
    def run(self, claims, max_iter=100, top=1):
        trustworthiness = {s: self.base_trust for s in claims['Source'].unique()}
        confidence = {}
        
        for i in range(max_iter):
            tw_old = list(trustworthiness.values())
                        
            for d in claims['DataItem'].unique():     
                vs_pairs = claims[(claims['DataItem'] == d)][['Value', 'Source']]
                for v in vs_pairs['Value'].unique():
                    
                    # Update Confidence
                    sources = vs_pairs.loc[(claims['Value'] == v)]['Source']
                    v_conf = np.log(1 - _map_dict(sources, trustworthiness)).sum()
                                        
                    # Adjust Confidence (SKIPPED)
                    
                    # Dampen Confidence
                    confidence[v] = 1 / (1 + np.exp(self.dampening_factor * v_conf))
                            
            for s in claims['Source'].unique():
                
                # Update Trustworthiness
                values = claims[claims['Source'] == s]['Value'].unique()
                trustworthiness[s] = _map_dict(values, confidence).sum() / len(values)

            # Check Convergence
            if self.tolerance > _error(list(trustworthiness.values()), tw_old):
                break
        
            if i >= max_iter - 1:
                print(f"TruthFinder reached maximum iteration [{max_iter}]")

        return _get_top_k(claims, confidence, top, "TruthFinder")
