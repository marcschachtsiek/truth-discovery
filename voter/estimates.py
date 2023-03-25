import numpy as np
import pandas as pd

from ._common import _map_dict, _error, _dict_norm, _get_top_k

class TwoEstimates():
    
    def __init__(self, base_trust, tolerance=0.001, lmbda=0.5) -> None:
        self.base_trust = base_trust
        self.tolerance = tolerance
        self.lmbda = lmbda

    def _get_info(self):
        return {'name': 'TwoEstimates', 
                'base_trust': self.base_trust, 
                'tolerance': self.tolerance,
                'lmbda': self.lmbda}
        
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
                    pos = (1 - _map_dict(sources, trustworthiness)).sum()
                    
                    sources = vs_pairs.loc[(claims['Value'] != v)]['Source']
                    neg = _map_dict(sources, trustworthiness).sum()
                    
                    temp = len(claims[claims['DataItem'] == d]['Source'])
                    confidence[v] = (pos + neg) / temp
            
            # Normalise Confidence
            confidence = _dict_norm(confidence, self.lmbda)

            for s in claims['Source'].unique():
                
                # Update Trustworthiness
                values = claims[claims['Source'] == s]['Value'].unique()
                pos = (1 - _map_dict(values, confidence)).sum()
                
                values = claims[claims['Source'] != s]['Value'].unique()
                neg = _map_dict(values, confidence).sum()
                
                temp = len(claims[claims['Source'] == s]['Value'].unique())
                trustworthiness[s] = (pos + neg) / temp

            # Normalise Trustworthiness
            trustworthiness = _dict_norm(trustworthiness, self.lmbda)

            # Check Convergence
            if self.tolerance > _error(list(trustworthiness.values()), tw_old):
                break
            
            if i >= max_iter - 1:
                print(f"TwoEstimates reached maximum iteration [{max_iter}]")

        return _get_top_k(claims, confidence, top, "TwoEstimates")


class ThreeEstimates():
    
    def __init__(self, base_trust, tolerance=0.001, lmbda=0.5, base_error_factor=0.1) -> None:
        self.base_trust = base_trust
        self.tolerance = tolerance  
        self.lmbda = lmbda
        self.base_error_factor = base_error_factor

    def _get_info(self):
        return {'name': 'ThreeEstimates', 
                'base_trust': self.base_trust, 
                'tolerance': self.tolerance,
                'lmbda': self.lmbda,
                'base_error_factor': self.base_error_factor}

    def run(self, claims, max_iter=100, top=1):
        trustworthiness = {s: self.base_trust for s in claims['Source'].unique()}
        error_factor = {v: self.base_error_factor for v in claims['Value'].unique()}
        
        confidence = {}
        
        for i in range(max_iter):
            tw_old = list(trustworthiness.values())

            # Update Confidence 
            for d in claims['DataItem'].unique():     
                vs_pairs = claims[(claims['DataItem'] == d)][['Value', 'Source']]
                for v in vs_pairs['Value'].unique():
                    
                    sources = vs_pairs.loc[(claims['Value'] == v)]['Source']
                    pos = (1 - _map_dict(sources, trustworthiness) * error_factor[v]).sum()
                    
                    sources = vs_pairs.loc[(claims['Value'] != v)]['Source']
                    neg = (_map_dict(sources, trustworthiness) * error_factor[v]).sum()
                    
                    temp = len(claims[claims['DataItem'] == d]['Source'])
                    confidence[v] = (pos + neg) / temp
            
            # Normalise Confidence
            confidence = _dict_norm(confidence, self.lmbda)

            # Update Error Factor
            for d in claims['DataItem'].unique():     
                vs_pairs = claims[(claims['DataItem'] == d)][['Value', 'Source']]
                
                norm = np.count_nonzero(_map_dict(vs_pairs['Source'], trustworthiness))
                
                for v in vs_pairs['Value'].unique():
                    
                    sources = vs_pairs.loc[(claims['Value'] == v)]['Source']
                    ts = _map_dict(sources, trustworthiness)
                    pos = ((1 - confidence[v]) / ts[ts != 0]).sum()
                    
                    sources = vs_pairs.loc[(claims['Value'] != v)]['Source']
                    ts = _map_dict(sources, trustworthiness)
                    neg = (confidence[v] / ts[ts != 0]).sum()
                    
                    error_factor[v] = (pos + neg) / norm
            
            # Normalise Error Factor
            error_factor = _dict_norm(error_factor, self.lmbda)

            # Update Confidence
            for s in claims['Source'].unique():

                values = claims[claims['Source'] == s]['Value'].unique()
                ev = _map_dict(values, error_factor)
                pos = ((1 - _map_dict(values, confidence)[ev != 0]) / ev[ev != 0]).sum()
                
                values = claims[claims['Source'] != s]['Value'].unique()
                ev = _map_dict(values, error_factor)
                neg = (_map_dict(values, confidence)[ev != 0] / ev[ev != 0]).sum() * len(claims[claims['Source'] == s]['DataItem'].unique())
                
                temp = claims[claims['Source'] == s]['Value'].unique()
                trustworthiness[s] = (pos + neg) / np.count_nonzero(_map_dict(temp, error_factor))

            # Normalise Trustworthiness
            trustworthiness = _dict_norm(trustworthiness, self.lmbda)

            # Check Convergence
            if self.tolerance > _error(list(trustworthiness.values()), tw_old):
                break

            if i >= max_iter - 1:
                print(f"ThreeEstimates reached maximum iteration [{max_iter}]")

        return _get_top_k(claims, confidence, top, "ThreeEstimates")
