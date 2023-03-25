import pandas as pd

class Optimal():
    
    def run(self, claims, truth):
        res = []
        for d in claims['DataItem'].unique():
            t = truth[d][0]
            res.append(t if t in claims[claims['DataItem'] == d]['Value'].to_numpy() else 0)
        return pd.DataFrame({"Optimal": res}).transpose()