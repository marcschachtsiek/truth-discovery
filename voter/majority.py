import pandas as pd

class Majority():

    def _get_info(self):
        return {'name': 'Majority'}

    def run(self, claims):
        res = []
        for d in claims['DataItem'].unique():
            res.append(claims[claims['DataItem'] == d]['Value'].mode()[0])
        return pd.DataFrame({"Majority": res}).transpose()