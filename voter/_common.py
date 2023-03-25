import numpy as np
import pandas as pd

def _map_dict(values, dictionary):
    if values.shape[0] == 0: return np.array([0])
    return np.vectorize(dictionary.__getitem__)(values)

def _error(tw, tw_old):
    return 1 - np.dot(tw, tw_old / (np.linalg.norm(tw) * np.linalg.norm(tw_old)))

def _get_highest_confidence(claims, confidence):
    res = []
    for d in claims['DataItem'].unique():
        values = claims[claims['DataItem'] == d]['Value'].unique()
        res.append(values[_map_dict(values, confidence).argmax()])
    return res

def _dict_norm(dictionary, lmbda):
    vals = dictionary.values()
    min_val, max_val = min(vals), max(vals)
        
    for key in dictionary:
        val = dictionary[key]
        x1 = _normalise(val, min_val, max_val)
        x2 = round(val)
        dictionary[key] = lmbda * x1 + (1 - lmbda) * x2
    
    return dictionary

def _normalise(val, val_min, val_max):
    if val_min == val_max: return val
    return val - val_min / (val_max - val_min)

def _get_top(claims, confidence):
    return pd.DataFrame({0: _get_highest_confidence(claims, confidence)}).transpose()

def _get_top_k(claims, confidence, k, prefix):
    if k == 1:
        df = _get_top(claims, confidence)
    else:
        cols = []
        for i, d in enumerate(claims['DataItem'].unique()):
            values = claims[claims['DataItem'] == d]['Value'].unique()
            confidences = _map_dict(values, confidence)
            top_k = np.argpartition(confidences, -k)[-k:]
            cols.append(pd.DataFrame({i: values[top_k[np.argsort(confidences[top_k])][::-1]]}))

        df = pd.concat(cols, axis=1)
    df.index = [f"{prefix}_{x+1}" for x in range(k)]
    return df