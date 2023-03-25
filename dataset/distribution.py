import numpy as np
import scipy as sc


class Constant():
    def __init__(self, value):
        assert value >= 0 and value <= 1
        self.value = value

    def _get_info(self):
        return {'name': 'Constant', 'value': self.value}
    
    def rvs(self):
        def _const(x):
            return np.zeros(x) + self.value
        return _const

    def pdf(self):
        raise NotImplementedError

class Uniform():
    def __init__(self, low, high):
        assert low >= 0 and high <= 1
        self.low, self.high = low, high
        self.dist = sc.stats.uniform(loc=low, scale=high-low)

    def _get_info(self):
        return {'name': 'Uniform', 'low': self.low, 'high': self.high}

    def rvs(self):
        return lambda x: self.dist.rvs(size=x)
    
    def pdf(self):
        return self.dist.pdf

class TruncExponential():
    def __init__(self, lmbda, flipped=False):
        self.lmbda, self.flipped = lmbda, flipped
        self.dist = sc.stats.truncexpon(b=lmbda, loc=0, scale=1/lmbda)

    def _get_info(self):
        return {'name': 'TruncExponential', 'lmbda': self.lmbda, 'flipped': self.flipped}

    def rvs(self):
        if self.flipped:
            return lambda x: 1 - self.dist.rvs(size=x)
        else:
            return lambda x: self.dist.rvs(size=x)
    
    def pdf(self):
        if self.flipped:
            return lambda x: self.dist.pdf(1 - x)
        else:
            return self.dist.pdf

class TruncNormal():
    def __init__(self, mean, stdv):
        self.mean, self.stdv = mean, stdv
        a, b = (0 - mean) / stdv, (1 - mean) / stdv
        self.dist = sc.stats.truncnorm(a=a, b=b, loc=mean, scale=stdv)

    def _get_info(self):
        return {'name': 'TruncNormal', 'mean': self.mean, 'stdv': self.stdv}

    def rvs(self):
        return lambda x: self.dist.rvs(size=x)
    
    def pdf(self):
        return self.dist.pdf

class TruncPareto():
    def __init__(self, alpha=None, flipped=False):
        if alpha is None:
            alpha = np.log10(5)/np.log10(4)
        self.alpha, self.flipped = alpha, flipped
        self.dist = sc.stats.truncpareto(b=alpha, c=2.0, loc=-1)

    def _get_info(self):
        return {'name': 'TruncPareto', 'alpha': self.alpha, 'flipped': self.flipped}

    def rvs(self):
        if self.flipped:
            return lambda x: 1 - self.dist.rvs(size=x)
        else:
            return lambda x: self.dist.rvs(size=x)
    
    def pdf(self):
        if self.flipped:
            return lambda x: self.dist.pdf(1 - x)
        else:
            return self.dist.pdf



#def Custom(cdf_inversion):
#    def _func(x):
#        vals = np.random.default_rng().uniform(size=x)
#        return cdf_inversion(vals)
#    return _func

#def CustomExponential(scale):
#    def _func(x):
#        vals = np.random.default_rng().uniform(size=x)
#        return -np.log(1 - (1 - np.exp(-(1/scale))) * vals) * scale
#    return _func