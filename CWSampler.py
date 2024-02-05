import math
from scipy import optimize
from scipy.stats import norm
from scipy.special import comb, logsumexp
from multiprocessing import Pool
from collections import Counter
import numpy as np
import warnings


### Reference by the authors Mukherjee, Son, & Bhattacharya, FLUCTUATIONS OF THE MAGNETIZATION IN THE p-SPIN CURIE-WEISS MODEL ###
### https://link.springer.com/article/10.1007/s00220-021-04182-z ####


# Binary entropy function: See below equation (2.1)
def I(x):
    if abs(x) < 1:
        return 1 / 2 * ((1 + x) * math.log(1 + x) + (1 - x) * math.log(1 - x))
    else:
        return math.log(2)

# H function: Equation (2.1)
def H(beta, h, p, x):
    return beta * (x ** p) + h * x - I(x)

# The first derivative of the H function in (2.1)
def H1(beta, h, p, x):
    return beta * p * x ** (p - 1) + h - math.atanh(x)

# The second derivative of the H function
def H2(beta, p, x):
    return beta * p*(p-1) * x ** (p-2) - 1/(1 - x**2)


# Define the maximizers of the function H
"""
s controls the step size in searching for the maximizer. s should be larger if the parameters are around the special points,
where we expect the two maximizers to be close to each other.
"""
def mstar(beta, h, p, s=10):
    def g(x):
        return H1(beta, h, p, x)
    # If it exists, find the first critical point of H on (0,1)
    limit = 1 - 1e-10
    i = 1
    m1 = 0
    while i < s:
        try:
            m1 = optimize.brentq(g, 1e-10, i / s)
            i = s
        except ValueError as e:
            i = i + 1
    # If it exists, find the last critical point on (0, 1): there cannot be more than 2 critical points on (0,1) by Lemma C.1
    i = 1
    m2 = 0
    while i < s:
        try:
            m2 = optimize.brentq(g, limit - i / s, limit)
            i = s
        except ValueError as e:
            i = i + 1
    # If it exists, find the first critical point on (-1,0)
    m3 = 0
    i = 1
    while i < 100:
        try:
            m3 = optimize.brentq(g, -i / s, -1e-10)
            i = s
        except ValueError as e:
            i = i + 1
    # If it exists, find the last critical point on (-1, 0)
    i = 1
    m4 = 0
    while i < s:
        try:
            m4 = optimize.brentq(g, -i / s, -limit)
            i = s
        except ValueError as e:
            i = i + 1
    # Find which critical point maximizes H on (-1,1) and return it
    m = [0, m1, m2, m3, m4]
    l = [H(beta, h, p, x) for x in m]
    maximizer_index = np.argmax(l)
    maximizer = m[maximizer_index]
    other_maximizer_indices = []
    if 1e-3< abs(H2(beta, p, maximizer)) < 1e-1:
        warnings.warn("The second derivative evaluated at the maximizer is close to 0 suggesting the parameters are near the special point. If the parameters are at the special point, we only have one maximizer. If they are near the special point, there can only be at most 2 maximizers. Make sure to increase the function argument 's' to avoid missing any maximizers")
    
    for i in range(5):
        if i != maximizer_index and abs(l[maximizer_index]- l[i]) < 1e-5:
            other_maximizer_indices.append(i)
    if len(other_maximizer_indices) == 0:
        if maximizer != 0:
            return np.array([maximizer])
        else:
            warnings.warn("mstar is 0 but could be close to 1 or -1.")
            return [np.array([maximizer])]
    else:
        list_of_maximizers = [maximizer] + [m[i] for i in other_maximizer_indices]
        final_maximizers = sorted(list(set(np.round(list_of_maximizers, 8))))
        return np.array(final_maximizers)



class CWModel:
    def __init__(self, N:int, beta:float, h:float, p:int):
        assert p >= 2
        assert beta >= 0
        assert N >= 1
        self.N, self.beta, self.h, self.p = N, beta, h, p
        self.support = np.arange(-N, N + 1, 2) / N
        self.maximizers = mstar(beta, h, p)
        m = self.maximizers
        self.sd = np.sqrt(-H2(beta, p, m))
        if len(m) > 1:
            self.weights = np.sqrt((m**2-1) * H2(beta, p, m))
            self.weights /= self.weights.sum()
            self.point = "critical"
        else:
            self.weights = 1
            if abs(H2(beta, p, m)) < 1e-2:
                self.point = "special"
            else:
                self.point = "regular"
        
    def pmf(self, beta=None, h=None):
        # if not initialized, use the class arguments
        if not beta:
            beta = self.beta
        if not h:
            h = self.h
        
        N, p = self.N, self.p
        # combinatorial term counting the number of 1s (pg 691)
        def coeff(x):
            return comb(N, (1 + x) / 2, exact = True)

        mag_values = self.support
        # Calculate the summands over only the first N/2 sequences and
        # Use symmetry of the binomial coefficient to save time
        log_pmf = np.zeros(N + 1)
        for i in range(int(N/2)):
            m = mag_values[i]
            coef = coeff(m)
            log_pmf[i] = math.log(coef) + N * H(beta, h, p, m)
            log_pmf[(N - 1 - i)] = math.log(coef) + N * H(beta, h, p, -m)
        # Only include the middle term if N is odd
        if N % 2 == 0:
            m = mag_values[int(N/2) + 1]
            coef = coeff(m)
            log_pmf[int(N/2) + 1] = math.log(coef) + N * H(beta, h, p, m)
        
        return np.exp(log_pmf - logsumexp(log_pmf))
    
    def generate_rvs(self, num_samples, seed=123):
        np.random.seed(seed)
        mag_values, pmf = self.support, self.pmf(self.beta, self.h)
        return np.random.choice(mag_values, size = num_samples, p = pmf)
    
    # MLE of beta when h is known for a random sample x from CW beta, h, p distribution
    def MLE_beta(self, x):
        N, beta, h, p = self.N, self.beta, self.h, self.p
        mag_values = self.support
        
        def f(beta_hat):
            return np.sum(self.pmf(beta_hat, h) * mag_values ** p) - x ** p
        
        i = -0.35
        while True and i < 0:
            try:
                return optimize.brentq(f, (1-N**i)*beta, (1+N**i)*beta)
            except ValueError as e:
                i += 0.1
        return np.inf
            
    # MLE of h when beta is known for a random sample x from CW beta, h, p distribution
    def MLE_h(self, x):
        N, beta, h, p = self.N, self.beta, self.h, self.p
        mag_values = self.support
        
        def f(h_hat):
            return np.sum(self.pmf(beta, h_hat) * mag_values) - x
        
        i = -0.35
        while True and i < 0:
            try:
                return optimize.brentq(f, h-N**i, h+N**i)
            except ValueError as e:
                i += 0.1
    
    # parallelize MLE_beta
    def MLE_beta_parallel(self, samples:list):
        N, beta, h, p = self.N, self.beta, self.h, self.p
        m = self.maximizers[-1]
        
        # by symmetry, MLE of beta is the same for p even and h = 0
        if p % 2 == 0 and h == 0:
            x_pos = abs(samples)
            samples_pos = Counter(x_pos)

        # dictionary of samples to avoid recalculating same MLE
        samples = Counter(samples)
         
        # Parallelize MLE_beta
        if p % 2 == 0 and h == 0:
            with Pool() as pool:
                mles_betas = pool.map_async(self.MLE_beta, list(samples_pos.keys())).get()
            return sum([[mles_betas[i]]*samples[x] for i, x in enumerate(samples_pos.keys())], [])
        else:
            with Pool() as pool:
                mles_betas = pool.map_async(self.MLE_beta, list(samples.keys())).get()
            return sum([[mles_betas[i]]*samples[x] for i, x in enumerate(samples.keys())], [])
    
    # parallelize MLE_h
    def MLE_h_parallel(self, samples:list):
        N, beta, h, p = self.N, self.beta, self.h, self.p 
        m = self.maximizers[-1]

        # dictionary of samples to avoid recalculating same MLE
        samples = Counter(samples)
        
        # Parallelize MLE of h
        with Pool() as pool:
            mles_h = pool.map_async(self.MLE_h, list(samples.keys())).get()
        
        return sum([[mles_h[i]]*samples[x] for i, x in enumerate(samples.keys())], [])
    
    # asymptotic theoretical quantiles
    def asymptotic_quantiles_h(self, q):
        assert 0 <= q <= 1
        sd = self.sd
        if self.point == 'regular':
            norm.ppf(q, scale=sd)
        elif self.point == 'special':
            if q == 0.5:
                return 0
            def f(t):
                return G1(t, beta, h, p, m) - q
            if q < 0.5:
                return optimize.brentq(f, -50, 0)
            else:
                return optimize.brentq(f, 0, 50)
        else:
            w = self.weights
            if p % 2 == 1:
                if q < w[0]/4:
                    return norm.ppf(2*q/w[0], scale = sd[0])
                elif w[0]/4 <= q <= w[0]/4 + 1/2:
                    return 0
                else:
                    return norm.ppf(2*q-1, scale=sd[1])
            if h == 0 and beta > beta_t:
                if q < 0.25:
                    return norm.ppf(2*q, scale = sd)
                elif 0.25 <= q <= 0.75:
                    return 0
                else:
                    return norm.ppf(2*q-1, scale=sd)
            if h == 0 and beta == beta_t:
                if q < w[0]/2:
                    return norm.ppf(q/w[0], scale = sd[0])
                elif w[0]/2 <= q <= 1-w[0]/2:
                    return 0
                else:
                    return norm.ppf((q-1), scale=sd)
            


