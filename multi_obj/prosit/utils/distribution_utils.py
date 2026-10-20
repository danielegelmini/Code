import warnings

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# "distribution" of a tree leaf that draws among the values observed in the leaf (kept as its "sampled"
# values) instead of from a fitted family: see time_discovery.build_models_ex
EMPIRICAL = "empirical"



def fit_distribution(data: any, dist_name: str) -> tuple:

    if dist_name == 'fixed':
        params = (np.mean(data),)
        generated_values = np.full(len(data), params[0])
        wass_distance = stats.wasserstein_distance(data, generated_values)

        return params, wass_distance, 'fixed'

    dist = getattr(stats, dist_name)

    # As in Vinci's prosit 1.0.4: the goodness of fit is the Wasserstein distance between the sorted data
    # and the quantiles of the fitted distribution (clipped to the data range). It used to be computed on a
    # random sample of the fitted distribution, so the chosen family changed from run to run and was
    # distorted by the out-of-range draws that sampling_from_dist replaced with the mean. The fit of
    # heavy-tailed families emits harmless RuntimeWarnings (overflow), silenced here.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        params = dist.fit(data)
        n = len(data)
        probs = (np.arange(1, n + 1) - 0.5) / n
        theoretical_quantiles = np.clip(dist.ppf(probs, *params), min(data), max(data))
        wass_distance = stats.wasserstein_distance(np.sort(data), theoretical_quantiles)
    if not np.isfinite(wass_distance):  # degenerate fit (NaN quantiles): never the best one
        wass_distance = np.inf

    return params, wass_distance, dist


def return_best_distribution(data: any, dist_search: list = ['fixed', 'norm', 'expon', 'lognorm', 'uniform']) -> tuple:

    dict_fitting_distributions = dict()
    dict_fitting_dist_params = dict()
    dict_wass = dict()

    # empty data first, then a single repeated value; data can be a list, a numpy array or a Series
    if len(data) == 0:
        return 'fixed', (0,)
    if len(set(data)) == 1:
        return 'fixed', (float(np.asarray(data).flat[0]),)

    for dist_name in dist_search:

        params, goodness_of_fit, dist = fit_distribution(data, dist_name)
        dict_fitting_distributions[dist_name] = dist
        dict_wass[dist_name] = goodness_of_fit
        dict_fitting_dist_params[dist_name] = params

    best_fit_dist_name = min(dict_wass, key=dict_wass.get)
    best_fit_dist = dict_fitting_distributions[best_fit_dist_name]
    best_fit_dist_params =  dict_fitting_dist_params[best_fit_dist_name]

    return best_fit_dist, best_fit_dist_params


def sampling_from_dist(
        dist: any, params: tuple, 
        min_value: float, max_value: float, mean_value: float, 
        n_sample: int = 1000
    ) -> np.array:

    if dist == 'fixed':
        return np.array([mean_value] * n_sample, dtype=float)

    # Draws outside [min_value, max_value] are clipped to the nearest bound, as in Vinci's prosit 1.0.4.
    # They used to be replaced with mean_value, which cut the tails of the distribution and put a spike
    # at the mean (above the observed maximum when mean_value is the mean of a wider sample).
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        l = dist.rvs(*params, n_sample)
    l = np.nan_to_num(l, nan=0.0, posinf=max_value, neginf=min_value)

    return np.clip(l, min_value, max_value).astype(float)



def plot_distribution(data: any, params: tuple, dist: any):

    plt.hist(data, bins=30, density=True, alpha=0.6, color='skyblue', label="Data Histogram")
    
    x = np.linspace(min(data), max(data), 1000)
    y = dist.pdf(x, *params)
    
    plt.plot(x, y, 'r-', lw=2, label=f'Fitted {dist.name} Distribution')
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()
    plt.title(f"Fitted {dist.name.capitalize()} Distribution")
    plt.show()


def remove_outliers(data: list, m: float = 20.0) -> list:

    data = np.asarray(data)
    if data.size == 0:  # np.median of an empty array only warns and returns NaN
        return []
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / (mdev if mdev else 1.0)
    
    return data[s < m].tolist()