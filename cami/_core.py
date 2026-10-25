'''Shared building blocks for the CaMI estimators.

Everything here is private API: input cleaning, symbolic encoding helpers,
symbolic-word indexing and the plug-in estimators computed from
probability tables.
'''
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger('cami')

_LOG_FUNCTIONS = {
    'bit': np.log2, 'bits': np.log2,
    'nat': np.log, 'nats': np.log,
    'ban': np.log10, 'bans': np.log10,
}


def log_function(units):
    '''Returns the logarithm matching the information units.'''
    try:
        return _LOG_FUNCTIONS[units]
    except (KeyError, TypeError):
        raise ValueError('Units must be bits or nat or ban. See help on function.') from None


def as_int(value, name):
    '''Accepts Python and numpy integers, rejects anything else.'''
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError('Error: %s must be an integer. See help on function' % name)
    return int(value)


def to_float_array(x):
    '''Copies a time-series into a 1-D float array (non-numeric values become NaN).'''
    if isinstance(x, (pd.Series, pd.DataFrame, pd.Index)):
        x = x.to_numpy()
    x = np.asarray(x)
    if x.dtype.kind in 'biuf':  # already numeric: a single float copy, no pandas round-trip
        return np.array(x, dtype=float).ravel()
    return np.array(pd.to_numeric(x.ravel(), errors='coerce'), dtype=float)


def _interpolate_nans(data):
    bad = np.isnan(data)
    if bad.any():
        good = ~bad
        data[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(good), data[good])
    return data


def _trim_and_interpolate(valid, *series):
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        raise ValueError('Error: time-series contains no numeric values')
    return tuple(_interpolate_nans(s[idx[0]:idx[-1] + 1]) for s in series)


def clean_series(x):
    '''Trims NaNs at the edges and linearly interpolates interior NaNs (on a copy).'''
    x = to_float_array(x)
    return _trim_and_interpolate(~np.isnan(x), x)[0]


def clean_pair(x, y):
    '''Like clean_series, trimming both series where either has edge NaNs.'''
    x, y = to_float_array(x), to_float_array(y)
    if len(x) != len(y):
        raise ValueError('Error: Length of x and y must be equal!')
    return _trim_and_interpolate(~np.isnan(x) & ~np.isnan(y), x, y)


def apply_delay(x, y, delay):
    '''Pairs x[n] with y[n+delay].'''
    delay = as_int(delay, 'delay')
    if delay > 0:
        return x[:-delay], y[delay:]
    if delay < 0:
        return x[-delay:], y[:delay]
    return x, y


def as_divisions(divs):
    '''Normalizes user-given partition divisions to a 1-D float array (or None).'''
    if divs is None:
        return None
    if np.isscalar(divs):
        divs = [divs]
    if isinstance(divs, (pd.Series, pd.Index)):
        divs = divs.to_numpy()
    if not isinstance(divs, (list, tuple, np.ndarray)):
        raise ValueError('partition data type not supported. x_divs and y_divs must both be None '
                         'or both be a float, list or tuple of same length')
    return np.asarray(divs, dtype=float).ravel()


def resolve_pair_divisions(x_divs, y_divs):
    x_divs, y_divs = as_divisions(x_divs), as_divisions(y_divs)
    if (x_divs is None) != (y_divs is None):
        raise ValueError('Inconsistent use of partition divisions: x_divs and y_divs must both be None '
                         'or both be a float, list or tuple of same length')
    if x_divs is not None and len(x_divs) != len(y_divs):
        raise ValueError('x_divs and y_divs must have same length')
    return x_divs, y_divs


# symbolic type -> (order of the difference encoded, how divisions are placed)
SYMBOLIC_TYPES = {
    'equal-divs': (0, 'width'), 'equal-points': (0, 'points'),
    'equal-growth': (1, 'width'), 'equal-growth-points': (1, 'points'),
    'equal-concavity': (2, 'width'), 'equal-concavity-points': (2, 'points'),
}


def encode(data, divisions):
    '''Symbol i is given to values in [divisions[i-1], divisions[i]).'''
    divisions = np.asarray(divisions, dtype=float)
    if np.any(np.diff(divisions) < 0):
        raise ValueError('Error: partition divisions must be in increasing order')
    return np.searchsorted(divisions, data, side='right')


def transform(data, symbolic_type):
    '''Returns the data to encode (the series or its 1st/2nd difference) and the division placement.'''
    try:
        order, placement = SYMBOLIC_TYPES[symbolic_type]
    except (KeyError, TypeError):
        raise ValueError('Error: Unacceptable argument of symbolic type. See help on function.') from None
    return (np.diff(data, order) if order else np.asarray(data)), placement


def place_divisions(data, placement, n_symbols):
    if placement == 'width':
        lo, hi = data.min(), data.max()
        return [lo + i * (hi - lo) / n_symbols for i in range(1, n_symbols)]
    ordered = np.sort(data)
    return [ordered[int(i * len(data) / n_symbols)] for i in range(1, n_symbols)]


def symbolize(data, symbolic_type, n_symbols):
    '''Encodes data (or its 1st/2nd difference) with automatically placed divisions.'''
    data, placement = transform(data, symbolic_type)
    return encode(data, place_divisions(data, placement, n_symbols))


def as_trials(x):
    '''Splits input into a list of 1-D trials: columns of a 2-D array, or items of a list of series.'''
    if isinstance(x, pd.DataFrame):
        x = x.to_numpy()
    if isinstance(x, (list, tuple)) and len(x) > 0 and np.ndim(x[0]) == 1:
        return [np.asarray(t) for t in x]
    x = np.asarray(x)
    if x.ndim == 1:
        return [x]
    if x.ndim != 2:
        raise ValueError('Error: trials must be given as a 2-D array (columns are trials) or a list of series')
    return list(x.T)


def first_zero_autocorrelation(data):
    '''Smallest lag at which the autocorrelation crosses zero (1 if it never does).'''
    old_corr_val = 1
    for i in range(1, len(data)):
        new_corr_val = np.corrcoef(data[:-i], data[i:])[0, 1]
        if old_corr_val > 0 and new_corr_val <= 0:
            return i
        old_corr_val = new_corr_val
    return 1


def resolve_tau(tau, *series):
    if tau is None:
        tau = max(first_zero_autocorrelation(s) for s in series)
        logger.info('Selected tau=%d by the method of first zero of auto-correlation', tau)
    return as_int(tau, 'tau')


def parse_symbolic_length(symbolic_length, max_parts=3):
    '''Returns (lx, lyp, lyf) from an int or a list/tuple of up to max_parts ints.'''
    message = 'Error: Symbolic length must be int or list/tuple with 2%s elements. See help on function' % (
        ' or 3' if max_parts == 3 else '')
    if isinstance(symbolic_length, (int, np.integer)) and not isinstance(symbolic_length, (bool, np.bool_)):
        lengths = (symbolic_length,) * 3
    elif isinstance(symbolic_length, (list, tuple, np.ndarray)) and 2 <= len(symbolic_length) <= max_parts:
        lengths = tuple(symbolic_length) + (symbolic_length[1],) * (3 - len(symbolic_length))
    else:
        raise TypeError(message)
    lengths = tuple(as_int(l, 'symbolic_length') for l in lengths)
    if min(lengths) < 1:
        raise ValueError('Error: symbolic lengths must be positive')
    return lengths


def prepare_pair(x, y, x_divs, y_divs, symbolic_type, n_symbols, tau, delay):
    '''Cleans, delays and encodes a pair of series.

    Returns x, y, Sx, Sy, n_symbols, tau.
    '''
    from .symbolic_encoding import symbolic_encoding
    x, y = clean_pair(x, y)
    x, y = apply_delay(x, y, delay)
    x_divs, y_divs = resolve_pair_divisions(x_divs, y_divs)
    n_symbols = as_int(n_symbols, 'n_symbols')
    if x_divs is None:
        Sx, Sy = symbolic_encoding(x, y, symbolic_type=symbolic_type, n_symbols=n_symbols)
    else:
        Sx, Sy = symbolic_encoding(x, y, x_divs=list(x_divs), y_divs=list(y_divs))
        n_symbols = len(x_divs) + 1
    tau = resolve_tau(tau, x, y)
    return x, y, Sx, Sy, n_symbols, tau


def clean_multivariate(x):
    '''Columns are variables. Edge rows with any NaN are trimmed, interior NaNs interpolated.'''
    if isinstance(x, pd.DataFrame):
        x = x.to_numpy()
    x = np.asarray(x)
    if x.dtype.kind not in 'biuf':
        x = np.asarray(x, dtype=object)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError('Error: x must be 2-D, with rows as time and columns as variables')
    columns = [to_float_array(col) for col in x.T]
    valid = np.logical_and.reduce([~np.isnan(col) for col in columns])
    return np.column_stack(_trim_and_interpolate(valid, *columns))


_TAU_AGGREGATES = {None: np.mean, 'average': np.mean, 'mean': np.mean,
                   'max': np.max, 'maximum': np.max, 'median': np.median, 'med': np.median}


def resolve_multivariate_tau(tau, x):
    if isinstance(tau, str) or tau is None:
        try:
            aggregate = _TAU_AGGREGATES[tau]
        except KeyError:
            raise ValueError("Error: tau must be an integer, None, 'mean', 'max' or 'median'") from None
        tau = max(1, int(aggregate([first_zero_autocorrelation(col) for col in x.T])))
        logger.info('Selected tau=%d by the method of first zero of auto-correlation (%s over variables)',
                    tau, aggregate.__name__)
    return as_int(tau, 'tau')


def multivariate_words(x, x_divs, symbolic_type, n_symbols, symbolic_length, tau):
    '''Symbolic words of every variable, one row per time where all words exist.'''
    x = clean_multivariate(x)
    n_vars = x.shape[1]
    if x_divs is None:
        n_symbols = as_int(n_symbols, 'n_symbols')
        symbols = [symbolize(col, symbolic_type, n_symbols) for col in x.T]
    else:
        if not isinstance(x_divs, (list, tuple, np.ndarray)):
            raise TypeError('x_divs must be 2D list, 2D tuple or None. Example: x_divs=[[0.3,0.4],[0.1,0.5],[0.3,0.2]]')
        if len(x_divs) != n_vars:
            raise ValueError('x_divs must have one list of divisions per variable')
        divisions = [as_divisions(d) for d in x_divs]
        if len({len(d) for d in divisions}) != 1:
            raise ValueError('length of every x_divs column must be equal (same number of divisions for each variable)')
        symbols = [encode(col, d) for col, d in zip(x.T, divisions)]
        n_symbols = len(divisions[0]) + 1
    tau = resolve_multivariate_tau(tau, x)
    length = as_int(symbolic_length, 'symbolic_length')
    words = np.column_stack([past_codes(S, length, tau, n_symbols) for S in symbols])
    words = words[(words >= 0).all(axis=1)]
    if len(words) == 0:
        raise ValueError('Error: time-series too short for the chosen symbolic_length and tau')
    return words, n_symbols ** length


def entropy_of_rows(rows, log, size):
    '''Plug-in entropy of the joint distribution of the rows; every column takes values in [0, size).'''
    return entropy_of_codes(log, *[(col, size) for col in rows.T])


def past_codes(S, length, tau, n_symbols):
    '''Index of the symbolic word formed by the `length` tau-spaced symbols before n.

    The word at n uses S[n-tau*length], ..., S[n-tau], the oldest symbol being the
    least significant digit. Entries without a complete word are -1.
    '''
    S = np.asarray(S, dtype=np.int64)
    n = len(S)
    codes = np.full(n, -1, dtype=np.int64)
    start = tau * length
    if start < n:
        acc = np.zeros(n - start, dtype=np.int64)
        for k in range(length):
            acc += S[k * tau:n - start + k * tau] * n_symbols ** k
        codes[start:] = acc
    return codes


def future_codes(S, length, tau, n_symbols):
    '''Index of the symbolic word formed by S[n], S[n+tau], ..., S[n+tau*(length-1)].

    Entries without a complete word are -1.
    '''
    S = np.asarray(S, dtype=np.int64)
    n = len(S)
    codes = np.full(n, -1, dtype=np.int64)
    stop = n - tau * (length - 1)
    if stop > 0:
        acc = np.zeros(stop, dtype=np.int64)
        for k in range(length):
            acc += S[k * tau:stop + k * tau] * n_symbols ** k
        codes[:stop] = acc
    return codes


def histogram(codes, size):
    '''Normalized histogram of the non-negative codes.'''
    counts = np.bincount(codes[codes >= 0], minlength=size).astype(float)
    if counts.sum() == 0:
        raise ValueError('Error: time-series too short for the chosen symbolic_length and tau')
    return counts / counts.sum()


def entropy_from_probs(p, log):
    p = np.asarray(p).ravel()
    p = p[p > 0]
    return float(-np.sum(p * log(p))) + 0.0  # + 0.0 avoids returning -0.0


def divergence(p_joint, p_product, log):
    '''sum p_joint * log(p_joint / p_product) over cells where both are positive.'''
    mask = (p_joint > 0) & (p_product > 0)
    return float(np.sum(p_joint[mask] * log(p_joint[mask] / p_product[mask])))


def mutual_info_from_probs(p_x, p_y, p_xy, log):
    return divergence(p_xy, p_x[:, None] * p_y[None, :], log)


def cami_from_probs(p_xp, p_ypf, p_xypf, log):
    return divergence(p_xypf, p_xp[:, None, None] * p_ypf[None, :, :], log)


def pair_words(Sx, Sy, lengths, tau, n_symbols, allow_empty=False):
    '''Past-x, past-y and future-y word codes at every time where all three exist.'''
    lx, lyp, lyf = lengths
    a = past_codes(Sx, lx, tau, n_symbols)
    b = past_codes(Sy, lyp, tau, n_symbols)
    c = future_codes(Sy, lyf, tau, n_symbols)
    valid = (a >= 0) & (b >= 0) & (c >= 0)
    if not valid.any() and not allow_empty:
        raise ValueError('Error: time-series too short for the chosen symbolic_length and tau')
    return (a[valid], n_symbols ** lx), (b[valid], n_symbols ** lyp), (c[valid], n_symbols ** lyf)


def counting_method(size, n_samples):
    ''''dense' (np.bincount over the whole alphabet) when the alphabet is small enough,
    'sparse' (sort-based np.unique over the observed keys) otherwise.'''
    return 'dense' if size <= max(2 * n_samples, 2 ** 20) and size < 2 ** 31 else 'sparse'


def count_keys(key, size, method=None):
    '''Counts of every observed key, in increasing key order (zeros dropped).'''
    method = method or counting_method(size, len(key))
    if method == 'dense':
        counts = np.bincount(key, minlength=size)
        return counts[counts > 0]
    return np.unique(key, return_counts=True)[1]


def entropy_of_codes(log, *words, method=None):
    '''Plug-in joint entropy of (codes, alphabet size) pairs.'''
    size = 1
    for __, n in words:
        size *= n
    if size < 2 ** 62:
        key = np.zeros(len(words[0][0]), dtype=np.int64)
        for codes, n in words:
            key = key * n + codes
        counts = count_keys(key, size, method)
    else:
        __, counts = np.unique(np.column_stack([codes for codes, __ in words]), axis=0, return_counts=True)
    return entropy_from_probs(counts / counts.sum(), log)


def mutual_info_from_words(a, b, log):
    return entropy_of_codes(log, a) + entropy_of_codes(log, b) - entropy_of_codes(log, a, b)


def cami_from_words(a, b, c, log):
    return entropy_of_codes(log, a) + entropy_of_codes(log, b, c) - entropy_of_codes(log, a, b, c)


def linear_rate(values):
    '''Slope of values vs. L over the 3-point window closest to linear.

    Returns (slope, idx) where the window covers L = idx+1 .. idx+3.
    '''
    from scipy.stats import linregress
    diff2 = np.diff(values, 2)
    a = min(abs(diff2))
    find_a = np.where(diff2 == a)[0]
    idx = find_a[0] if len(find_a) > 0 else np.where(diff2 == -a)[0][0]
    return linregress(np.arange(3) + idx + 1, values[idx:idx + 3]).slope, idx
