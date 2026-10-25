'''Shared building blocks for the CaMI estimators.

Everything here is private API: input cleaning, symbolic encoding helpers,
symbolic-word indexing and the plug-in estimators computed from
probability tables.
'''
import numpy as np
import pandas as pd

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
    x = np.asarray(x).ravel()
    return np.array(pd.to_numeric(x, errors='coerce'), dtype=float)


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
        print('Selected tau=', tau, ' by the method of first zero of auto-correlation', sep='')
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

    Entries without a complete word are -1. The last word starts at n = len(S)-tau*length-1,
    mirroring the original implementation.
    '''
    S = np.asarray(S, dtype=np.int64)
    n = len(S)
    codes = np.full(n, -1, dtype=np.int64)
    stop = n - tau * length
    if stop > 0:
        acc = np.zeros(stop, dtype=np.int64)
        for k in range(length):
            acc += S[k * tau:stop + k * tau] * n_symbols ** k
        codes[:stop] = acc
    return codes


def histogram(codes, size):
    '''Normalized histogram of the non-negative codes.'''
    counts = np.bincount(codes[codes >= 0], minlength=size).astype(float)
    return counts / counts.sum()


def entropy_from_probs(p, log):
    p = np.asarray(p).ravel()
    p = p[p > 0]
    return float(-np.sum(p * log(p)))


def divergence(p_joint, p_product, log):
    '''sum p_joint * log(p_joint / p_product) over cells where both are positive.'''
    mask = (p_joint > 0) & (p_product > 0)
    return float(np.sum(p_joint[mask] * log(p_joint[mask] / p_product[mask])))


def mutual_info_from_probs(p_x, p_y, p_xy, log):
    return divergence(p_xy, p_x[:, None] * p_y[None, :], log)


def cami_from_probs(p_xp, p_ypf, p_xypf, log):
    return divergence(p_xypf, p_xp[:, None, None] * p_ypf[None, :, :], log)


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
