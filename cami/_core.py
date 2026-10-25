'''Shared building blocks for the CaMI estimators.

Everything here is private API: input cleaning, symbolic encoding helpers,
symbolic-word indexing and the plug-in estimators computed from
probability tables.
'''
import logging
import warnings

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
    '''A 1-D float view or copy of a time-series (non-numeric values become NaN).

    The result may share memory with the input: callers must not modify it in place
    (_interpolate_nans copies before writing).
    '''
    if isinstance(x, (pd.Series, pd.DataFrame, pd.Index)):
        x = x.to_numpy()
    x = np.asarray(x)
    if x.dtype.kind in 'biuf':  # already numeric: no pandas round-trip, no copy for float64
        return np.asarray(x, dtype=float).ravel()
    return np.array(pd.to_numeric(x.ravel(), errors='coerce'), dtype=float)


def _interpolate_nans(data):
    bad = np.isnan(data)
    if bad.any():
        data = data.copy()  # never write into the caller's array
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


def symbol_dtype(n_symbols):
    return np.uint8 if n_symbols <= 256 else np.uint16 if n_symbols <= 65536 else np.int64


def encode(data, divisions):
    '''Symbol i is given to values in [divisions[i-1], divisions[i]) (compact integer dtype).'''
    divisions = np.asarray(divisions, dtype=float)
    if np.any(np.diff(divisions) < 0):
        raise ValueError('Error: partition divisions must be in increasing order')
    data = np.asarray(data, dtype=float)
    dtype = symbol_dtype(len(divisions) + 1)
    if len(divisions) > 16:
        return np.searchsorted(divisions, data, side='right').astype(dtype)
    # few divisions: one comparison pass each, without a temporary int64 array
    symbols = np.zeros(data.shape, dtype=dtype)
    for division in divisions:
        symbols += data >= division
    symbols[np.isnan(data)] = len(divisions)  # as np.searchsorted places NaN
    return symbols


def transform(data, symbolic_type, axis=-1):
    '''Returns the data to encode (the series or its 1st/2nd difference along axis) and the division placement.'''
    try:
        order, placement = SYMBOLIC_TYPES[symbolic_type]
    except (KeyError, TypeError):
        raise ValueError('Error: Unacceptable argument of symbolic type. See help on function.') from None
    return (np.diff(data, order, axis=axis) if order else np.asarray(data)), placement


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


def _segment_correlation(data, lag):
    with np.errstate(all='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return np.corrcoef(data[:-lag], data[lag:])[0, 1]


def first_zero_autocorrelation(data, direct_lags=8):
    '''Smallest lag at which the autocorrelation crosses zero (1 if it never does).

    The autocorrelation at lag i is corrcoef(data[:-i], data[i:]) (each segment with
    its own mean and variance). The first lags are computed directly; the rest come
    from one FFT and cumulative sums, in O(N log N). Lags whose sign is uncertain at
    floating-point precision are recomputed directly, so the result is the same as
    computing every lag with np.corrcoef.
    '''
    data = np.asarray(data, dtype=float)
    n = len(data)
    old_corr_val = 1
    for i in range(1, min(direct_lags + 1, n)):
        new_corr_val = _segment_correlation(data, i)
        if old_corr_val > 0 and new_corr_val <= 0:
            return i
        old_corr_val = new_corr_val
    if n <= direct_lags + 1:
        return 1
    d = data - data.mean()  # correlations are shift-invariant; centring improves precision
    lags = np.arange(direct_lags + 1, n)
    m = (n - lags).astype(float)
    csum = np.concatenate(([0.0], np.cumsum(d)))
    csq = np.concatenate(([0.0], np.cumsum(d * d)))
    size = 1 << int(np.ceil(np.log2(2 * n - 1)))
    spectrum = np.fft.rfft(d, size)
    cross = np.fft.irfft(spectrum * np.conj(spectrum), size)[lags]
    sum_a, sum_b = csum[n - lags], csum[n] - csum[lags]
    var_a = csq[n - lags] - sum_a ** 2 / m
    var_b = csq[n] - csq[lags] - sum_b ** 2 / m
    with np.errstate(all='ignore'):
        r = (cross - sum_a * sum_b / m) / np.sqrt(var_a * var_b)
    # recompute where rounding could change the sign or the segment is (nearly) constant
    scale = csq[n] / m
    uncertain = ~np.isfinite(r) | (np.abs(r) < 1e-9 * n / m) | (var_a <= 1e-9 * scale * m) | (var_b <= 1e-9 * scale * m)
    for k in np.flatnonzero(uncertain):
        r[k] = _segment_correlation(data, lags[k])
    previous = np.concatenate(([old_corr_val], r[:-1]))
    with np.errstate(invalid='ignore'):
        crossing = np.flatnonzero((previous > 0) & (r <= 0))
    return int(lags[crossing[0]]) if crossing.size else 1


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
    x, y = clean_pair(x, y)
    x, y = apply_delay(x, y, delay)
    x_divs, y_divs = resolve_pair_divisions(x_divs, y_divs)
    n_symbols = as_int(n_symbols, 'n_symbols')
    if x_divs is None:
        Sx, Sy = symbolize(x, symbolic_type, n_symbols), symbolize(y, symbolic_type, n_symbols)
    else:
        Sx, Sy = encode(x, x_divs), encode(y, y_divs)
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
    del x
    start, stop = tau * length, len(symbols[0])
    if stop <= start:
        raise ValueError('Error: time-series too short for the chosen symbolic_length and tau')
    return [past_window(S, length, tau, n_symbols, start, stop) for S in symbols], n_symbols ** length


def entropy_of_rows(columns, log, size):
    '''Plug-in joint entropy of columns (2-D array or list of arrays) taking values in [0, size).'''
    if isinstance(columns, np.ndarray) and columns.ndim == 2:
        columns = columns.T
    return entropy_of_codes(log, *[(col, size) for col in columns])


def code_dtype(size):
    if size >= 2 ** 63:
        raise ValueError('Error: n_symbols**symbolic_length is too large (more than 2**63 words)')
    return np.int32 if size < 2 ** 31 else np.int64


def past_window(S, length, tau, n_symbols, start, stop):
    '''Codes of the past words (as in past_codes) at the times n in [start, stop).

    S may be 2-D (time along axis 0, one column per trial).
    '''
    codes = np.zeros((stop - start,) + S.shape[1:], dtype=code_dtype(n_symbols ** length))
    for k in range(length - 1, -1, -1):  # Horner: oldest symbol is the least significant digit
        first = start - tau * length + k * tau
        codes *= n_symbols
        codes += S[first:first + stop - start]
    return codes


def future_window(S, length, tau, n_symbols, start, stop):
    '''Codes of the future words (as in future_codes) at the times n in [start, stop); S may be 2-D.'''
    codes = np.zeros((stop - start,) + S.shape[1:], dtype=code_dtype(n_symbols ** length))
    for k in range(length - 1, -1, -1):
        codes *= n_symbols
        codes += S[start + k * tau:stop + k * tau]
    return codes


def pair_window(n, lengths, tau):
    '''Times [start, stop) where the past of x, past of y and future of y all exist.'''
    lx, lyp, lyf = lengths
    return tau * max(lx, lyp), n - tau * (lyf - 1)


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
    start, stop = pair_window(len(Sx), lengths, tau)
    if stop <= start:
        if not allow_empty:
            raise ValueError('Error: time-series too short for the chosen symbolic_length and tau')
        start = stop = 0
    return ((past_window(Sx, lx, tau, n_symbols, start, stop), n_symbols ** lx),
            (past_window(Sy, lyp, tau, n_symbols, start, stop), n_symbols ** lyp),
            (future_window(Sy, lyf, tau, n_symbols, start, stop), n_symbols ** lyf))


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
        key = np.array(words[0][0], dtype=np.int64)
        for codes, n in words[1:]:
            key *= n
            key += codes
        counts = count_keys(key, size, method)
    else:
        __, counts = np.unique(np.column_stack([codes for codes, __ in words]), axis=0, return_counts=True)
    return entropy_from_probs(counts / counts.sum(), log)


def mutual_info_from_words(a, b, log):
    return entropy_of_codes(log, a) + entropy_of_codes(log, b) - entropy_of_codes(log, a, b)


def cami_from_words(a, b, c, log):
    return entropy_of_codes(log, a) + entropy_of_codes(log, b, c) - entropy_of_codes(log, a, b, c)


def _group_counts(sub_keys, counts, is_sorted):
    '''Counts aggregated by sub_key, in increasing sub_key order.'''
    if len(sub_keys) == 0:
        return counts
    if is_sorted:
        starts = np.flatnonzero(np.r_[True, sub_keys[1:] != sub_keys[:-1]])
        return np.add.reduceat(counts, starts)
    __, inverse = np.unique(sub_keys, return_inverse=True)
    return np.bincount(inverse, weights=counts)


def effect_key(b, c):
    '''Joint key of past and future effect words, b*nc + c (int64).'''
    (cb, __), (cc, nc) = b, c
    key = np.multiply(cb, nc, dtype=np.int64)
    key += cc
    return key


def sample_counts(key, size):
    '''For every sample, how many samples share its key.'''
    if counting_method(size, len(key)) == 'dense':
        return np.bincount(key, minlength=size)[key]
    __, inverse, counts = np.unique(key, return_inverse=True, return_counts=True)
    return counts[inverse]


def pointwise_values(a, b, c, log, normalized, out):
    '''Pointwise MI, CaMI and TE of every sample (direction a -> future c).

    Values are computed once per occupied word cell and gathered per sample:
    pcami = log p(abc)/(p(a) p(bc)), pte = log p(abc) p(b)/(p(ab) p(bc)),
    pmi = log p(ab)/(p(a) p(b)); normalized values are divided by -log p(abc)
    (-log p(ab) for pmi). The values are written into out = (pmi, pcami, pte)
    arrays (None to skip one). Returns pair_entropies of the same words.
    '''
    (ca, na), (cb, nb), (cc, nc) = a, b, c
    total = len(ca)
    key = np.multiply(ca, nb, dtype=np.int64)
    key += cb
    key *= nc
    key += cc
    size = na * nb * nc
    with np.errstate(divide='ignore', invalid='ignore'):
        if counting_method(size, total) == 'dense':
            counts = np.bincount(key, minlength=size).reshape(na, nb, nc)
            entropies = entropies_from_table(counts, log)
            n_abc = counts.astype(float)
            del counts
            n_ab = n_abc.sum(axis=2)
            n_bc = n_abc.sum(axis=0)
            n_a = n_ab.sum(axis=1)
            n_b = n_ab.sum(axis=0)
            pcami = log(n_abc * total / (n_a[:, None, None] * n_bc[None, :, :]))
            pte = log(n_abc * n_b[None, :, None] / (n_ab[:, :, None] * n_bc[None, :, :]))
            pmi = log(n_ab * total / (n_a[:, None] * n_b[None, :]))
            if normalized:
                pcami /= -log(n_abc / total)
                pte /= -log(n_abc / total)
                pmi /= -log(n_ab / total)
            pmi = np.broadcast_to(pmi[:, :, None], n_abc.shape)  # indexed by the same key
            tables, indices = (pmi, pcami, pte), (key, key, key)
        else:
            entropies = pair_entropies(a, b, c, log, method='sparse')
            keys, inverse, n_abc = np.unique(key, return_inverse=True, return_counts=True)
            del key
            ab_keys, bc_keys = keys // nc, keys % (nb * nc)
            a_keys, b_keys = ab_keys // nb, bc_keys // nc

            def group_count(sub_keys):  # count of each cell's marginal group, per unique cell
                __, group, __ = np.unique(sub_keys, return_inverse=True, return_counts=True)
                return np.bincount(group, weights=n_abc)[group]

            n_abc = n_abc.astype(float)
            n_ab, n_bc, n_a, n_b = (group_count(k) for k in (ab_keys, bc_keys, a_keys, b_keys))
            pcami = log(n_abc * total / (n_a * n_bc))
            pte = log(n_abc * n_b / (n_ab * n_bc))
            pmi = log(n_ab * total / (n_a * n_b))
            if normalized:
                pcami /= -log(n_abc / total)
                pte /= -log(n_abc / total)
                pmi /= -log(n_ab / total)
            tables, indices = (pmi, pcami, pte), (inverse,) * 3
    for table, index, target in zip(tables, indices, out):
        if target is not None:
            np.take(table.astype(target.dtype).ravel(), index, out=target)
    return entropies


def _entropy_of_counts(counts, log):
    counts = counts[counts > 0]
    return entropy_from_probs(counts / counts.sum(), log)


def entropies_from_table(table, log):
    '''pair_entropies from a dense (na, nb, nc) table of joint counts.'''
    ab = table.sum(axis=2)
    return {'a': _entropy_of_counts(ab.sum(axis=1), log), 'b': _entropy_of_counts(ab.sum(axis=0), log),
            'ab': _entropy_of_counts(ab.ravel(), log), 'bc': _entropy_of_counts(table.sum(axis=0).ravel(), log),
            'abc': _entropy_of_counts(table.ravel(), log)}


def pair_entropies(a, b, c, log, method=None, bc_key=None):
    '''Entropies H(a), H(b), H(a,b), H(b,c), H(a,b,c) from one joint count.

    a, b, c are (codes, alphabet size) pairs. The marginal counts are derived from
    the joint counts, in the same order as counting them separately, so the
    entropies are identical to separate entropy_of_codes calls. bc_key, if given,
    is effect_key(b, c), reused when only a changes (surrogates).
    '''
    (ca, na), (cb, nb), (cc, nc) = a, b, c
    size = na * nb * nc
    if size >= 2 ** 62:
        return {'a': entropy_of_codes(log, a), 'b': entropy_of_codes(log, b), 'ab': entropy_of_codes(log, a, b),
                'bc': entropy_of_codes(log, b, c), 'abc': entropy_of_codes(log, a, b, c)}
    if bc_key is None:  # built in place, without a second int64 array
        key = np.multiply(ca, nb, dtype=np.int64)
        key += cb
        key *= nc
        key += cc
    else:
        key = np.multiply(ca, nb * nc, dtype=np.int64)
        key += bc_key

    def entropy(counts):
        counts = counts[counts > 0]
        return entropy_from_probs(counts / counts.sum(), log)

    method = method or counting_method(size, len(key))
    if method == 'dense':
        return entropies_from_table(np.bincount(key, minlength=size).reshape(na, nb, nc), log)
    keys, counts = np.unique(key, return_counts=True)
    del key
    ab_keys = keys // nc
    bc_keys = keys % (nb * nc)
    return {'a': entropy(_group_counts(ab_keys // nb, counts, True)),
            'b': entropy(_group_counts(bc_keys // nc, counts, False)),
            'ab': entropy(_group_counts(ab_keys, counts, True)),
            'bc': entropy(_group_counts(bc_keys, counts, False)),
            'abc': entropy(counts)}


def measures_from_entropies(h):
    '''(CaMI, mutual information, transfer entropy) from pair_entropies output.'''
    cami = h['a'] + h['bc'] - h['abc']
    mutual_info = h['a'] + h['b'] - h['ab']
    return cami, mutual_info, cami - mutual_info


class Prepared:
    '''A cleaned, delayed and encoded pair of series with cached symbolic word codes.

    Word codes are extended one symbol at a time when longer words are requested,
    and only the longest computed length is kept per (kind, variable).
    '''

    def __init__(self, x, y, x_divs, y_divs, symbolic_type, n_symbols, tau, delay):
        self.x, self.y, Sx, Sy, self.n_symbols, self.tau = prepare_pair(
            x, y, x_divs, y_divs, symbolic_type, n_symbols, tau, delay)
        self.symbols = {'x': Sx, 'y': Sy}
        self.n = len(Sx)
        self._cache = {}

    def with_x_symbols(self, Sx):
        '''A copy using other symbols for x (same length), sharing y's symbols and cached codes.'''
        other = object.__new__(Prepared)
        other.__dict__.update(self.__dict__)
        other.symbols = {'x': Sx, 'y': self.symbols['y']}
        other._cache = {key: value for key, value in self._cache.items() if key[1] == 'y'}
        return other

    def _codes(self, kind, var, length):
        cached = self._cache.get((kind, var))
        if cached is not None and cached[0] == length:
            return cached[1]
        S, tau, ns, n = self.symbols[var], self.tau, self.n_symbols, self.n
        if kind == 'past':
            start = tau * length
            if start >= n:
                codes = np.zeros(0, dtype=np.int32)
            elif cached is not None and cached[0] == length - 1:
                codes = cached[1][tau:].astype(code_dtype(ns ** length))
                codes *= ns
                codes += S[:n - start]
            else:
                codes = past_window(S, length, tau, ns, start, n)
        else:
            stop = n - tau * (length - 1)
            if stop <= 0:
                codes = np.zeros(0, dtype=np.int32)
            elif cached is not None and cached[0] == length - 1:
                codes = cached[1][:stop].astype(code_dtype(ns ** length))
                codes += S[tau * (length - 1):].astype(codes.dtype) * codes.dtype.type(ns ** (length - 1))
            else:
                codes = future_window(S, length, tau, ns, 0, stop)
        self._cache[(kind, var)] = (length, codes)
        return codes

    def past(self, var, length):
        '''Past word codes of var for the times n in [tau*length, n).'''
        return self._codes('past', var, length)

    def future(self, var, length):
        '''Future word codes of var for the times n in [0, n - tau*(length-1)).'''
        return self._codes('future', var, length)

    def words(self, direction, lengths):
        '''(codes, size) of past cause, past effect and future effect over their common window.'''
        cause, effect = ('x', 'y') if direction == 'xy' else ('y', 'x')
        lx, lyp, lyf = lengths
        start, stop = pair_window(self.n, lengths, self.tau)
        if stop <= start:
            raise ValueError('Error: time-series too short for the chosen symbolic_length and tau')
        tau, ns = self.tau, self.n_symbols
        return ((self.past(cause, lx)[start - tau * lx:stop - tau * lx], ns ** lx),
                (self.past(effect, lyp)[start - tau * lyp:stop - tau * lyp], ns ** lyp),
                (self.future(effect, lyf)[start:stop], ns ** lyf))

    def entropies(self, direction, lengths, log, keep_future=False):
        '''pair_entropies of the words; future codes are dropped afterwards unless they
        will be extended (keep_future), as each direction uses its own.'''
        result = pair_entropies(*self.words(direction, lengths), log)
        if not keep_future:
            self.drop_future(direction)
        return result

    def drop_future(self, direction):
        '''Frees the cached future codes of the effect of this direction.'''
        self._cache.pop(('future', 'y' if direction == 'xy' else 'x'), None)


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
