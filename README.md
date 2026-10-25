# CaMI-Python: causality toolbox for Python

Information-theoretical tools to study dynamics and causal influence between
variables from their time-series: entropy, mutual information, **Causal Mutual
Information (CaMI)**, transfer entropy, the directionality index, their rates,
pointwise (local) versions, total correlation, many-short-trials estimation and
surrogate-data significance tests.

## Installation

```
pip install git+https://github.com/artvalencio/cami-python
```

Requirements: Python 3.9 or later, numpy (>= 1.20), pandas (>= 1.1), scipy
(>= 1.5) and matplotlib (>= 3.3, only needed for plots). They are installed
automatically; `requirements.txt` lists them for manual installation.

## Quick start

```python
import numpy as np
import cami

rng = np.random.default_rng(0)
x = rng.random(5000)
y = np.roll(x, 1)                         # y follows x one step later: X drives Y

cami.cami(x, y, two_sided=True)           # (~1.0, ~0.0) bits: information flows X -> Y
cami.transfer_entropy(x, y)               # ~1.0 bits
cami.directionality(x, y)                 # > 0: net flow X -> Y
cami.mutual_info(x, y, delay=1)           # x[n] vs y[n+1]: ~1.0 bits

cami.measures(x, y, n_symbols=4, symbolic_length=2)   # all of the above in one pass

# is the measure above chance? (surrogate data)
cami.surrogate_test(x, y, measure='transfer_entropy', n_surrogates=99, seed=0)

# many short experiments of the same phenomenon (columns are trials)
cami.multithread_causality(x.reshape(100, 50), y.reshape(100, 50), symbolic_length=2)
```

Messages such as the automatically selected `tau` go to the `cami` logger;
to see them: `import logging; logging.basicConfig(level=logging.INFO)`.

## How the measures are computed

1. **Symbolic encoding.** Each series is split into `n_symbols` cells and every
   value is replaced by its cell index. The divisions are either given
   (`x_divs`, `y_divs`) or placed automatically by `symbolic_type`:

   | `symbolic_type` | encodes | divisions |
   |---|---|---|
   | `'equal-divs'` (default) | values | equal-width cells |
   | `'equal-points'` | values | equal number of points per cell |
   | `'equal-growth'`, `'equal-growth-points'` | first differences | as above |
   | `'equal-concavity'`, `'equal-concavity-points'` | second differences | as above |

2. **Symbolic words.** Sequences of `L` symbols spaced by `tau` form words:
   the past of x (`lx` symbols before time n), the past of y (`lyp`) and the
   future of y (`lyf` symbols from n on). `symbolic_length` is `L`, `(lx, ly)`
   or `(lx, lyp, lyf)`; `tau=None` chooses the first zero crossing of the
   auto-correlation. `delay` pairs `x[n]` with `y[n + delay]`.
3. **Plug-in estimates.** Word frequencies give the probabilities, and

   - MI = H(X_past) + H(Y_past) − H(X_past, Y_past)
   - CaMI(X→Y) = H(X_past) + H(Y_past, Y_future) − H(X_past, Y_past, Y_future)
   - TE(X→Y) = CaMI(X→Y) − MI
   - DI = CaMI(X→Y) − CaMI(Y→X)

   Units: `'bits'` (default), `'nat'` or `'ban'`.

Edge NaNs are trimmed and interior NaNs interpolated; inputs may be lists,
numpy arrays or pandas Series and are never modified.

## Functions

| Function | Returns |
|---|---|
| `entropy(x, ...)` | Shannon entropy of the symbolic words of x |
| `joint_entropy(x, y, ...)` | joint entropy of x and y |
| `mutual_info(x, y, ...)` | mutual information |
| `cami(x, y, ..., two_sided=False)` | Causal Mutual Information X→Y (and Y→X) |
| `transfer_entropy(x, y, ..., two_sided=False)` | transfer entropy X→Y (and Y→X) |
| `directionality(x, y, ...)` | directionality index |
| `measures(x, y, ...)` | dict with all of the above in one pass |
| `mutual_info_rate`, `cami_rate`, `transfer_entropy_rate` | slope of the measure vs. symbolic length |
| `pointwise(x, y, ...)` | time-series of the local (pointwise) MI, CaMI, TE and DI, with optional plots |
| `multivar_joint_entropy(X, ...)`, `total_correlation(X, ...)` | joint entropy and total correlation of the columns of X |
| `multithread_causality(X, Y, ...)` | measures pooled over many short trials (columns or a list of series) |
| `surrogate_test(x, y, measure=..., ...)` | value, surrogate distribution, threshold and p-value |
| `symbolic_encoding(x, y, ...)`, `get_prob(Sx, Sy, ...)` | the symbols, and the word probability tables (`sparse=True` for large alphabets) |

Every function is documented: `help(cami.cami)`.

## Performance

Estimates count only the observed symbolic words (dense counting for small
alphabets, sorted keys for large ones), so long words and large alphabets
work, and series longer than 2^20 samples are counted in chunks. Version 1.5.0
compared with 1.403 (same results), N = 10^6 samples:

| Call | 1.403 | 1.5.0 |
|---|---|---|
| `cami(two_sided=True)`, L=3, 4 symbols | 619 ms / 81 MB | 52 ms / 28 MB |
| `transfer_entropy(two_sided=True)`, L=2 | 728 ms | 37 ms |
| `cami_rate` (L = 1..6) | 2355 ms | 120 ms |
| `total_correlation`, 4 variables | 5027 ms / 202 MB | 128 ms / 65 MB |
| `pointwise(two_sided=True)`, L=3 | 913 ms / 146 MB | 202 ms / 79 MB |
| `surrogate_test`, 99 surrogates (N = 10^5) | 1167 ms | 70 ms |
| automatic tau (N = 2·10^4) | 844 ms | 13 ms |

`surrogate_test` also accepts `n_jobs` (threads) and a `progress` callback.
Benchmarks: `python benchmarks/run.py LABEL`.

## Changes in 1.5.0

Bug fixes that change numerical results:
- marginal probabilities are derived from the joint distribution (they used
  a wider time window, so e.g. the MI of a constant and a random series was
  not zero), and the last `tau` samples are now used as future words
- `transfer_entropy(two_sided=True)` returned a wrong (possibly negative)
  Y→X value with asymmetric symbolic lengths
- `multivar_joint_entropy` and `total_correlation` used pooled marginals
  instead of the joint distribution; divisions are now placed per variable
- `'equal-concavity'` encoded first instead of second differences; negative
  delays reduced both series to one sample; pointwise values were misaligned
  for unequal symbolic lengths

Other fixes: pandas Series and numpy-array divisions are accepted everywhere,
inputs are not modified, `'equal-growth-points'` works, clear errors for
series that are too short, all-NaN or of different lengths, and the library
logs instead of printing.

New: `measures`, `multithread_causality`, `surrogate_test`,
`get_prob(sparse=True)`, `pointwise(dtype=...)`.

## Tests

```
pip install -e .[test]
pytest              # fast tests
pytest -m slow      # time and memory budgets
```

## Licence and acknowledgements

Open source licence: GNU GPL v3 (see `License.MD`).

Developers: Arthur Valencio [1], Norma Valencio [2], Murilo S. Baptista [3]

[1] Institute of Computing/University of Campinas and Research, Innovation and Dissemination Center for Neuromathematics
[2] Department of Environmental Sciences/Federal University of São Carlos
[3] Institute for Complex Systems and Mathematical Biology/University of Aberdeen

Funding agencies: FAPESP (S. Paulo Research Foundation), Brazil (2018-onwards). CNPq (National Council for Scientific and Technologic Development), Brazil (2015-2018).

Acknowledgements: Arthur Valencio is supported by the FAPESP grant 2018/09900-8 and acknowledges this work as part of the activities of the FAPESP Research Innovation and Dissemination Center for Neuromathematics (grant 2013/07699-0, S. Paulo Research Foundation). A.V. contributions are partly derived from previous research funded by CNPq grant 206246/2014-5. N.V. is supported by CNPq research productivity fellowship (grant ). N.V. and M.S.B. are supported by The Royal Society (grant ).

```
CaMI Python: package for causality inference using information theory

Copyright (C) 2020  Arthur Valencio, Norma Valencio, Murilo S. Baptista

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
```

## Contact

Arthur Valencio — e-mail: arthuric(at)unicamp.br

Institute of Computing, Unicamp, Av. Albert Einstein, 1251, Cidade Universitaria, Campinas/SP, Brazil, CEP 13083-852
