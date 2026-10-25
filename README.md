# CaMI-Python: Causality toolbox for Python.

This package provides functions for dynamical and causality
investigation using information theory and nonlinear time-series
analysis tools

-----------------------------------------------------------------------------
-----------------------------------------------------------------------------

Requirements:

Python>=3.8, numpy, pandas, scipy and matplotlib (installed automatically).

-----------------------------------------------------------------------------
-----------------------------------------------------------------------------
Installation:

    pip install git+https://github.com/artvalencio/cami-python

Now you can import the package in Python 3 using: import cami

-----------------------------------------------------------------------------
Quick start:

    import numpy as np
    import cami

    rng = np.random.default_rng(0)
    x = rng.random(5000)
    y = np.roll(x, 1)                      # y follows x one step later

    cami.cami(x, y, two_sided=True)        # (~1.0, ~0.0) bits: X -> Y
    cami.transfer_entropy(x, y)
    cami.directionality(x, y)              # > 0: net flow X -> Y
    cami.mutual_info(x, y, delay=1)

    # many short experiments (columns are trials) pooled together
    cami.multithread_causality(x_trials, y_trials, symbolic_length=2)

    # is the measure above chance level? (surrogate data test)
    cami.surrogate_test(x, y, measure='transfer_entropy', n_surrogates=99, seed=0)

Main functions: entropy, joint_entropy, mutual_info, cami, transfer_entropy,
directionality, the *_rate versions, pointwise, multivar_joint_entropy,
total_correlation, multithread_causality, surrogate_test, symbolic_encoding
and get_prob. Use help(cami.<function>) for the parameters.

Messages such as the automatically selected tau are sent to the 'cami'
logger. To see them:

    import logging; logging.basicConfig(level=logging.INFO)

-----------------------------------------------------------------------------
Changes in 1.5.0:

Bug fixes that change numerical results:
- marginal probabilities are derived from the joint distribution (they
  used a wider time window, so e.g. MI of a constant and a random series
  was non-zero)
- the last tau samples of a series are now used as future words
- with asymmetric symbolic lengths, transfer_entropy(two_sided=True)
  returned a wrong (possibly negative) Y->X value
- multivar_joint_entropy and total_correlation were computed from pooled
  marginals instead of the joint distribution; divisions are now placed
  per variable
- symbolic_type='equal-concavity' encoded first instead of second differences
- negative delays reduced both series to a single sample
- pointwise values were misaligned for unequal symbolic lengths

Other fixes: pandas Series and numpy-array divisions are accepted
everywhere, input arrays are no longer modified, 'equal-growth-points'
works, clear errors for series that are too short, all-NaN or of
different lengths, and the library no longer prints.

Performance: estimators are vectorized and count only the observed
symbolic words, so large alphabets and long words no longer need huge
dense tables.

-----------------------------------------------------------------------------
Running the tests:

    pip install -e .[test]
    pytest              # add -m "not slow" to skip performance tests

-----------------------------------------------------------------------------

-----------------------------------------------------------------------------

Licence and acknoledgements

Open source licence: GNU GPL v3

Developers: Arthur Valencio [1], Norma Valencio [2], Murilo S. Baptista [3]

Institutions: 

[1] Institute of Computing/University of Campinas and Research, Innovation and Dissemination Center for Neuromathematics

[2] Department of Environmental Sciences/Federal University of São Carlos

[3] Institute for Complex Systems and Mathematical Biology/University of Aberdeen

Funding agencies: FAPESP (S. Paulo Research Foundation), Brazil (2018-onwards). CNPq (National Council for Scientific and Technologic Development), Brazil (2015-2018).

Acknowledgements: Arthur Valencio is supported by the FAPESP grant 2018/09900-8 and 	     acknowledges this work as part of the activities of the FAPESP Research 	     Innovation and Dissemination Center for Neuromathematics (grant 2013/07699-0, S. Paulo Research Foundation). A.V. contributions are partly derived from previous research funded by CNPq grant 206246/2014-5. N.V. is supported by CNPq research productivity fellowship (grant ). N.V. and M.S.B. are supported by The Royal Society (grant ).

Licence text:

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

-----------------------------------------------------------------------------

Contact:

Arthur Valencio

e-mail: arthuric(at)unicamp.br

address:

Institute of Computing, Unicamp

Av. Albert Einstein, 1251

Cidade Universitaria

Campinas/SP, Brazil

CEP 13083-852 
