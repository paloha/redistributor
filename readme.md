# Redistributor
An algorithm for automatic transformation of data from arbitrary distribution into arbitrary distribution. Source distribution can be known beforehand and or learned from the data. Transformation is piecewise smooth, monotonic and invertible.

Implemented as a Scikit-learn transformer. Can be fitted on 1D vector (for more dimensions use Redistributor_multi wrapper) and saved to be used later for transforming other data assuming the same source distribution.

More detailed description can be found in the docstring of the Redistributor class which is located in the file ```redistributor/redistributor.py```.

# Redistributor_multi
Multi-dimensional wrapper for Redistributor. Allows to use multiple Redistributors on equal-sized slices (submatrices) of N-dimensional input array. Utilizes parallel processing with low memory footprint.

## Installation
```
git clone git@gitlab.com:paloha/redistributor.git
cd redistributor
python3 -m virtualenv .venv
source .venv/bin/activate
pip install .
```

## Example of usage
Simplest possible example:
```
from redistributor import Redistributor
r = Redistributor(bbox=bounding_box)
r.fit(training_data)
transformed = r.transform(validation_data)
inversed = r.inverse_transform(transformed)
```
For more detailed examples, take a look into the iPython notebooks.

## Features

#### Redistributor

* Data can be from arbitrary interval
* Source distribution can be specified explicitly
* Arbitrary target distribution can be used
* Possibility of computing empirical error of fit
* Handles small amount of non unique elements in data

#### Redistributor_multidim

* Applies multiple Redistributors to subarrays at once
* Utilizes parallel computing
* Has low memory footprint

# License
This project is licensed under the terms of the MIT license.
See license.txt for details.
