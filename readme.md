
<img src="https://gitlab.com/paloha/redistributor/uploads/e1bbea08834112646af45e6917324379/avatar.png" alt="match_colors" width="20%">

# Redistributor

**Redistributor** is a Python package which forces a collection of scalar samples to follow a desired distribution. When given independent and identically distributed samples of some random variable $S$ and the continuous cumulative distribution function of some desired target $T$, it provably produces a consistent estimator of the transformation $R$ which satisfies $R(S)=T$ in distribution. As the distribution of $S$ or $T$ may be unknown, we also include algorithms for efficiently estimating these distributions from samples. This allows for various interesting use cases in image processing, where Redistributor serves as a remarkably simple and easy-to-use tool that is capable of producing visually appealing results. The package is implemented in Python and is optimized to efficiently handle large data sets, making it also suitable as a preprocessing step in machine learning.
<br>

<img src="https://gitlab.com/paloha/redistributor/uploads/ce5305668697d3bdf6035c839aceb2c2/match_colors.jpg" alt="Example of matching colors" width="100%">
<small><i><center>Matching colors of a reference image – one of the use cases of Redistributor</center></i></small>

## Installation

<!-- ```bash
pip install redistributor
``` -->
Install the latest version directly from the [repository](https://gitlab.com/paloha/redistributor):
```bash
pip install git+https://gitlab.com/paloha/redistributor
```

## Quick-start

```python
from redistributor import Redistributor as R
from redistributor import LearnedDistribution as L
from scipy.stats import dgamma, norm

S = dgamma(7).rvs(size=1000)  # Samples from source distribution
target = norm(0, 1)  # In this example, target is set explicitly
r = R(source=L(S), target=target)  # Estimate the transformation
output = r.transform(S)  # Data now follows the target distribution
```
More in `examples.ipynb`.

## Documentation
Documentation is available in `docs` folder.


## News & Changelog

* :hammer: Package is still under development
* 2022.10 - Preprint published on ArXiv :tada:
* 2022.09 - Redistributor v1.0 (complete rewrite)
* 2021.10 - Redistributor v0.2 (generalization to arbitrary source & target)
* 2018.08 - Introducing Redistributor (generalization to arbitrary target)
* 2018.07 - Introducing Gaussifier package (now deprecated)

## How to cite

If you use Redistributor in your research, please cite the following paper:
```
@article{redistributor2022,
  title={Redistributor: Transforming Empirical Data Distributions},
  author={Harar, P. and Elbrächter, D. and Dörfler, M. and Johnson, K.},
  eprinttype={ArXiv},
  eprint={...}
}
```

## License
This project is licensed under the terms of the MIT license.
See `license.txt` for details.
