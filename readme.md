:warning: | Still under development
:---: | :---

This repository introduces two main classes, namely **Redistributor** and **LearnedDistribution**.

**Redistributor** is a tool for automatic transformation of empirical data distributions. It is implemented as a **Scikit-learn transformer**. It allows users to transform their data from arbitrary distribution into another arbitrary distribution. The source and target distributions, if known beforehand, can be specified exactly (e.g. as a [Continuous Scipy distribution](https://docs.scipy.org/doc/scipy/reference/tutorial/stats/continuous.html#continuous-distributions-in-scipy-stats) or any other class which has `cdf` and `pdf` methods implemented), or can be inferred from the data using `LearnedDistribution` class. Transformation is **piece-wise-linear, monotonic, invertible**, and can be **saved for later use** on different data assuming the same source distribution.

**LearnedDistribution** is a subclass of [Scipy.stats.rv_continous](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.html#scipy-stats-rv-continuous) class. It is a continuous random variable obtained by estimating the empirical distribution of a user provided array of numeric data `x`. It can be used to sample new random points from the learned distribution.


<!-- The empirical distribution can be inferred from a 1D array of data. To redistribute multiple slices of your data use `Redistributor_multi` class which has a **low memory footprint** and utilizes **parallel computing** to apply multiple `Redistributor` objects. -->

## Installation

:warning: | Not yet published on PyPi. Coming soon.
:---: | :---

The code is hosted in this [GitLab repository](https://gitlab.com/paloha/redistributor).
To install the released version from Pypi use:

```bash
pip install redistributor
```
Or install the bleeding edge directly from git:
```bash
pip install git+https://gitlab.com/paloha/redistributor
```
For development, install the package in editable mode with extra dependencies for documentation and testing:
```bash
# Clone the repository
git clone git@gitlab.com:paloha/redistributor.git
cd redistributor

 # Use virtual environment [optional]
python3 -m virtualenv .venv
source .venv/bin/activate

# Install with pip in editable mode
pip install -e .[dev]
```

## Compatibility
...

## Dependencies

Required packages for `Redistributor` are specified in the `install_requires` list in the `setup.py` file.

Extra dependencies for running the tests, compiling the documentation, or running the examples are specified in the `extras_require` dictionary in the same file.

The full version-locked list of dependencies and subdependencies is frozen in `requirements.txt`. Installing with `pip install -r requirements.txt` in a virtual environment should always lead to a fully functional project.



[comment]: <> (written in katex https://katex.org/docs/supported.html)

## Mathematical description

Assume we are given data \(x\sim S\) distributed according to some source distribution \(S\) on \(\mathbb{R}\) and our goal is to find a transformation \(R\) such that \(R(x)\sim T\) for some target distribution \(T\) on \(\mathbb{R}\).

One can mathematically show that a suitable \(R\colon \mathbb{R} \to \mathbb{R}\) is given by
$$
R := F_{T}^{-1} \circ F_{S},
$$
where \(F_S\) and \(F_T\) are the cumulative distribution functions of \(S\) and \(T\), respectively.

If \(S\) and \(T\) is unknown, one can use approximations \(\tilde{F}_S\) and \(\tilde{F}_T\) of the corresponding cumulative distribution functions given by interpolating (partially) sorted data
$$
(x_i)_{i=1}^N \ \text{with} \ x_i \sim S
$$
$$
(y_i)_{i=1}^M \ \text{with} \ y_i \sim T.
$$
Defining
$$
\tilde{R} := \tilde{F}_{T}^{-1} \circ \tilde{F}_S,
$$
one can, under suitable conditions, show that
$$
\tilde{R} \xrightarrow[N,M\to \infty]{} R.
$$

## How to cite
...

## License
This project is licensed under the terms of the MIT license.
See `license.txt` for details.

## Acknowledgement
This work was supported by the *International Mobility of Researchers* (program call no.: [CZ.02.2.69/0.0/0.0/16027/0008371](https://opvvv.msmt.cz/vyzva/vyzva-c-02-16-027-mezinarodni-mobilita-vyzkumnych-pracovniku.htm)).
![opvvv](https://gitlab.com/paloha/redistributor/uploads/19903a1b9e00015faa2b61234a99b911/opvvv.jpg)
