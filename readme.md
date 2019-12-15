uncertaintypropagation

# Uncertainty propagation for nonlinear functions

This is an implementation of three uncertainty propagation methods: Linear uncertainty propagation (e.g. as used in the (Extended) Kalman Filter), unscented transform (e.g. as used in the Unscented Kalman Filter), and a quadratic uncertainty propagation method.

The implementation uses `pytorch` and `numpy`.


## Requirements

Requirements are in `requirements.txt`. Note that they also include the requirements for the example. `torchdiffeq` must be installed manually from [GitHub](https://github.com/rtqichen/torchdiffeq) in order to run the example.


## Usage

For example usage and pretty plots, see `example.py`.
Uncertainty propagation is there demonstrated for a time series: a chaotic attractor system that is simulated using `torchdiffeq`. The resulting figures compare a Monte Carlo simulation's sample mean and covariance diagonal over time with the mean and covariance diagonal predicted by the uncertainty propagation methods.

```
python3 example.py
```


## Sample output from example.py

[img-1](plots/chaotic_time_series.pdf)
