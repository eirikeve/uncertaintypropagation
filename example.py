from time import process_time

import numpy as np
import torch
import torch.nn as nn
import matplotlib
from matplotlib import cycler
from matplotlib import rc
import matplotlib.pyplot as plt
import torchdiffeq

import methods

"""
example.py
Eirik Vesterkjær, 2019

Demonstration of three uncertainty propagatation methods on a chaotic attractor system

For applying this on a machine learning model, the vector field can be replaced with e.g. a neural network
"""

def chaotic_attractor_vector_field(t, x):
    d = torch.tensor([ x[1],
                      -x[0] + x[1]*x[2],
                      1 - x[1]*x[1]
                    ])
    return d

class StateTransition(nn.Module):
    """
    Wrapper class for creating a transition function from a vector field
    """
    def __init__(self, dx, dt, torchdiffeq_method='rk4'):
        """
        @arg dx: vector field (see chaotic_attractor_vector_field)
        @arg dt: time step length
        @torchdiffeq_method: ODE solver method for torchdiffeq.odeint
        """
        nn.Module.__init__(self)
        self.dt = dt
        self.step_time = torch.tensor([0., dt])
        self.dx = dx
        self.method = torchdiffeq_method
    def forward(self, x):
        """
        transition from a state to the next time step's state
        @arg x: (D) tensor state. Can be a (N, D) tensor,
                    in which case the transition is applied separately (not sequentially)
                    for all N inputs
        """
        # Handle batch inputs as well as single inputs
        if len(x.shape) <= 1:
            x = x.view(1,-1)
        ys = torch.zeros((x.shape[0], 3), requires_grad=True)
        for i in range(ys.shape[0]):
            y = torchdiffeq.odeint(self.dx, x[i], self.step_time, method=self.method)[-1]
            ys[i] = y
        return ys


def run_example():

    # Step length [s]
    dt = 0.1
    # Simulation time horizon [s]
    tmax = 10
    # Approximate ground truth distribution over time with Monte Carlo simulation
    run_with_monte_carlo = True
    # Torchdiffeq ODE solver method, e.g. 'rk4' or 'adams'
    torchdiffeq_method = 'rk4'
    # Initial state
    x0 = torch.tensor([-1., -1., -1.])
    # Initial covariance
    sigma_scale = 0.01
    Sigma_x0 = sigma_scale * torch.tensor( [
        [1.0, 0.0, 0.0 ],
        [0.0, 1.0, 0.0 ],
        [0.0, 0.0, 1.0 ]])

    # Unscented transform Kappa (should be n-3 where n is the number of dimensions of the input)
    UT_kappa = 0.0
    UT = methods.UnscentedTf(kappa=UT_kappa)
    Lin = methods.LinearApprox()
    Quad = methods.QuadraticApprox()
    UP_methods = [Lin, Quad, UT]

    # State transition function
    state_transition = StateTransition(chaotic_attractor_vector_field, dt, torchdiffeq_method)

    # Memory allocation
    steps = int(tmax/dt)
    ts = torch.linspace(0, tmax, steps)
    xs = torch.zeros((len(UP_methods), steps, len(x0)), requires_grad=False)
    Sigma_xs = torch.zeros((len(UP_methods), steps, len(x0), len(x0)), requires_grad=False)

    if run_with_monte_carlo:
        gt_samples=  250
        MonteCarlo = methods.MonteCarlo(gt_samples)
        gt = MonteCarlo.sample_distribution(gt_samples, x0, Sigma_x0)
        mus_gt = torch.zeros((steps, len(x0)), requires_grad=False)
        Sigmas_gt = torch.zeros((steps, len(x0), len(x0)), requires_grad=False)
        mus_gt[0,:],  Sigmas_gt[0,:,:] = sample_mean_and_cov(gt)

    # Set initial values:
    for m in range(len(UP_methods)):
        xs[m,0,:] = x0
        Sigma_xs[m,0,:,:] = Sigma_x0

    # Run uncertainty propagation
    t0 = process_time()
    for step in range(steps-1):
        print(f"Progress: [{100.*step/steps:.2f}%]", end="\r")
        # propagate GT
        with torch.no_grad():
            if run_with_monte_carlo:
                for i in range(gt.shape[0]):
                    gt[i] = state_transition(gt[i]).detach()
                mu, Sigma = sample_mean_and_cov(gt)
                mus_gt[step+1,:] = mu.detach()
                Sigmas_gt[step+1,:,:] = Sigma.detach()
        # propagate all methods
        for m, method in enumerate(UP_methods):
            mu, Sigma = \
                        method.apply(state_transition, xs[m,step,:], Sigma_xs[m,step,:,:])
            #print(Sigma)
            #if m == 0 or m==1:
            #    Sigma = Sigma + eps
            xs[m,step+1,:], Sigma_xs[m,step+1,:,:] = mu.detach(), Sigma.detach()

    t1 = process_time()
    exec_time = t1 - t0
    print(f"Propagation time for {steps} steps of {dt} sec:", exec_time)

    # The diagonal of this is the states' variances
    sqrt_Sigma_xs = Sigma_xs**0.5
    if run_with_monte_carlo:
        sqrt_Sigma_mu_gt = Sigmas_gt**0.5

    set_plot_rc()
    for m, method in enumerate(UP_methods):
        method_name = type(method).__qualname__
        fig, axs = plt.subplots(nrows=len(x0), figsize=(8, 10))
        axs = axs.reshape(-1)

        for state,ax in enumerate(axs):
            plt.sca(ax)
            if state == 0:
                plt.title(r"$\hat{\mathbf{x}}_{"+method_name+r"}(t) \pm 3 STD$ for $\Sigma_{x,0} = "+str(sigma_scale)+"\cdot\mathbf{I}$, $\Delta_t = "+str(dt)+"$")
            plt.ylabel("$x_{"+str(state+1)+"}$")
            plt.xlabel("t")
            if run_with_monte_carlo:
                plot_with_confidence(ts, mus_gt[:,state].detach().numpy(), sqrt_Sigma_mu_gt[:,state,state].detach().numpy(), label="MonteCarlo")
            plot_with_confidence(ts, xs[m,:,state].detach().numpy(), sqrt_Sigma_xs[m,:,state,state].detach().numpy(), label=method_name)
            plt.legend()
        plt.tight_layout()
    plt.show()
    input("Press [Enter] to exit")


#########################################
# Utilities
#########################################

def set_plot_rc(marker=False, linewidth=2, usetex=True):
    """
    set_plot_rc
    sets matplotlib plot style
    """
    if marker:
        colors = (cycler(color=['#EE6666', '#3366BB', '#9988DD',  '#77AA33', '#FF8888', '#000000',]) +
                   cycler(linestyle=['-', '--',   '-.',       ':',     '-.',       ':']) +
                  cycler(marker=[' ',    ' ',    ' ',       ' ',     'x',       '+'] ))
    else:
        colors = (cycler(color= ['#EE6666', '#3366BB', '#9988DD', '#77AA33', '#FF8888', '#000000', ]) +
                  cycler(linestyle=['-', '--',   '-.',       ':',     '-.',       ':']))
    rc('axes', facecolor='#F6F6F6', edgecolor='none',
       axisbelow=True, grid=True, prop_cycle=colors)
    rc('grid', color='w', linestyle='solid')
    rc('xtick', direction='out', color='gray')
    rc('ytick', direction='out', color='gray')
    rc('patch', edgecolor='#E6E6E6')
    rc('lines', linewidth=linewidth)
    rc('font',**{'family':'serif','serif':['Palatino']})
    #rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    rc('text', usetex=usetex)
    rc('legend', framealpha=0.5)



def plot_with_confidence(ts, means, stddevs, **kwargs):
    """
    plot_with_confidence
    plots an uncertain variable given by a mean and standard deviation over time
    a central line for the mean, and a surrounding shaded area +- 3 std devs
    @arg ts: time stamps (nd vector)
    @arg means: mean values over time (nd vector)
    @arg stddevs: std devs over time (nd vector)
    """
    uppers = means + 3*stddevs
    lowers = means - 3*stddevs
    plt.fill_between(ts, lowers, uppers, alpha=0.5)
    plt.plot(ts, means, **kwargs)


def sample_mean_and_cov(samples):
    """
    sample_mean_and_cov
    computes the sample mean and covariance of a set of samples
    @arg samples: (N, D) tensor where N is number of samples and D is number of dimensions per sample
    @return mu (D) tensor, Sigma (D,D) tensor
    """
    mu = torch.mean(samples, dim=0).view(-1)
    Sigma = torch.zeros((mu.shape[0], mu.shape[0]))
    n = samples.shape[0]
    if len(mu) > 1:
        devs = samples - mu.view(1,-1).repeat(n, 1)
        for i in range(devs.shape[0]):
            Sigma += torch.ger(devs[i], devs[i]) / (n - 1)
    else:
        Sigma = (torch.sum((samples - mu)**2).view(1) / (n - 1)).view(1,1)
    return mu, Sigma



if __name__ == "__main__":
    run_example()
