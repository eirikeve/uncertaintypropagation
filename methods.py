import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

"""
methods.py
Eirik Vesterkjær, 2019

Implementation of some uncertainty propagation methods
for propagating means and covariances through functions
"""

class UPBase():
    """
    Uncertainty Propagation base class
    Abstract class for propagating (gaussian) uncertainty through nonlinear functions
    """
    def __init__(self):
        self.mu_ = None
        self.Sigma_ = None
        self.mode_ = None

    def apply(self, expression, mu_in, Sigma_in):
        """
        Perform UP on expression, given an input with mean mu and covariance Sigma
        """
        raise NotImplementedError()

    @property
    def mu(self):
        """
        mu: mean of distribution. Set automatically after applying UP
        """
        return self._mu
    @property
    def Sigma(self):
        """
        Sigma: covariante of distribution
        """
        return self._Sigma
    @property
    def mode(self):
        """
        mode of distribution. Not implemented per now
        """
        raise NotImplementedError("Mode computation not implemented")
    @mu.setter
    def mu(self,value):
        self._mu = value
    @Sigma.setter
    def Sigma(self, value):
        self._Sigma = value
    @mode.setter
    def mode(self, value):
        self._mode = value

    def sample_distribution(self, count_samples, mu = None, Sigma = None):
        if mu is None:
            mu = self.mu
        if Sigma is None:
            Sigma = self.Sigma
        mu = mu.view(-1)
        Sigma = Sigma.view(mu.shape[0], -1)
        samples = torch.empty((count_samples, mu.shape[0]), dtype=torch.float)
        MVN = torch.distributions.multivariate_normal.MultivariateNormal(mu, Sigma)
        for i in range(samples.shape[0]):
            samples[i] = MVN.sample()
        return samples.squeeze()

class UnscentedTf(UPBase):
    """
    Unscented transform with base set sigma points
    https://www.spiedigitallibrary.org/conference-proceedings-of-spie/3068/0000/New-extension-of-the-Kalman-filter-to-nonlinear-systems/10.1117/12.280797.full?SSO=1
    https://nbviewer.jupyter.org/github/sbitzer/UKF-exposed/blob/master/UKF.ipynb
    """
    def __init__(self, kappa = 0.0):
        UPBase.__init__(self)
        """
        n: input state dimension
        count_sigma_pts:
        """
        self.n = None
        self.k = kappa

    @property
    def n(self):
        return self._n
    @n.setter
    def n(self, value):
        self._n = value

    @property
    def k(self):
        return self._k
    @k.setter
    def k(self, value):
        self._k = value
    @property
    def count_sigma_pts(self):
        return self._n * 2 + 1

    def apply(self, expression, mu_in, Sigma_in):
        self.n = mu_in.shape[0]
        self.init_weights()
        X = self.get_sigma_points(mu_in, Sigma_in)
        Y =  expression(X)
        mu_out, Sigma_out = self.weighted_estimate(Y)

        self.mu = mu_out
        self.Sigma = Sigma_out
        return self.mu, self.Sigma

    def init_weights(self):
        if self.k <= -self.n:
            raise ValueError("Kappa was set to {self.k} but it must be greater than {-self.n}")
        self.W_mu = torch.ones((self.count_sigma_pts))
        self.W_mu  /= 2*(self.n + self.k)
        self.W_mu[0] = self.k / (self.n + self.k)
        self.W_cov = self.W_mu

    def get_sigma_points(self, mu, Sigma):
        X = mu.view(1,-1).repeat(self.count_sigma_pts,1)
        L = torch.tensor(np.sqrt(self.n + self.k)*np.linalg.cholesky(Sigma))
        X[1:self.n+1, :] = X[1:self.n+1, :] + L
        X[self.n+1:,:]   = X[self.n+1:,:]   - L
        return X

    def weighted_estimate(self, Y):
        #mu_out = torch.sum(Y*self.W_mu.view(-1,1).repeat(1,self.n), dim=0)

        mu_out = torch.matmul(torch.t(Y), self.W_mu)
        devs = Y - mu_out.view(1,-1).repeat(self.count_sigma_pts,1)

        Sigma_out = torch.zeros((Y.shape[1], Y.shape[1]))

        for i in range(self.count_sigma_pts):
            Sigma_out += torch.ger( devs[i,:]*self.W_cov[i], devs[i,:] )
        return mu_out, Sigma_out




class ApproximationTransform(UPBase):
    def __init__(self, order):
        UPBase.__init__(self)
        self.order = order
        self.terms = None
    @property
    def order(self):
        return self._order
    @order.setter
    def order(self, value):
        if value not in [1, 2]:
            raise ValueError(f"Order must be: 1 or 2")
        self._order = value
    @property
    def terms(self):
        return self._terms
    @terms.setter
    def terms(self, values):
        if values is not None and len(values) != self.order + 1:
            raise ValueError("Must have {self.order+1} terms for approx of order {self.order}")
        self._terms = values
    def apply(self, expression, mu_in, Sigma_in):
        quad = True if self.order == 2 else False
        y, J, H = ApproximationTransform.local_approximation(expression, mu_in, quad)

        # if linear, H will be None (and won't affect later computations)
        self.terms = [y, J, H] if quad else [y, J]
        self.Sigma = ApproximationTransform.propagate_covariance(mu_in, Sigma_in, mu_in, J, H)
        self.mu = y + ApproximationTransform.output_mean_hessian_update(Sigma_in, H)
        return self.mu, self.Sigma

    @staticmethod
    def local_approximation(expression: nn.Module, x_eval, quadratic=True, output_mask=None):
        x_eval = torch.autograd.Variable(x_eval, requires_grad=True)
        y = expression(x_eval).view(-1)

        if output_mask is not None:
            # If not all outputs are relevant wrt. the covariance
            y = torch.matmul(output_mask, y)
        J = torch.autograd.Variable(torch.zeros(y.shape[0],
                                            x_eval.shape[0]),
                                            requires_grad=True)
        J.retain_grad()

        # Loop over each output, compute the gradient of that wrt. the input
        for y_i in range( y.shape[0] ):
            gradient_mask = torch.zeros(y.shape)
            # This just sets which output we compute the grad wrt. to
            gradient_mask[y_i] = 1
            j =  torch.autograd.grad(  y,
                                       x_eval,
                                       grad_outputs=gradient_mask,
                                       allow_unused=True,
                                       retain_graph=True,
                                       create_graph=True if quadratic else False)[0]
            if j is not None:
                j.retain_grad()
                J[y_i,:] = j
                J.retain_grad()


        if not quadratic:
            return y, J, None




        # quadratic approx
        H = torch.zeros(y.shape[0], x_eval.shape[0], x_eval.shape[0])
        # Compute Hessian --> Jacobian of the rows of the Jacobian wrt. inputs
        for row in range(J.shape[0]):
            #grad_y_i = torch.autograd.Variable(J[row,:], requires_grad=True)
            for j in range(y.shape[0]):
                gradient_mask = torch.zeros(y.shape)
                gradient_mask[j] = 1
                h = None
                # last pass through; don't store the graph any more.
                if row + 1== J.shape[0] and j + 1 == y.shape[0]:
                    h = torch.autograd.grad(    J[row,:],
                                                x_eval,
                                                grad_outputs=gradient_mask,
                                                allow_unused=True,
                                                retain_graph=False,
                                                create_graph=False)[0]

                else:
                    h = torch.autograd.grad(    J[row,:],
                                                x_eval,
                                                grad_outputs=gradient_mask,
                                                allow_unused=True,
                                                retain_graph=True,
                                                create_graph=False)[0]
                if h is not None:
                    H[row,j,:] = h
        return y, J, H

    @staticmethod
    def make_Sigma_y_jh(J, H, mu_x, Sigma_x):

        m = J.shape[0]
        n = J.shape[1]
        Sigma_y_jh = torch.zeros((m,m))
        # RIP runtime for dimensionalities
        for r in range(m): # row
            for c in range(m): # col
                for i in range(n):
                    for j in range(n):
                        for k in range(n):
                            Sigma_y_jh[r,c] += J[r,i]*H[c,j,k]*(
                                Sigma_x[i,j]*mu_x[k] + Sigma_x[i,k]*mu_x[j]
                            )
        return Sigma_y_jh
    @staticmethod
    def make_Sigma_y_hh(J, H, mu_x, Sigma_x):
        m = J.shape[0]
        n = J.shape[1]
        Sigma_y_hh = torch.zeros((m,m))
        # RIP runtime for dimensionalities
        for r in range(m): # row
            for c in range(m): # col
                for i in range(n):
                    for j in range(n):
                        for k in range(n):
                            for l in range(n):
                                Sigma_y_hh[r,c] += H[r,i,j]*H[c,j,k]*(
                                    Sigma_x[i,k]*Sigma_x[j,l] +
                                    Sigma_x[i,l]*Sigma_x[j,k] +
                                    Sigma_x[i,k]*mu_x[j]*mu_x[l] +
                                    Sigma_x[i,l]*mu_x[j]*mu_x[k] +
                                    Sigma_x[j,k]*mu_x[i]*mu_x[l] +
                                    Sigma_x[j,l]*mu_x[i]*mu_x[k]
                                )
        return Sigma_y_hh

    @staticmethod
    def propagate_covariance(x, Sigma_x, x_eval, J, H=None):
        eps = 1e-10
        mu_x = x - x_eval
        Sigma_y_jj = torch.matmul(torch.matmul(J, Sigma_x), torch.t(J))
        if H is None:
            return Sigma_y_jj
        Sigma_y_hh = ApproximationTransform.make_Sigma_y_hh(J, H, mu_x, Sigma_x)
        # save time since Sigma_y_jk will be zero if x = x_eval
        if torch.sum(torch.abs(x - x_eval)) < eps:
            Sigma_y_jh = ApproximationTransform.make_Sigma_y_jh(J, H, mu_x, Sigma_x)
            return Sigma_y_jj +  0.5*Sigma_y_jh + 0.5*torch.t(Sigma_y_jh) + 0.25*Sigma_y_hh
        return Sigma_y_jj + 0.25*Sigma_y_hh

    @staticmethod
    def output_mean_hessian_update(Sigma_x, H=None):
        update_term = torch.zeros(Sigma_x.shape[0])
        if H is None:
            return update_term
        for i in range(H.shape[0]):
            update_term[i] = 0.5*torch.sum(Sigma_x*H)
            return update_term


# These have functions that are overridden
# such that we get __qualname__ that is distinguishable for those functions.
# making it easier to print info about them
class LinearApprox(ApproximationTransform):
    def __init__(self):
        ApproximationTransform.__init__(self, 1)
    def apply(self, expression, mu_in, Sigma_in):
        return super().apply(expression, mu_in, Sigma_in)
class QuadraticApprox(ApproximationTransform):
    def __init__(self):
        ApproximationTransform.__init__(self, 2)
    def apply(self, expression, mu_in, Sigma_in):
        return super().apply(expression, mu_in, Sigma_in)


class MonteCarlo(UPBase):
    def __init__(self, count_samples=1000):
        UPBase.__init__(self)
        self.count_samples = count_samples
    @property
    def count_samples(self):
        return self._count_samples
    @count_samples.setter
    def count_samples(self, value):
        self._count_samples = value

    def apply(self, expression, mu_in, Sigma_in):
        self.expression = expression
        self.mu_in = mu_in
        self.Sigma_in = Sigma_in
        input_distribution = self.sample_distribution(self.count_samples, mu_in, Sigma_in)
        output_distribution = expression(input_distribution)
        self.mu = (torch.mean(output_distribution, dim=0)).view(-1)
        self.Sigma = torch.zeros((self.mu.shape[0], self.mu.shape[0]))
        if len(self.mu) > 1:
            devs = output_distribution - self.mu.view(1,-1).repeat(output_distribution.shape[0], 1)
            for i in range(devs.shape[0]):
                self.Sigma += torch.ger(devs[i], devs[i]) / (output_distribution.shape[0] - 1)
        else:
            self.Sigma = (torch.sum((output_distribution - self.mu)**2).view(1) / (output_distribution.shape[0] - 1)).view(1,1)
        return self.mu, self.Sigma
    def sample_true_posterior(self, expression, mu_in, Sigma_in, count_samples):
        input_distribution = self.sample_distribution(count_samples, mu_in, Sigma_in)
        output_distribution = expression(input_distribution)
        return output_distribution
