"""
Simple comparison between CV and Bayesian model selection
"""
import numpy as np
import matplotlib.pyplot as plt

def logsumexp(arr, axis=None):
    """Computes the sum of arr assuming arr is in the log domain.

    Returns log(sum(exp(arr))) while minimizing the possibility of
    over/underflow.

    Examples
    --------

    >>> import numpy as np
    >>> a = np.arange(10)
    >>> np.log(np.sum(np.exp(a)))
    9.4586297444267107
    >>> logsumexp(a)
    9.4586297444267107
    """
    # if axis is specified, roll axis to 0 so that broadcasting works below
    if axis is not None:
        arr = np.rollaxis(arr, axis)
        axis = 0

    # Use the max to normalize, as with the log this is what accumulates
    # the fewest errors
    vmax = arr.max(axis=axis)
    out = np.log(np.sum(np.exp(arr - vmax), axis=axis))
    out += vmax

    return out


def log_norm(y, mu, sigma):
    return (-0.5 * np.log(2 * np.pi * sigma ** 2)
            -0.5 * ((y - mu) / sigma) ** 2)


def theta_max(x, y, deg):
    return np.polyfit(x, y, deg)


def log_likelihood(x, y, theta, allow_broadcast=True):
    if allow_broadcast:
        x = np.asarray(x)
        y = np.asarray(y)
        assert x.ndim == 1
        assert x.shape == y.shape
        theta = np.asarray(theta)
        thetashape = theta.shape
        theta = theta.reshape((-1, theta.shape[-1]))[:, :, np.newaxis]
        power = np.arange(theta.shape[-2], dtype=float)[::-1][:, np.newaxis]
        
        mu = theta * x ** power
        mu = mu.sum(-2)
        mu = mu.reshape(thetashape[:-1] + x.shape)
    else:
        mu = np.polyval(theta, x)

    return log_norm(y, mu, 1).sum(-1)


def model_selection_crossval(x, y, Ntrain):
    i = np.arange(len(x))
    np.random.shuffle(i)
    x_train = x[i[:Ntrain]]
    y_train = y[i[:Ntrain]]
    x_cv = x[i[Ntrain:]]
    y_cv = y[i[Ntrain:]]

    theta_train1 = theta_max(x_train, y_train, 0)
    logL_cv1 = log_likelihood(x_cv, y_cv, theta_train1).sum()

    theta_train2 = theta_max(x_train, y_train, 1)
    logL_cv2 = log_likelihood(x_cv, y_cv, theta_train2).sum()

    return logL_cv1, logL_cv2


def model_selection_bayes(x, y, theta_min, theta_max, Nsamp=100):
    theta0 = np.linspace(theta_min[0], theta_max[0], Nsamp)
    theta1 = np.linspace(theta_min[1], theta_max[1], Nsamp)

    dtheta0 = theta0[1] - theta0[0]
    dtheta1 = theta1[1] - theta1[0]

    theta_model1 = theta1[:, None]

    theta0_grid, theta1_grid = np.meshgrid(theta0, theta1)
    theta_model2 = np.vstack((theta0_grid.ravel(),
                              theta1_grid.ravel())).T

    # 1D model
    prior = 1. / (Nsamp * dtheta1)
    logL_grid1 = log_likelihood(x, y, theta_model1)
    posterior1 = logsumexp(logL_grid1 * prior) + np.log(dtheta1)

    # 2D model
    prior = 1. / (Nsamp * dtheta0 * Nsamp * dtheta1)
    logL_grid2 = log_likelihood(x, y, theta_model2)
    logL_grid2 = logL_grid2.reshape((Nsamp, Nsamp))
    posterior2 = logsumexp(logL_grid2 * prior) + np.log(dtheta0 * dtheta1)

    fig, ax = plt.subplots(2)
    
    ax[0].plot(theta1, np.exp(logL_grid1))
    ax[1].contour(theta1, theta0, np.exp(logL_grid2.T))
    
    return posterior1, posterior2


# data: simple line with noise
np.random.seed(0)
N = 100
Ntrain = 90
model = [0.2, 1]
dy = 1

x = np.random.normal(5, 2, 100)
y = model[0] * x + np.random.normal(model[1], dy, N)

print model_selection_crossval(x, y, Ntrain)
print model_selection_bayes(x, y, [0, 0], [1, 3])

print np.log(1000)

fig, ax = plt.subplots()
ax.scatter(x, y)
plt.show()

