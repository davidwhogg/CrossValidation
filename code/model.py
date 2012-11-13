import numpy as np
from scipy import optimize


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
    """Return the log of the norm

    This follows all broadcasting rules.

    Example
    -------
    >>> log_norm(1, 0, 1)
    -1.4189385332046727
    >>> np.log(norm(0, 1).pdf(1))
    -1.4189385332046727
    """
    return (-0.5 * np.log(2 * np.pi * sigma ** 2)
            -0.5 * ((y - mu) / sigma) ** 2)


class Model(object):
    def __init__(self, mu_params=0, sigma_params=1, rseed=3):
        if not hasattr(self, 'n_params'):
            raise ValueError('Model must be subclassed and have '
                             'an n_params static attribute.')
        self.mu_params = mu_params
        self.sigma_params = sigma_params
        self.random = np.random.RandomState(rseed)
        self.theta = self.random.normal(mu_params, sigma_params, self.n_params)

    def __call__(self, t, theta=None):
        if theta is None:
            theta = self.theta
        return self.eval(theta, t)
    
    def eval(self, theta, t):
        raise NotImplementedError

    def func(self, theta, t, y):
        return self.eval(theta, t) - y

    def generate(self, N, tmin=-1, tmax=1, sigma=1):
        t = tmin + (tmax - tmin) * self.random.rand(N)
        dy = self.random.normal(0, sigma, N)
        return t, self.eval(self.theta, t) + dy

    def fit(self, t, y, x0=None, **kwargs):
        if hasattr(self, 'Dfun'):
            kwargs['Dfun'] = self.Dfun
            kwargs['col_deriv'] = 1
        else:
            kwargs['Dfun'] = None

        if x0 is None:
            x0 = np.ones(self.n_params)

        return optimize.leastsq(self.func, x0, args=(t, y), **kwargs)[0]

    def log_likelihood(self, t, y, sigma_y, theta):
        return log_norm(self.eval(theta, t), y, sigma_y).sum(-1)

    def log_prior(self, theta_arr):
        theta_arr = np.array(theta_arr + [None])[:-1]

        return np.sum([log_norm(t.reshape(t.shape[:-1]), mu, sigma)
                       for (t, mu, sigma)
                       in np.broadcast(theta_arr, self.mu_params,
                                       self.sigma_params)], 0)

    def cross_val_score(self, t, y, dy, x0=None, validation_size=1, **kwargs):
        assert validation_size >= 1
        t = np.asarray(t)
        y = np.asarray(y)
        log_cv_score = 0
        for i in range(0, len(t), validation_size):
            t_val = t[i:i+validation_size]
            y_val = y[i:i+validation_size]

            if len(t_val) == 0:
                continue

            t_train = np.concatenate([t[:i], t[i + validation_size:]])
            y_train = np.concatenate([y[:i], y[i + validation_size:]])

            theta = self.fit(t_train, y_train, x0=x0, **kwargs)
            log_cv_score += log_norm(self.eval(theta, t_val), y_val, dy).sum()

        return log_cv_score

    def log_likelihood_grid(self, t, y, sigma_y,
                            theta0_range, theta1_range, prior=True):
        assert self.n_params == 2
        theta = [theta0_range.reshape((-1, 1, 1)),
                 theta1_range.reshape((1, -1, 1))]

        logL = self.log_likelihood(t, y, sigma_y, theta)

        if prior:
            log_prior = np.sum([log_norm(t[:, :, 0], mu, sigma)
                                for (t, mu, sigma)
                                in np.broadcast(theta, self.mu_params,
                                                self.sigma_params)])
        else:
            log_prior = 0

        return logL + log_prior

    def bayes_integral(self, t, y, sigma_y,
                       theta_min, theta_max, Ntheta,
                       prior=True):
        dthetas = [(tmax - tmin) * 1. / N
                   for (tmin, tmax, N)
                   in np.broadcast(theta_min, theta_max, Ntheta)]
        slices = [slice(tmin, tmax, dtheta)
                  for (tmin, tmax, dtheta)
                  in np.broadcast(theta_min, theta_max, dthetas)]
        slices = tuple(slices + [slice(0,1)])

        thetas = np.ogrid[slices]
        logL = self.log_likelihood(t, y, sigma_y, thetas)

        if prior:
            return (logsumexp(logL + self.log_prior(thetas[:-1]))
                    + np.sum(np.log(dthetas)))
        else:
            return logsumexp(logL) + np.sum(np.log(dthetas))
