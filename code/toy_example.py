import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt


def log_norm(y, mu, sigma):
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

        return optimize.leastsq(self.func, x0, args=(t, y), **kwargs)

    def cross_val(self, t, y, dy, x0, validation_size=1, **kwargs):
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

            theta, ierr = self.fit(t_train, y_train, x0=x0, **kwargs)
            log_cv_score += log_norm(self.eval(theta, t_val), y_val, dy).sum()

        return log_cv_score


class CubicModel(Model):
    n_params = 2

    def eval(self, theta, t):
        return theta[0] * t + theta[1] * (t ** 3)

    def Dfun(self, theta, t, y):
        return [t, t ** 3]


class SineModel(Model):
    n_params = 2

    def eval(self, theta, t):
        return theta[0] * np.sin(theta[1] * t)
    
    def Dfun(self, theta, t, y):
        return [np.sin(theta[1] * t), theta[0] * t * np.cos(theta[1] * t)]


model1 = CubicModel(rseed=5)
model2 = SineModel(rseed=5)
tmin = -2
tmax = 2

t1, y1 = model1.generate(32, tmin, tmax, sigma=0.125)
theta1 = model1.theta
theta11, ierr11 = model1.fit(t1, y1, x0=[1, 0])
theta12, ierr12 = model2.fit(t1, y1, x0=[1, 0])
print model1.cross_val(t1, y1, 0.125, [1, 0])
print model2.cross_val(t1, y1, 0.125, [1, 0])
print

t2, y2 = model2.generate(32, tmin, tmax, sigma=0.125)
theta2 = model2.theta
theta21, ierr21 = model1.fit(t2, y2, x0=[1, 0])
theta22, ierr22 = model2.fit(t2, y2, x0=[1, 0])
print model1.cross_val(t2, y2, 0.125, [1, 0])
print model2.cross_val(t2, y2, 0.125, [1, 0])

t = np.linspace(tmin, tmax, 100)
fig, ax = plt.subplots(2)

ax[0].plot(t, model1(t))
ax[0].errorbar(t1, y1, 0.125, fmt='.k')
ax[0].plot(t, model1(t, theta11))
ax[0].plot(t, model2(t, theta12))

ax[1].plot(t, model2(t))
ax[1].errorbar(t2, y2, 0.125, fmt='.k')
ax[1].plot(t, model1(t, theta21))
ax[1].plot(t, model2(t, theta22))

plt.show()
