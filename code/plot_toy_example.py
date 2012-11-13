import numpy as np
import matplotlib.pyplot as plt
from model import Model, logsumexp


class LineModel(Model):
    name = 'line'
    n_params = 1

    def eval(self, theta, t):
        return theta[0] * t

    def Dfun(self, theta, t, y):
        return [t]

    def fit(self, t, y, *args, **kwargs):
        return [np.dot(t, y) / np.dot(t, t)]


class SineModel(Model):
    name = 'sine'
    n_params = 1
    const = np.pi / 2

    def eval(self, theta, t):
        return theta[0] * np.sin(t) * self.const

    def Dfun(self, theta, t, y):
        return [np.sin(t) * self.const]

    def fit(self, t, y, *args, **kwargs):
        return [np.dot(np.sin(t), y) / np.dot(np.sin(t), np.sin(t))
                / self.const]


def create_plots(rseed=3, tmin=-2, tmax=2, sigma=1.0, N=64):
    """Create the plots for the document"""
    model1 = LineModel(rseed=rseed)
    model2 = SineModel(rseed=rseed)

    print "parameters:", model1.theta, model2.theta

    t1, y1 = model1.generate(N, tmin, tmax, sigma=sigma)
    t2, y2 = model2.generate(N, tmin, tmax, sigma=sigma)

    t12 = model1.fit(t2, y2)
    t21 = model2.fit(t1, y1)

    #------------------------------------------------------------
    # first: plot points drawn from the model
    fig, ax = plt.subplots(2, sharex=True)

    ax[0].errorbar(t1, y1, sigma, fmt='bo', ecolor='#AAAAFF')
    ax[1].errorbar(t2, y2, sigma, fmt='rs', ecolor='#FFAAAA')

    trange = np.linspace(tmin, tmax, 100)
    ax[0].plot(trange, model1.eval(model1.theta, trange), '-b')
    ax[1].plot(trange, model2.eval(model2.theta, trange), '-r')

    ax[0].plot(trange, model2.eval(t21, trange), '--r')
    ax[1].plot(trange, model1.eval(t12, trange), '--b')

    ax[0].set_title(r"Model 1: $y(t) = a_1 t$")
    ax[1].set_title(r"Model 2: $y(t) = a_2 \frac{\pi}{2}\sin(t)$")

    ax[1].set_xlabel('t')
    ax[0].set_ylabel('y')
    ax[1].set_ylabel('y')

    #------------------------------------------------------------
    # second: plot to describe the delta-func approximation
    fig, ax = plt.subplots(2, sharex=True)

    (t, y) = (t1, y1)
    theta = np.linspace(-5, 5, 1000)
    theta_arg = [theta[:, None]]
    dtheta = theta[1] - theta[0]

    for model, axi in zip((model1, model2), ax):
        log_P_Sk = np.array([model.log_likelihood(t[i:i + 1],
                                                  y[i:i + 1],
                                                  sigma, theta_arg)
                             for i in range(len(t))])
        log_P_Tk = np.array([model.log_likelihood(np.concatenate([t[:i],
                                                                  t[i + 1:]]),
                                                  np.concatenate([y[:i],
                                                                  y[i + 1:]]),
                                                  sigma, theta_arg)
                             for i in range(len(t))])
        log_prior = model.log_prior(theta_arg)

        log_P_Tk = log_P_Tk + log_prior
        log_P_Tk -= logsumexp(log_P_Tk, 1)[:, None]
        log_P_Tk -= np.log(dtheta)

        axi.plot(theta, np.exp(log_P_Tk[0]), '-b',
                 label=r'$P(\theta|T_k, M, I)$')
        axi.plot(theta, np.exp(log_P_Sk[0]), '--g',
                 label=r'$P(S_k|\theta, M, I)$')
        axi.legend(loc=2)

    ax[0].set_title('Model 1 (line)')
    ax[1].set_title('Model 2 (sine)')

    ax[0].set_xlabel('$a_1$')
    ax[1].set_xlabel('$a_2$')
    ax[0].set_ylabel('y')
    ax[1].set_ylabel('y')
    
    ax[0].set_xlim(-5, 5)

    #------------------------------------------------------------
    # third: plot to describe the bracket approximation
    fig, ax = plt.subplots(2, subplot_kw=dict(yscale='log'))

    (t, y) = (t1, y1)
    
    for model in [model1, model2]:
        logPTk = np.array([model.bayes_integral(np.concatenate([t[:i],
                                                                t[i + 1:]]),
                                                np.concatenate([y[:i],
                                                                y[i + 1:]]),
                                                sigma, -5, 5, 1000)
                           for i in range(len(t) + 1)])

        K = len(t)
        PTk_plot = logPTk[:-1]
        mean_plot = np.ones(K) * logPTk[-1] * (K - 1) / K

        l = ax[0].plot(np.arange(K), np.exp(PTk_plot))
        ax[0].plot(np.arange(K), np.exp(mean_plot),
                   ls=':', c=l[0].get_color())

        ax[1].plot(np.arange(K),
                   np.exp(PTk_plot.cumsum() - mean_plot.cumsum()),
                   c=l[0].get_color())

    ax[1].plot(np.arange(K), np.ones(K), ':k')
    
    ax[0].set_xlabel('k')
    ax[1].set_xlabel('k')

    ax[0].legend(ax[0].lines[:2],
                 (r'$P(T_k|M,I)$',
                  r'$P(D|M,I)^{(K-1)/K}$'))
    ax[1].legend(ax[1].lines[:1],
                 (r'$\frac{1}{P(D|M,I)^{(K-1)k/K}}\prod_{i=1}^k P(T_i|M,I)$',),
                 loc=3)

    ax[0].set_xlim(0, K-1)
    ax[1].set_xlim(0, K-1)

    #------------------------------------------------------------
    # Print the L_CV and Bayes Integral scores
    LCV_11 = model1.cross_val_score(t1, y1, sigma)
    LCV_21 = model2.cross_val_score(t1, y1, sigma)

    BI_11 = model1.bayes_integral(t1, y1, sigma, -5, 5, 1000)
    BI_21 = model2.bayes_integral(t1, y1, sigma, -5, 5, 1000)
    
    print "L_CV:"
    print " %.2f %.2f (%.2f)\n" % (LCV_11, LCV_21, LCV_11 - LCV_21)
    print "L_Bayes:"
    print " %.2f %.2f (%.2f)\n" % (BI_11, BI_21, BI_11 - BI_21)


def print_basic(rseed=3, tmin=-2, tmax=2, sigma=1.0, N=32):
    model1 = LineModel(rseed=rseed)
    model2 = SineModel(rseed=rseed)

    print model1.theta
    print model2.theta

    t1, y1 = model1.generate(N, tmin, tmax, sigma=sigma)
    t2, y2 = model2.generate(N, tmin, tmax, sigma=sigma)

    LCV_11 = model1.cross_val_score(t1, y1, sigma)
    LCV_21 = model2.cross_val_score(t1, y1, sigma)

    LCV_12 = model1.cross_val_score(t2, y2, sigma)
    LCV_22 = model2.cross_val_score(t2, y2, sigma)

    BI_11 = model1.bayes_integral(t1, y1, sigma, -5, 5, 1000)
    BI_12 = model1.bayes_integral(t2, y2, sigma, -5, 5, 1000)

    BI_21 = model2.bayes_integral(t1, y1, sigma, -5, 5, 1000)
    BI_22 = model2.bayes_integral(t2, y2, sigma, -5, 5, 1000)

    print "model 1 data:"
    print "  L_CV  = %.2f   %.2f  (delta = %.2f)" % (LCV_11, LCV_21,
                                                     LCV_21 - LCV_11)
                                    
    print "  Bayes = %.2f   %.2f  (delta = %.2f)" % (BI_11, BI_21,
                                                     BI_21 - BI_11)
    print

    print "model 2 data:"
    print "  L_CV  = %.2f   %.2f  (delta = %.2f)" % (LCV_12, LCV_22,
                                                     LCV_22 - LCV_12)
                                    
    print "  Bayes = %.2f   %.2f  (delta = %.2f)" % (BI_12, BI_22,
                                                     BI_22 - BI_12)
    print


print_basic()
create_plots()
plt.show()
