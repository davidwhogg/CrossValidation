import numpy as np
import matplotlib.pyplot as plt

from XKCDify import xkcd_line, XKCDify, xkcd_fill, xkcd_plot


fig, ax = plt.subplots(1, 2, figsize=(8, 4), facecolor='w',
                       subplot_kw=dict(xticks=[], yticks=[], frameon=False))
fig.subplots_adjust(left=0, right=1,
                    bottom=0, top=1, wspace=0)

theta = np.pi / 4
t = np.linspace(0, 2 * np.pi, 1000)
d = np.vstack([5 * np.sin(t), np.cos(t)])
R = [[np.cos(theta), -np.sin(theta)],
     [np.sin(theta), np.cos(theta)]]

np.random.seed(0)
#------------------------------------------------------------
# First plot: full distribution
d1 = np.dot(R, d)
d2 = 0.5 * d1

xkcd_args = dict(f2=0.01, f3=10, mag=5, xlim=(-7, 7), ylim=(-4, 4))
fill_args = dict(ec='k', fc='gray', lw=2, alpha=0.2)

xkcd_fill(d1[0], d1[1], ax[0], xkcd_args, fill_args)
xkcd_fill(d2[0], d2[1], ax[0], xkcd_args, fill_args)

ax[0].set_title('original')

#------------------------------------------------------------
# Second plot: collapsed distribution
d1 = np.vstack([0.3 * np.sin(t),
                3.0 * np.cos(t)])
d2 = 0.5 * d1
xkcd_args = dict(f2=0.01, mag=1, xlim=(-7, 7), ylim=(-4, 4))
fill_args = dict(ec='k', fc='gray', lw=2, alpha=0.2)

xkcd_fill(d1[0] + 1, d1[1] + 1, ax[1], xkcd_args, fill_args)
xkcd_fill(d2[0] + 1, d2[1] + 1, ax[1], xkcd_args, fill_args)

ax[1].set_title('collapsed')

for ax_i in ax:
    ax_i.set_xlim(-7, 7)
    ax_i.set_ylim(-4, 4)

    ax_i.text(-8.1, 4.2, 'D', ha='right', va='center')
    sup1 = ax_i.text(-7.1, 4.4, 'TR', ha='right', va='center')
    #ax_i.text(-7.2, 4.2, r'$D^{\rm TR}$',
    #          ha='right', va='center', size=16)

    ax_i.text(7.2, -4.4, '0', ha='right', va='center')
    ax_i.text(7.2, -4.4, '-', ha='right', va='center')
    sup2 = ax_i.text(7.7, -4.6, 'M', ha='right', va='center')
    #ax_i.text(7.2, -4.2, r'$\theta_M$',
    #          ha='center', va='top', size=16)

    ax_i.text(-8.3, 1, 'D', ha='right', va='center')
    sup3 = ax_i.text(-7.3, 1.2, 'TR', ha='right', va='center')
    sup4 = ax_i.text(-7.1, 0.8, 'obs', ha='right', va='center')
    #ax_i.text(-7.2, 1, r'$D^{\rm TR}_{\rm obs}$',
    #          ha='right', va='center', size=16)

    ax_i.text(1, -4.6, '0', ha='right', va='center')
    ax_i.text(1, -4.6, '-', ha='right', va='center')
    sup5 = ax_i.text(1.2, -4.8, 'M', ha='left', va='center')
    sup6 = ax_i.text(1.2, -4.4, 'max', ha='left', va='center')
    #ax_i.text(1, -4.2, r'$\hat{\theta}^{\max}_M$',
    #          ha='center', va='top', size=16)

    XKCDify(ax_i)
    for sup in [sup1, sup2, sup3, sup5]:
        sup.set_size(12)

    for sup in [sup4, sup6]:
        sup.set_size(10)

    xkcd_args = dict(f2=0.01, f3=10, mag=1, xlim=(-7, 7), ylim=(-4, 4))
    plot_args = dict(ls='--', c='k', lw=2)

    xkcd_plot([-7, 1], [1, 1], ax_i, xkcd_args, plot_args)
    xkcd_plot([1, 1], [-4, 1], ax_i, xkcd_args, plot_args)

plt.show()
