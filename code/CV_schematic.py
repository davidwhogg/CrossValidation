import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from XKCDify import XKCDify


x = np.linspace(0, 10, 1000)

y_train = norm.pdf(x, 5.4, 0.3)
y_val = norm.pdf(x, 4.8, 2)
prior = norm.pdf(x, 3, 5)

prior /= prior.sum()
y_train /= y_train.sum()
y_val /= y_val.sum()

fig, ax = plt.subplots()
ax.plot(x, y_train, label='Training Sample')
ax.plot(x, y_val, label='Validation Sample')
ax.plot(x, prior, label='Prior')
ax.legend()
ax.set_xlabel('x')
ax.set_ylabel('scaled Likelihood / Probability')

XKCDify(ax, mag=0.5, expand_axes=True, ylabel_rotation=90)

plt.show()
