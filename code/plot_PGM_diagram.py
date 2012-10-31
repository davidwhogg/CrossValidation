import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 6), facecolor='w',
                       subplot_kw=dict(frameon=False, xticks=[], yticks=[],
                                       xlim=(0, 10), ylim=(0, 10)))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

circ_locs = [(5, 8), (5, 4.5), (2, 2), (8, 2)]
labels = ['$M$', r'$\theta_M$', '$D^{TR}$', '$D^{CV}$']
arrows = [(0, 1), (1, 2), (1, 3)]
radius = 1

for loc, label in zip(circ_locs, labels):
    ax.add_patch(plt.Circle(loc, radius, fc='w', ec='k', lw=2))
    ax.text(loc[0], loc[1], label, ha='center', va='center', size=20)

for arrow in arrows:
    ax.annotate("", circ_locs[arrow[1]], circ_locs[arrow[0]],
                arrowprops=dict(arrowstyle='->', lw=2,
                                shrinkA=-50 * radius, shrinkB=45 * radius),
                ha='center', va='center')


plt.show()
