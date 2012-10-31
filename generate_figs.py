import sys
import os
import gc

base_dir = os.path.split(os.path.abspath(__file__))[0]
code_dir = os.path.join(base_dir, 'code')
fig_dir = os.path.join(base_dir, 'documents/figs/')

# For relative imports, we need this
sys.path.append(code_dir)

if not os.path.exists(code_dir):
    raise ValueError('code dir does not exist')

if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

import matplotlib
matplotlib.use('Agg') #don't display plots
import matplotlib.pyplot as plt

plot_scripts = [f for f in os.listdir(code_dir)
                if f.startswith('plot') and f.endswith('.py')]

os.chdir(code_dir)

for script in plot_scripts:
    fig_fmt = os.path.join(fig_dir,
                           os.path.splitext(script)[0] + '_%i.pdf')

    if os.path.exists(fig_fmt % 1):
        output_modtime = os.stat(fig_fmt % 1).st_mtime
        source_modtime = os.stat(script).st_mtime

        if output_modtime >= source_modtime:
            print "skipping script %s: no modifications" % script
            continue

    print "running script %s" % script
    print "  saving to %s" % fig_dir
    
    plt.close('all')
    execfile(os.path.basename(script), {'pl' : plt,
                                        'plt' : plt,
                                        'pylab' : plt})
    
    fig_mgr_list = matplotlib._pylab_helpers.Gcf.get_all_fig_managers()
    figlist = [manager.canvas.figure for manager in fig_mgr_list]
    figlist = sorted(figlist, key = lambda fig: fig.number)
    for fig in figlist:
        fig.savefig(fig_fmt % fig.number)

    plt.close('all')
    gc.collect()
