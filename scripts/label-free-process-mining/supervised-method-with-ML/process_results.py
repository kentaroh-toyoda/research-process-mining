import pandas as pd
import re
from sklearn.metrics import f1_score
from plotnine import *
from matplotlib import rc
# enable LaTeX math expression in plots
# ref: https://datascienceworkshops.com/blog/plotnine-grammar-of-graphics-for-python/
rc('text', usetex=True)

# check what fonts are available in matplotlib/plotnine
# ref: https://stackoverflow.com/a/18821968
import matplotlib.font_manager
[f.name for f in matplotlib.font_manager.fontManager.ttflist]
# set font family and size
font_family = 'cmr10'
font_size = 14
# set graph dimentions
height = 6
width = 8

# read a result file
results = pd.read_csv('./results.csv')
# drop the evaluation_datetime column as it is not necessarry
results = results.drop(['evaluation_datetime'], axis=1)
# shorten dataset names (remove directories and extensions)
results['dataset'] = results['dataset'].apply(lambda x: re.findall(r'.*/(.*)\.csv$', x)[0])
# replace _ with -
results['dataset'] = results['dataset'].apply(lambda x: re.sub('_', '-', x))
# calculate F1 score by parameters
parameters = ['dataset', 'n_rows', 'r_train', 'r_other', 'RF_n_estimators']
results = results.groupby(parameters) \
        .apply(lambda x: f1_score(x['class'], x['predicted'], average=None)) \
        .reset_index(name='f1_score')
# note: f1_score() in scikit-learn arranges labels alphabetically
labels=sorted(['case:concept:name', 'concept:name', 'time:timestamp', 'other'])
# expand the f1 score column into columns
# ref: https://stackoverflow.com/a/35491399
results[labels] = pd.DataFrame(results.f1_score.to_list(), index=results.index)
# melt the f1 score columns
results = pd.melt(results, id_vars=parameters, value_vars=labels, \
        var_name='label', value_name='f1_score')

# F1 score by dataset
tmp = results \
    .groupby(['dataset', 'label'])['f1_score'] \
    .describe() \
    .reset_index()
# save it as a LaTeX table
with open('f1_datasets_table.tex', 'w') as t:
    t.write(tmp.to_latex(index=False))

# or plot it
plot = ggplot(tmp, aes(x='dataset', y='f1_score')) \
        + geom_bar(stat='identity', position='dodge', width=0.1) \
        + facet_grid('.~label') \
        + xlab('Dataset') \
        + ylab('$F_1$') \
        + theme(text=element_text(family=font_family, size=font_size))
# save the plot
plot.save('f1_datasets.pdf', height=height, width=width)

# F1 score versus N_estimators in RF
tmp = results[(results['n_rows'] == 100) & (results['r_other'] == 5)] \
        .reset_index(drop=True)

plot = ggplot(tmp) \
        + geom_bar(aes(x='RF_n_estimators', y='f1_score'), \
        stat='identity', position='dodge', width=0.1) \
        + scale_x_log10(breaks=list(set(tmp['RF_n_estimators']))) \
        + facet_wrap('~label') \
        + xlab('$N_{\\mathrm{trees}}$') \
        + ylab('$F_1$') \
        + theme(text=element_text(family=font_family, size=font_size))
# save the plot
plot.save('f1_n_estimators.pdf', height=height, width=width)

# F1 score versus r_other
tmp = results[(results['n_rows'] == 1000) & (results['RF_n_estimators'] == 100)] \
        .reset_index(drop=True)
tmp
plot = ggplot(tmp) \
        + geom_bar(aes(x='r_other', y='f1_score'), \
        stat='identity', position='dodge', width=0.3) \
        + scale_x_continuous(breaks=list(set(tmp['r_other']))) \
        + facet_wrap('~label') \
        + xlab('$r_{\\mathrm{other}}$') \
        + ylab('$F_1$') \
        + theme(text=element_text(family=font_family, size=font_size))
# save the plot
plot.save('f1_r_other.pdf', height=height, width=width)


results['dataset'][0]
