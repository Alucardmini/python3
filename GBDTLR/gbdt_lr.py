# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2020/6/29'


from __future__ import absolute_import, division, print_function
import numpy as np
import pandas as pd
import tensorflow as tf


print(tf.__version__)

tf.logging.set_verbosity(tf.logging.ERROR)
tf.set_random_seed(123)

dftrain = pd.read_csv('https://storage.googleapis.com/tfbt/titanic_train.csv')
dfeval = pd.read_csv('https://storage.googleapis.com/tfbt/titanic_eval.csv')
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

fc = tf.feature_column
CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck',
                       'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']


def one_hot_cat_column(feature_name, vocab):
    return fc.indicator_column(
        fc.categorical_column_with_vocabulary_list(feature_name, vocab))

feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
    vocabulary = dftrain[feature_name].unique()
    feature_columns.append(one_hot_cat_column(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(fc.numeric_column(feature_name, dtype=tf.float32))

NUM_EXAMPLES = len(y_train)


# Use entire batch since this is such a small dataset.
NUM_EXAMPLES = len(y_train)


def make_input_fn(X, y, n_epochs=None, shuffle=True):
  def input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((X.to_dict(orient='list'), y))
    if shuffle:
      # batchsize = NUM_EXAMPLES 时，batchsize过大会报kernal restarting 错误
      dataset = dataset.shuffle(64)
    # For training, cycle thru dataset as many times as need (n_epochs=None).
    dataset = (dataset
      .repeat(n_epochs)
      .batch(64))
    return dataset
  return input_fn

# Training and evaluation input functions.
train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn = make_input_fn(dfeval, y_eval, shuffle=False, n_epochs=1)

print(len(dftrain))

params = {
  'n_trees': 100,
  'max_depth': 3,
  'n_batches_per_layer': 1,
  # You must enable center_bias = True to get DFCs. This will force the model to
  # make an initial prediction before using any features (e.g. use the mean of
  # the training labels for regression or log odds for classification when
  # using cross entropy loss).
  'center_bias': True
}

est = tf.estimator.BoostedTreesClassifier(feature_columns, **params)
est.train(train_input_fn, max_steps=100)
results = est.evaluate(eval_input_fn)
pd.Series(results).to_frame()

import matplotlib.pyplot as plt
import seaborn as sns
sns_colors = sns.color_palette('colorblind')
pred_dicts = list(est.experimental_predict_with_explanations(eval_input_fn))

print(pred_dicts[0])

labels = y_eval.values
probs = pd.Series([pred['probabilities'][1] for pred in pred_dicts])
df_dfc = pd.DataFrame([pred['dfc'] for pred in pred_dicts])
df_dfc.describe().T

# Sum of DFCs + bias == probabality.
bias = pred_dicts[0]['bias']
dfc_prob = df_dfc.sum(axis=1) + bias
np.testing.assert_almost_equal(dfc_prob.values,
                               probs.values)

# Plot results.
ID = 1
example = df_dfc.iloc[ID]  # Choose ith example from evaluation set.
TOP_N = 8  # View top 8 features.
sorted_ix = example.abs().sort_values()[-TOP_N:].index
ax = example[sorted_ix].plot(kind='barh', color=sns_colors[3])
ax.grid(False, axis='y')

ax.set_title('Feature contributions for example {}\n pred: {:1.2f}; label: {}'.format(ID, probs[ID], labels[ID]))
ax.set_xlabel('Contribution to predicted probability')


# Boilerplate code for plotting :)
def _get_color(value):
    """To make positive DFCs plot green, negative DFCs plot red."""
    green, red = sns.color_palette()[2:4]
    if value >= 0: return green
    return red


def _add_feature_values(feature_values, ax):
    """Display feature's values on left of plot."""
    x_coord = ax.get_xlim()[0]
    OFFSET = 0.15
    for y_coord, (feat_name, feat_val) in enumerate(feature_values.items()):
        t = plt.text(x_coord, y_coord - OFFSET, '{}'.format(feat_val), size=12)
        t.set_bbox(dict(facecolor='white', alpha=0.5))
    from matplotlib.font_manager import FontProperties
    font = FontProperties()
    font.set_weight('bold')
    t = plt.text(x_coord, y_coord + 1 - OFFSET, 'feature\nvalue',
                 fontproperties=font, size=12)


def plot_example(example):
    TOP_N = 8  # View top 8 features.
    sorted_ix = example.abs().sort_values()[-TOP_N:].index  # Sort by magnitude.
    example = example[sorted_ix]
    colors = example.map(_get_color).tolist()
    ax = example.to_frame().plot(kind='barh',
                                 color=[colors],
                                 legend=None,
                                 alpha=0.75,
                                 figsize=(10, 6))
    ax.grid(False, axis='y')
    ax.set_yticklabels(ax.get_yticklabels(), size=14)

    # Add feature values.
    _add_feature_values(dfeval.iloc[ID][sorted_ix], ax)
    return ax


example = df_dfc.iloc[ID]  # Choose IDth example from evaluation set.
ax = plot_example(example)
ax.set_title('Feature contributions for example {}\n pred: {:1.2f}; label: {}'.format(ID, probs[ID], labels[ID]))
ax.set_xlabel('Contribution to predicted probability', size=14)


# Boilerplate plotting code.
def dist_violin_plot(df_dfc, ID):
    # Initialize plot.
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Create example dataframe.
    TOP_N = 8  # View top 8 features.
    example = df_dfc.iloc[ID]
    ix = example.abs().sort_values()[-TOP_N:].index
    example = example[ix]
    example_df = example.to_frame(name='dfc')

    # Add contributions of entire distribution.
    parts = ax.violinplot([df_dfc[w] for w in ix],
                          vert=False,
                          showextrema=False,
                          widths=0.7,
                          positions=np.arange(len(ix)))
    face_color = sns_colors[0]
    alpha = 0.15
    for pc in parts['bodies']:
        pc.set_facecolor(face_color)
        pc.set_alpha(alpha)

    # Add feature values.
    _add_feature_values(dfeval.iloc[ID][sorted_ix], ax)

    # Add local contributions.
    ax.scatter(example,
               np.arange(example.shape[0]),
               color=sns.color_palette()[2],
               s=100,
               marker="s",
               label='contributions for example')

    # Legend
    # Proxy plot, to show violinplot dist on legend.
    ax.plot([0, 0], [1, 1], label='eval set contributions\ndistributions',
            color=face_color, alpha=alpha, linewidth=10)
    legend = ax.legend(loc='lower right', shadow=True, fontsize='x-large',
                       frameon=True)
    legend.get_frame().set_facecolor('white')

    # Format plot.
    ax.set_yticks(np.arange(example.shape[0]))
    ax.set_yticklabels(example.index)
    ax.grid(False, axis='y')
    ax.set_xlabel('Contribution to predicted probability', size=14)

dist_violin_plot(df_dfc, ID)
plt.title('Feature contributions for example {}\n pred: {:1.2f}; label: {}'.format(ID, probs[ID], labels[ID]));

# importances = est.experimental_feature_importances(normalize=True)
# df_imp = pd.Series(importances)
#
# # Visualize importances.
# N = 8
# ax = (df_imp.iloc[0:N][::-1]
#     .plot(kind='barh',
#           color=sns_colors[0],
#           title='Gain feature importances',
#           figsize=(10, 6)))
# ax.grid(False, axis='y')
#
# ## 比较适合连续型特征，离散型特征是沿着y轴的竖线，不太容易看密度，类别型（字符串）不行
# FEATURE = 'age'
# feature = pd.Series(df_dfc[FEATURE].values, index=dfeval[FEATURE].values).sort_index()
# ax = sns.regplot(feature.index.values, feature.values, lowess=True);
# ax.set_ylabel('contribution')
# ax.set_xlabel(FEATURE)
# ax.set_xlim(0, 100)


# def permutation_importances(est, X_eval, y_eval, metric, features):
#     """Column by column, shuffle values and observe effect on eval set.
#
#     source: http://explained.ai/rf-importance/index.html
#     A similar approach can be done during training. See "Drop-column importance"
#     in the above article."""
#     baseline = metric(est, X_eval, y_eval)
#     imp = []
#     for col in features:
#         save = X_eval[col].copy()
#         X_eval[col] = np.random.permutation(X_eval[col])
#         m = metric(est, X_eval, y_eval)
#         X_eval[col] = save
#         imp.append(baseline - m)
#     return np.array(imp)
#
#
# def accuracy_metric(est, X, y):
#     """TensorFlow estimator accuracy."""
#     eval_input_fn = make_input_fn(X,
#                                   y=y,
#                                   shuffle=False,
#                                   n_epochs=1)
#     return est.evaluate(input_fn=eval_input_fn)['accuracy']
#
#
# features = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS
# importances = permutation_importances(est, dfeval, y_eval, accuracy_metric,
#                                       features)
# df_imp = pd.Series(importances, index=features)
#
# sorted_ix = df_imp.abs().sort_values().index
# ax = df_imp[sorted_ix][-5:].plot(kind='barh', color=sns_colors[2], figsize=(10, 6))
# ax.grid(False, axis='y')
# ax.set_title('Permutation feature importance')
#
# from numpy.random import uniform, seed
# from matplotlib.mlab import griddata
#
# # Create fake data
# seed(0)
# npts = 5000
# x = uniform(-2, 2, npts)
# y = uniform(-2, 2, npts)
# z = x*np.exp(-x**2 - y**2)


plt.show()





















