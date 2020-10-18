# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2020/6/30'


from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd
import tensorflow as tf

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 1000)
pd.set_option("display.max_colwidth", 1000)
pd.set_option('display.width', 1000)

tf.logging.set_verbosity(tf.logging.ERROR)
tf.set_random_seed(123)

# Load dataset.
dftrain = pd.read_csv('https://storage.googleapis.com/tfbt/titanic_train.csv')
dfeval = pd.read_csv('https://storage.googleapis.com/tfbt/titanic_eval.csv')
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck',
                       'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

feature_columns = []

for feature_name in CATEGORICAL_COLUMNS:
    vocabulary = dftrain[feature_name].unique()
    feature_columns.append(
        tf.feature_column.indicator_column(
            tf.feature_column.categorical_column_with_vocabulary_list(
                feature_name, vocabulary
            )))

for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))


def make_input_func(x, y, n_epochs=None, shuffle=True):

    def input_func():
        dataset = tf.data.Dataset.from_tensor_slices((x.to_dict(orient='list'), y))

        if shuffle:
            dataset = dataset.shuffle(64)
        dataset = (dataset
                   .repeat(n_epochs)
                   .batch(64))
        return dataset

    return input_func


params = {
    'n_trees': 100,
    'max_depth': 3,
    'n_batches_per_layer': 1,
    'center_bias': True
}

train_input_func = make_input_func(dftrain, y_train)
eval_input_func = make_input_func(dfeval, y_eval, shuffle=False, n_epochs=1)

est = tf.estimator.BoostedTreesRegressor(feature_columns, **params)

est = tf.estimator.BoostedTreesClassifier(feature_columns, **params)
est.train(train_input_func, max_steps=500)
results = est.evaluate(eval_input_func)
print(pd.Series(results).to_frame())

import matplotlib.pyplot as plt
import seaborn as sns
sns.color = sns.color_palette('colorblind')

pred_dicts = list(est.experimental_predict_with_explanations(eval_input_func))

print(pred_dicts[0])

labels = y_eval.values
probs = pd.Series([pred['probabilities'][1] for pred in pred_dicts])
df_dfc = pd.DataFrame([pred['dfc'] for pred in pred_dicts])
print(df_dfc.describe().T)


# Sum of DFCs + bias == probabality.
bias = pred_dicts[0]['bias']
dfc_prob = df_dfc.sum(axis=1) + bias
np.testing.assert_almost_equal(dfc_prob.values,
                               probs.values)


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
  TOP_N = 8 # View top 8 features.
  sorted_ix = example.abs().sort_values()[-TOP_N:].index  # Sort by magnitude.
  example = example[sorted_ix]
  colors = example.map(_get_color).tolist()
  ax = example.to_frame().plot(kind='barh',
                          color=[colors],
                          legend=None,
                          alpha=0.75,
                          figsize=(10,6))
  ax.grid(False, axis='y')
  ax.set_yticklabels(ax.get_yticklabels(), size=14)

  # Add feature values.
  _add_feature_values(dfeval.iloc[ID][sorted_ix], ax)
  return ax


# Plot results.
ID = 182
example = df_dfc.iloc[ID]  # Choose ith example from evaluation set.

TOP_N = 8  # View top 8 features.
sorted_ix = example.abs().sort_values()[-TOP_N:].index

print(sorted_ix)
print(example.abs().sort_values()[-TOP_N:])
ax = plot_example(example)
ax.set_title('Feature contributions for example {}\n pred: {:1.2f}; label: {}'.format(ID, probs[ID], labels[ID]))
ax.set_xlabel('Contribution to predicted probability', size=14)
plt.show()
