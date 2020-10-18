# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2020/9/13 2:36 PM'


from __future__ import absolute_import, division, print_function
import numpy as np
import pandas as pd
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)
tf.set_random_seed(123)

# csv_path='/Users/wuxikun/Documents/Book/好书精读/parallel_ml_tutorial-master/notebooks/titanic_train.csv'
# csv_path='https://storage.googleapis.com/tfbt/titanic_train.csv'
csv_path='/Users/wuxikun/Downloads/titanic/train.csv'
df_train = pd.read_csv(csv_path)
df_eval = pd.read_csv(csv_path)

print(df_train.head())

y_train = df_train.pop("Survived")
y_eval = df_eval.pop("Survived")

fc = tf.feature_column
categorical_cols = ['Sex', 'SibSp', 'Parch', 'Pclass',
                       'Embarked']
numeric_cols = ['Age', 'Fare']


def one_hot_cat_column(feature_name, vocab):
    return tf.feature_column.indicator_column(
        tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocab)
    )

feature_columns = []

for feature_name in categorical_cols:
    vocabulary = df_train[feature_name].unique()
    feature_columns.append(one_hot_cat_column(feature_name, vocabulary))

for feature_name in numeric_cols:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))


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

train_input_fn = make_input_fn(df_train, y_train)
eval_input_fn = make_input_fn(df_train, y_eval, shuffle=False, n_epochs=1)

print(len(df_train))


params = {
    'n_trees':100,
    'max_depth': 3,
    'n_batches_per_layer': 1,
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

