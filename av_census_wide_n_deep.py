# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Distributed training and evaluation of a wide and deep model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile
import os
import json

from six.moves import urllib

from tensorflow.contrib.learn.python.learn import learn_runner

import pandas as pd
import tensorflow as tf

# Setup logging level of tensorflow
# DEBUG, INFO, WARN, ERROR, FATAL
tf.logging.set_verbosity(tf.logging.INFO)

# Constants: Data download URLs
TRAIN_DATA_URL = "http://mlr.cs.umass.edu/ml/machine-learning-databases/adult/adult.data"
TEST_DATA_URL = "http://mlr.cs.umass.edu/ml/machine-learning-databases/adult/adult.test"


COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num",
           "marital_status", "occupation", "relationship", "race", "gender",
           "capital_gain", "capital_loss", "hours_per_week", "native_country",
           "income_bracket"]
LABEL_COLUMN = "label"
CATEGORICAL_COLUMNS = ["workclass", "education", "marital_status", "occupation",
                       "relationship", "race", "gender", "native_country"]
CONTINUOUS_COLUMNS = ["age", "education_num", "capital_gain", "capital_loss",
                      "hours_per_week"]


def maybe_download_train(data_dir):
    """Maybe downloads test data and returns test file names."""
    train_file_name = os.path.join(data_dir, "adult.data")
    if not os.path.isfile(train_file_name):
        urllib.request.urlretrieve(TRAIN_DATA_URL, train_file_name)

    return train_file_name


def maybe_download_test(data_dir):
    """Maybe downloads test data and returns test file names."""
    test_file_name = os.path.join(data_dir, "adult.test")
    if not os.path.isfile(test_file_name):
        urllib.request.urlretrieve(TEST_DATA_URL, test_file_name)
    return test_file_name


def build_estimator(model_dir, model_type, run_config):
    """Build an estimator."""
    # Sparse base columns.
    gender = tf.contrib.layers.sparse_column_with_keys(column_name="gender",
                                                       keys=["female", "male"])
    education = tf.contrib.layers.sparse_column_with_hash_bucket(
        "education", hash_bucket_size=1000)
    relationship = tf.contrib.layers.sparse_column_with_hash_bucket(
        "relationship", hash_bucket_size=100)
    workclass = tf.contrib.layers.sparse_column_with_hash_bucket(
        "workclass", hash_bucket_size=100)
    occupation = tf.contrib.layers.sparse_column_with_hash_bucket(
        "occupation", hash_bucket_size=1000)
    native_country = tf.contrib.layers.sparse_column_with_hash_bucket(
        "native_country", hash_bucket_size=1000)

    # Continuous base columns.
    age = tf.contrib.layers.real_valued_column("age")
    education_num = tf.contrib.layers.real_valued_column("education_num")
    capital_gain = tf.contrib.layers.real_valued_column("capital_gain")
    capital_loss = tf.contrib.layers.real_valued_column("capital_loss")
    hours_per_week = tf.contrib.layers.real_valued_column("hours_per_week")

    # Transformations.
    age_buckets = tf.contrib.layers.bucketized_column(age,
                                                      boundaries=[
                                                          18, 25, 30, 35, 40, 45,
                                                          50, 55, 60, 65
                                                      ])

    # Wide columns and deep columns.
    wide_columns = [gender, native_country, education, occupation, workclass,
                    relationship, age_buckets,
                    tf.contrib.layers.crossed_column([education, occupation],
                                                     hash_bucket_size=int(1e4)),
                    tf.contrib.layers.crossed_column(
                        [age_buckets, education, occupation],
                        hash_bucket_size=int(1e6)),
                    tf.contrib.layers.crossed_column([native_country, occupation],
                                                     hash_bucket_size=int(1e4))]
    deep_columns = [
        tf.contrib.layers.embedding_column(workclass, dimension=8),
        tf.contrib.layers.embedding_column(education, dimension=8),
        tf.contrib.layers.embedding_column(gender, dimension=8),
        tf.contrib.layers.embedding_column(relationship, dimension=8),
        tf.contrib.layers.embedding_column(native_country,
                                           dimension=8),
        tf.contrib.layers.embedding_column(occupation, dimension=8),
        age,
        education_num,
        capital_gain,
        capital_loss,
        hours_per_week,
    ]

    if model_type == "wide":
        m = tf.contrib.learn.LinearClassifier(model_dir=model_dir,
                                              feature_columns=wide_columns,
                                              config=run_config)
    elif model_type == "deep":
        m = tf.contrib.learn.DNNClassifier(model_dir=model_dir,
                                           feature_columns=deep_columns,
                                           hidden_units=[100, 50],
                                           config=run_config)
    else:
        m = tf.contrib.learn.DNNLinearCombinedClassifier(
            model_dir=model_dir,
            linear_feature_columns=wide_columns,
            dnn_feature_columns=deep_columns,
            dnn_hidden_units=[100, 50],
            fix_global_step_increment_bug=True,
            config=run_config)
    return m


def input_fn(df):
    """Input builder function."""
    # Creates a dictionary mapping from each continuous feature column name (k) to
    # the values of that column stored in a constant Tensor.
    continuous_cols = {k: tf.constant(df[k].values) for k in CONTINUOUS_COLUMNS}
    # Creates a dictionary mapping from each categorical feature column name (k)
    # to the values of that column stored in a tf.SparseTensor.
    categorical_cols = {
        k: tf.SparseTensor(
            indices=[[i, 0] for i in range(df[k].size)],
            values=df[k].values,
            dense_shape=[df[k].size, 1])
        for k in CATEGORICAL_COLUMNS}
    # Merges the two dictionaries into one.
    feature_cols = dict(continuous_cols)
    feature_cols.update(categorical_cols)
    # Converts the label column into a constant Tensor.
    label = tf.constant(df[LABEL_COLUMN].values)
    # Returns the feature columns and the label.
    return feature_cols, label


def get_df_train(data_dir):
    """Read train data."""
    train_file_name = maybe_download_train(data_dir)
    df_train = pd.read_csv(
        tf.gfile.Open(train_file_name),
        names=COLUMNS,
        skipinitialspace=True,
        engine="python")
    # remove NaN elements
    df_train = df_train.dropna(how='any', axis=0)

    df_train[LABEL_COLUMN] = (
        df_train["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)
    return df_train


def get_df_test(data_dir):
    """Read test data."""
    test_file_name = maybe_download_test(data_dir)
    df_test = pd.read_csv(
        tf.gfile.Open(test_file_name),
        names=COLUMNS,
        skipinitialspace=True,
        skiprows=1,
        engine="python")
    # remove NaN elements
    df_test = df_test.dropna(how='any', axis=0)

    df_test[LABEL_COLUMN] = (
        df_test["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)
    return df_test


FLAGS = None


def _create_experiment_fn(output_dir):
    """Experiment creation function."""
    data_dir = FLAGS.data_dir
    model_dir = FLAGS.model_dir
    model_type = FLAGS.model_type
    train_steps = FLAGS.train_steps

    # Get configuration from environment variables.
    run_config = tf.contrib.learn.RunConfig()
    estimator = build_estimator(model_dir, model_type, run_config)
    return tf.contrib.learn.Experiment(
        estimator=estimator,
        train_input_fn=lambda: input_fn(get_df_train(data_dir)),
        eval_input_fn=lambda: input_fn(get_df_test(data_dir)),
        train_steps=train_steps,
        eval_steps=1
    )


def main(_):
    # Setup cluster spec
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")
    cluster_spec = {"ps": ps_hosts, "worker": worker_hosts}

    job_name = FLAGS.job_name
    task_index = FLAGS.task_index

    cluster = tf.train.ClusterSpec(cluster_spec)
    server = tf.train.Server(cluster,
                             job_name=job_name,
                             task_index=task_index)

    if job_name == "ps":
        server.join()
    elif job_name == "worker":
        # Set TF_CONFIG environment variable. RunConfig will get configuration
        # from environment variables.
        os.environ["TF_CONFIG"] = json.dumps(
            {"cluster": cluster_spec,
            "task": {"type": job_name, "index": task_index}})

        #----------------------------------------
        # Three API for trainning and evaluation.
        #----------------------------------------
        # API one
        #learn_runner.run(experiment_fn=_create_experiment_fn,
        #                 output_dir="NOT-USED",
        #                 schedule="local_run") # call exeperiment's local_run() method

        # API two
        #e = _create_experiment_fn(output_dir="NOT-USED")
        #e.local_run() # call experiment's train_and_evaluate() method

        # API three
        e = _create_experiment_fn(output_dir="NOT-USED")
        e.train_and_evaluate() # call estimator's fit() and evaluate() method



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/tmp",
        help="Directory for storing the cesnsus data."
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="",
        help="Base directory for output models."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="wide_n_deep",
        help="Valid model types: {'wide', 'deep', 'wide_n_deep'}."
    )
    parser.add_argument(
        "--train_steps",
        type=int,
        default=200,
        help="Number of training steps."
    )
    parser.add_argument(
        "--ps_hosts",
        type=str,
        default="",
        help="Hosts of ps job. E.g. 'localhost:6660,localhost:6661'"
    )
    parser.add_argument(
        "--worker_hosts",
        type=str,
        default="",
        help="Hosts of worker job. E.g. 'localhost:6662,localhost:6663'"
    )
    parser.add_argument(
        "--job_name",
        type=str,
        default="",
        help="Job name of the process: 'ps' or 'worker'."
    )
    parser.add_argument(
        "--task_index",
        type=int,
        default=0,
        help="Index of task within the job."
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
