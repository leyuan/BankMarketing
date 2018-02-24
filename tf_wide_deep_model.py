import sys
import random
import pandas as pd
import tensorflow as tf
from IPython.display import display
from sklearn.model_selection import train_test_split

data = pd.read_csv("rawData/bank-additional-full.csv", sep=";")
# Encode it as integer for machine learning algorithms
data = data.replace({"y" : {"no" : 0, "yes" : 1}})

data.columns = ['age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week',
                'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'emp_var_rate', 'cons_price_idx',
                'cons_conf_idx', 'euribor3m', 'nr_employed', 'y']

data = data.drop(["default"], axis=1)
data = data.drop(["duration"], axis=1)
data = data.drop(["pdays"], axis=1)

COLUMNS = ['age', 'job', 'marital', 'education', 'housing', 'loan', 'contact', 'month', 'day_of_week',
           'campaign', 'previous', 'poutcome', 'emp_var_rate', 'cons_price_idx',
           'cons_conf_idx', 'euribor3m', 'nr_employed', 'y']

FEATURES = ['age', 'job', 'marital', 'education', 'housing', 'loan', 'contact', 'month', 'day_of_week',
           'campaign', 'previous', 'poutcome', 'emp_var_rate', 'cons_price_idx',
           'cons_conf_idx', 'euribor3m', 'nr_employed']

LABEL = "y"


# display(data.head())
# print("NAs for data:"+str(data.isnull().values.sum()))

train, test = train_test_split(data, test_size=0.1)
# model_dir = "wide_model/"
model_dir = "wide_deep_model/"

def build_model_columns():
    age = tf.feature_column.numeric_column('age')
    job = tf.feature_column.categorical_column_with_vocabulary_list('job', data.job.unique())
    marital = tf.feature_column.categorical_column_with_vocabulary_list('marital', data.marital.unique())
    education = tf.feature_column.categorical_column_with_vocabulary_list('education', data.education.unique())
    loan = tf.feature_column.categorical_column_with_vocabulary_list('loan', data.loan.unique())
    contact = tf.feature_column.categorical_column_with_vocabulary_list('contact', data.contact.unique())
    month = tf.feature_column.categorical_column_with_vocabulary_list('month', data.month.unique())
    day_of_week = tf.feature_column.categorical_column_with_vocabulary_list('day_of_week', data.day_of_week.unique())
    campaign = tf.feature_column.numeric_column('campaign')
    previous = tf.feature_column.numeric_column('previous')
    poutcome = tf.feature_column.categorical_column_with_vocabulary_list('poutcome', data.poutcome.unique())
    emp_var_rate = tf.feature_column.numeric_column('emp_var_rate')
    cons_price_idx = tf.feature_column.numeric_column('cons_price_idx')
    euribor3m = tf.feature_column.numeric_column('euribor3m')
    nr_employed = tf.feature_column.numeric_column('nr_employed')


    base_columns = [age, job, marital, education, loan, contact, month, day_of_week, campaign, previous,
                    poutcome, emp_var_rate, cons_price_idx, euribor3m, nr_employed]

    crossed_columns = [
        tf.feature_column.crossed_column(['job', 'education'], hash_bucket_size=1000),
        tf.feature_column.crossed_column(['previous', 'poutcome'], hash_bucket_size=1000)
    ]

    wide_columns = base_columns + crossed_columns
    deep_columns = [
        age,
        euribor3m,
        nr_employed,
        campaign,
        previous,
        emp_var_rate,
        cons_price_idx,
        tf.feature_column.indicator_column(job),
        tf.feature_column.indicator_column(poutcome)
    ]

    return wide_columns, deep_columns

def build_estimator():
    wide_columns, deep_columns = build_model_columns()
    hidden_units = [100, 75, 50, 25]
    run_config = tf.estimator.RunConfig().replace(
      session_config=tf.ConfigProto(device_count={'GPU': 0}))

    # return tf.estimator.LinearClassifier(
    #     model_dir=model_dir,
    #     feature_columns=wide_columns,
    #     config=run_config)

    return tf.estimator.DNNLinearCombinedClassifier(
        model_dir=model_dir,
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=hidden_units,
        config=run_config
    )

def input_fn(data_set, num_epochs=1, shuffle=True):
    return tf.estimator.inputs.pandas_input_fn(
        x=pd.DataFrame({k: data_set[k].values for k in FEATURES}),
        y=pd.Series(data_set[LABEL].values),
        num_epochs=num_epochs,
        shuffle=shuffle
    )

def main(unused_argv):
    train_epochs = 40
    epochs_per_eval = 2

    model = build_estimator()

    for n in range(train_epochs // epochs_per_eval):
        model.train(input_fn=input_fn(train))

        results = model.evaluate(input_fn=input_fn(test))

        # Display evaluation metrics
        print('Results at epoch', (n + 1) * epochs_per_eval)
        print('-' * 60)

        for key in sorted(results):
            print('%s: %s' % (key, results[key]))


if __name__ == '__main__':
    tf.app.run(main=main, argv=[sys.argv[0]])
