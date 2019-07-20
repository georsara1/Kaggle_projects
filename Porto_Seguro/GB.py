
import pandas as pd
import numpy as np
import h2o
import os

h2o.init(max_mem_size="2G")
h2o.remove_all()

from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.random_forest import H2ORandomForestEstimator


train_all = h2o.import_file('train2.csv')
train_df = train_all[:500000,:]
validation_df = train_all[500000:,:]
test = h2o.import_file('test2.csv')

train_x_variables = train_all.col_names[1:]
train_y_variable = train_all.col_names[0]


rf_v1 = H2ORandomForestEstimator(
    model_id="rf_covType_v1",
    ntrees=150,
    max_depth=30,
    stopping_rounds=5,
    score_each_iteration=False,
    seed=1000000)

model = rf_v1.train(train_x_variables, train_y_variable, training_frame=train_df, validation_frame=validation_df)
rf_v1.score_history()

importances = rf_v1.varimp(use_pandas=True)
impotances_top20=importances['variable'][0:20]
impotances_top20 = impotances_top20.tolist()

rf_v2 = H2ORandomForestEstimator(
    model_id="rf_covType_v1",
    ntrees=300,
    max_depth=30,
    score_each_iteration=True)

model2 = rf_v2.train(impotances_top20, train_y_variable, training_frame=train_all)

rf_v2__predictions = rf_v2.predict(test)
pred=rf_v2__predictions.as_data_frame()

my_gbm_metrics = rf_v1.model_performance(validation_df)
my_gbm_metrics.show()


rf_predictions = rf_v1.predict(test)
pred=rf_predictions.as_data_frame()

gbm_v3 = H2OGradientBoostingEstimator(
    ntrees=30,
    learn_rate=0.3,
    max_depth=10,
    sample_rate=0.7,
    col_sample_rate=0.7,
    stopping_rounds=2,
    stopping_tolerance=0.01, #10-fold increase in threshold as defined in rf_v1
    score_each_iteration=True,
    model_id="gbm_covType_v3",
    seed=2000000
)
gbm_v3.train(train_x, train_y, training_frame=train)
pred_gbm = gbm_v3.predict(test)
pred2 = pred_gbm.as_data_frame()

h2o.shutdown(prompt=False)