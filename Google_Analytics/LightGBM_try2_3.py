#Session-level CV-score: 1.603, User-level score: 1.5863, PL 1.4458
#Session-level CV-score: 1.5986, User-level score: 1.5809 PL 1.4419
#Session-level CV-score: 1.5992, User-level score: 1.5815 PL 1.4415 (minus device.browser)
#Session-level CV-score: 1.599, User-level score: 1.5812 PL 1.4409 (minus device.browser, "trafficSource.keyword")
#Session-level CV-score: 1.5999, User-level score: 1.5815 PL 1.4413 (minus device.browser, "trafficSource.keyword", 'trafficSource.adContent')
#Session-level CV-score: 1.599, User-level score: 1.5808 PL 1.4418 (minus device.browser, "trafficSource.keyword",
# 'trafficSource.adContent', 'trafficSource.adwordsClickInfo.gclId')
#Session-level CV-score: 1.5978, User-level score: 1.5795 (all encodings)
#Session-level CV-score: 1.5874, User-level score: 1.5756 (all encodings + custom interactions)

import numpy as np
import pandas as pd
import time
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
import lightgbm as lgb
from tqdm import tqdm
from sklearn import preprocessing
from itertools import combinations
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans

#Import data
print('Importing data...')
df_train = pd.read_csv('train_flat.csv', dtype={'fullVisitorId': 'str'})
df_test = pd.read_csv('test_flat.csv', dtype={'fullVisitorId': 'str'})

df_train["totals.transactionRevenue"] = df_train["totals.transactionRevenue"].astype('float')


#Find and drop constant columns
const_cols = [c for c in df_train.columns if df_train[c].nunique(dropna=False)==1 ]

cols_to_drop = const_cols + ['sessionId']

df_train = df_train.drop(cols_to_drop + ["trafficSource.campaignCode"], axis=1)
df_test = df_test.drop(cols_to_drop, axis=1)


#-----------------------Feature engineering------------------------
print('Feature engineering...')
#Convert numerical variables to float
num_cols = ["totals.hits", "totals.pageviews",
            "visitNumber", "visitStartTime",
            #'totals.bounces',  'totals.newVisits'
            ]

for col in num_cols:
    df_train[col] = df_train[col].astype(float)
    df_test[col] = df_test[col].astype(float)

print('1.Calculating logs, squares and inverses of numerical features...')
for col in num_cols:
    df_train[col+'_log'] = np.log1p(df_train[col])
    df_test[col+'_log'] = np.log1p(df_test[col].astype(float))
    df_train[col+'^2'] = df_train[col]**2
    df_test[col+'^2'] = df_test[col]**2
    df_train[col+'_inv'] = 1/(df_train[col])
    df_test[col+'_inv'] = 1/df_test[col]



print('2.Clustering...')
num_cols_to_cluster = ['totals.hits',
                       #'totals.newVisits',
                       'visitNumber',
                       'totals.pageviews']

X1 = df_train[num_cols_to_cluster]
X2 = df_test[num_cols_to_cluster]
#X1['totals.newVisits'] = X1['totals.newVisits'].fillna(0)
X1['totals.pageviews'] = X1['totals.pageviews'].fillna(0)
#X2['totals.newVisits'] = X2['totals.newVisits'].fillna(0)
X2['totals.pageviews'] = X2['totals.pageviews'].fillna(0)
kmeans = KMeans().fit(X1)
kmeans_train = kmeans.predict(X1)
kmeans_test = kmeans.predict(X2)
df_train['cluster'] = kmeans_train
df_test['cluster'] = kmeans_test

print('3.Visit start time...')
df_train['formated_visitStartTime'] = df_train['visitStartTime'].apply(
    lambda x: time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(x)))
df_train['formated_visitStartTime'] = pd.to_datetime(df_train['formated_visitStartTime'])
df_train['visit_hour'] = df_train['formated_visitStartTime'].apply(lambda x: x.hour)
del df_train['formated_visitStartTime']

df_test['formated_visitStartTime'] = df_test['visitStartTime'].apply(
    lambda x: time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(x)))
df_test['formated_visitStartTime'] = pd.to_datetime(df_test['formated_visitStartTime'])
df_test['visit_hour'] = df_test['formated_visitStartTime'].apply(lambda x: x.hour)
del df_test['formated_visitStartTime']

print('4.visitid - visit start time...')
df_train['diff_visitId_time'] = df_train['visitId'] - df_train['visitStartTime']
df_train['diff_visitId_time'] = (df_train['diff_visitId_time'] != 0).astype(int)
df_test['diff_visitId_time'] = df_test['visitId'] - df_test['visitStartTime']
df_test['diff_visitId_time'] = (df_test['diff_visitId_time'] != 0).astype(int)

print('5.Binning by quantiles...')
quantile_list = [0, .25, .5, .75, 1.]
percentile_list = [.1, .2, .3, .4, .5, .6, .7, .8, .9, .10]
for col in ["totals.hits",
            "totals.pageviews",
            'visitStartTime',
            #'diff_visitId_time',
            'visitNumber'
            ]:
    df_train[col+'_quantiles'] = pd.qcut(df_train[col], q=quantile_list, duplicates='drop')
    df_test[col + '_quantiles'] = pd.qcut(df_test[col], q=quantile_list, duplicates='drop')
    df_train[col+'_percentiles'] = pd.qcut(df_train[col], q=percentile_list, duplicates='drop')
    df_test[col + '_percentiles'] = pd.qcut(df_test[col], q=percentile_list, duplicates='drop')

print('6.Date engineering...')
format_str = '%Y%m%d'
df_train['formated_date'] = df_train['date'].apply(lambda x: datetime.strptime(str(x), format_str))
df_train['month'] = df_train['formated_date'].apply(lambda x:x.month)
df_train['quarter_month'] = df_train['formated_date'].apply(lambda x:x.day//8)
df_train['day'] = df_train['formated_date'].apply(lambda x:x.day)
df_train['weekday'] = df_train['formated_date'].apply(lambda x:x.weekday())

del df_train['date']
del df_train['formated_date']

df_test['formated_date'] = df_test['date'].apply(lambda x: datetime.strptime(str(x), format_str))
df_test['month'] = df_test['formated_date'].apply(lambda x:x.month)
df_test['quarter_month'] = df_test['formated_date'].apply(lambda x:x.day//8)
df_test['day'] = df_test['formated_date'].apply(lambda x:x.day)
df_test['weekday'] = df_test['formated_date'].apply(lambda x:x.weekday())

del df_test['date']
del df_test['formated_date']

print('7.Total hits mean...')
#df_train['totals.hits'] = df_train['totals.hits'].astype(int)
df_train['mean_hits_per_day'] = df_train.groupby(['day'])['totals.hits'].transform('mean')
#del  df_train['day']

#df_test['totals.hits'] = df_test['totals.hits'].astype(int)
df_test['mean_hits_per_day'] = df_test.groupby(['day'])['totals.hits'].transform('mean')
#del  df_test['day']

print('8.Page views mean...')
#df_train['totals.hits'] = df_train['totals.hits'].astype(int)
df_train['totals.pageviews_per_day'] = df_train.groupby(['day'])['totals.hits'].transform('mean')
#del  df_train['day']

#df_test['totals.hits'] = df_test['totals.hits'].astype(int)
df_test['totals.pageviews_per_day'] = df_test.groupby(['day'])['totals.hits'].transform('mean')
#del  df_test['day']


print('9.Feature interactions...')
def numeric_interaction_terms(df, columns):
    for c in combinations(columns,2):
        df['{} / {}'.format(c[0], c[1]) ] = df[c[0]] / df[c[1]]
        df['{} * {}'.format(c[0], c[1]) ] = df[c[0]] * df[c[1]]
        df['{} - {}'.format(c[0], c[1]) ] = df[c[0]] - df[c[1]]
        df['{} - {}'.format(c[1], c[0])] = df[c[1]] / df[c[0]]
        df['{} - {}'.format(c[1], c[0])] = df[c[1]] - df[c[0]]
    return df

LOG_NUMERIC_COLUMNS = ['visitNumber', 'totals.hits', 'totals.pageviews', 'diff_visitId_time', 'mean_hits_per_day']

df_train = numeric_interaction_terms(df_train,LOG_NUMERIC_COLUMNS)
df_test = numeric_interaction_terms(df_test,LOG_NUMERIC_COLUMNS)

#fill in missing (for Neural networks probably, maybe not for LightGBM)
df_train['totals.bounces'] = df_train['totals.bounces'].fillna('0')
df_test['totals.bounces'] = df_test['totals.bounces'].fillna('0')

df_train['totals.newVisits'] = df_train['totals.newVisits'].fillna('0')
df_test['totals.newVisits'] = df_test['totals.newVisits'].fillna('0')

df_train['trafficSource.adwordsClickInfo.isVideoAd'] = df_train['trafficSource.adwordsClickInfo.isVideoAd'].fillna(True)
df_test['trafficSource.adwordsClickInfo.isVideoAd'] = df_test['trafficSource.adwordsClickInfo.isVideoAd'].fillna(True)

df_train['trafficSource.isTrueDirect'] = df_train['trafficSource.isTrueDirect'].fillna(False)
df_test['trafficSource.isTrueDirect'] = df_test['trafficSource.isTrueDirect'].fillna(False)

#Custom feature interactions
def custom(data):
    print('custom..')
    data['device_deviceCategory_channelGrouping'] = data['device.deviceCategory'] + "_" + data['channelGrouping']
    data['channelGrouping_browser'] = data['device.browser'] + "_" + data['channelGrouping']
    data['channelGrouping_OS'] = data['device.operatingSystem'] + "_" + data['channelGrouping']

    for i in ['geoNetwork.city', 'geoNetwork.continent', 'geoNetwork.country', 'geoNetwork.metro',
              'geoNetwork.networkDomain', 'geoNetwork.region', 'geoNetwork.subContinent']:
        for j in ['device.browser', 'device.deviceCategory', 'device.operatingSystem', 'trafficSource.source']:
            data[i + "_" + j] = data[i] + "_" + data[j]

    #data['content.source'] = data['trafficSource.adContent'] + "_" + data['source.country']
    #data['medium.source'] = data['trafficSource.medium'] + "_" + data['source.country']
    return data


df_train = custom(df_train)
df_test = custom(df_test)


# Label encode categorical variables
# cat_cols = ["channelGrouping", "device.browser",
#             "device.deviceCategory", "device.operatingSystem",
#             "geoNetwork.city", "geoNetwork.continent",
#             "geoNetwork.country", "geoNetwork.metro",
#             "geoNetwork.networkDomain", "geoNetwork.region",
#             "geoNetwork.subContinent", "trafficSource.adContent",
#             "trafficSource.adwordsClickInfo.adNetworkType",
#             "trafficSource.adwordsClickInfo.gclId",
#             "trafficSource.adwordsClickInfo.page",
#             "trafficSource.adwordsClickInfo.slot", "trafficSource.campaign",
#             "trafficSource.keyword", "trafficSource.medium",
#             "trafficSource.referralPath", "trafficSource.source",
#             'trafficSource.adwordsClickInfo.isVideoAd',
#             'trafficSource.isTrueDirect', 'device.isMobile',
#             'totals.hits_quantiles', 'totals.pageviews_quantiles',
#             'visitNumber_quantiles', 'visitStartTime_quantiles',
#             'totals.hits_percentiles', 'totals.pageviews_percentiles',
#             'visitNumber_percentiles', 'visitStartTime_percentiles',
#             'totals.bounces', 'totals.newVisits'
#             ]

cat_cols = [col for col in df_train.columns if df_train[col].dtype not in ['float64', 'int64', 'int32']]

for col in tqdm(cat_cols):
    print(col)
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(df_train[col].values.astype('str')) + list(df_test[col].values.astype('str')))
    df_train[col] = lbl.transform(list(df_train[col].values.astype('str')))
    df_test[col] = lbl.transform(list(df_test[col].values.astype('str')))


#Frequency encoding of categorical variables
print('Frequency encoding...')
def frequency_encoding(frame, col):
    freq_encoding = frame.groupby([col]).size()/frame.shape[0]
    freq_encoding = freq_encoding.reset_index().rename(columns={0:'{}_Frequency'.format(col)})
    return frame.merge(freq_encoding, on=col, how='left')


for col in tqdm(cat_cols):
    df_train = frequency_encoding(df_train, col)
    df_test = frequency_encoding(df_test, col)

#To keep only Frequency encoding uncommend the two lines below:
# for col in cat_cols:
#     del df_train[col]
#     del df_test[col]


#Mean Encoding
def mean_k_fold_encoding(col, alpha):
    target_name = 'totals.transactionRevenue'
    target_mean_global = df_train[target_name].mean()

    nrows_cat = df_train.groupby(col)[target_name].count()
    target_means_cats = df_train.groupby(col)[target_name].mean()
    target_means_cats_adj = (target_means_cats * nrows_cat +
                             target_mean_global * alpha) / (nrows_cat + alpha)
    # Mapping means to test data
    encoded_col_test = df_test[col].map(target_means_cats_adj)

    kfold = KFold(n_splits=5, shuffle=True, random_state=1989)
    parts = []
    for trn_inx, val_idx in kfold.split(df_train):
        df_for_estimation, df_estimated = df_train.iloc[trn_inx], df_train.iloc[val_idx]
        nrows_cat = df_for_estimation.groupby(col)[target_name].count()
        target_means_cats = df_for_estimation.groupby(col)[target_name].mean()

        target_means_cats_adj = (target_means_cats * nrows_cat +
                                 target_mean_global * alpha) / (nrows_cat + alpha)

        encoded_col_train_part = df_estimated[col].map(target_means_cats_adj)
        parts.append(encoded_col_train_part)

    encoded_col_train = pd.concat(parts, axis=0)
    encoded_col_train.fillna(target_mean_global, inplace=True)
    encoded_col_train.sort_index(inplace=True)

    return encoded_col_train, encoded_col_test


for col in tqdm(cat_cols):
    temp_encoded_tr, temp_encoded_te = mean_k_fold_encoding(col, 5)
    new_feat_name = 'mean_k_fold_{}'.format(col)
    df_train[new_feat_name] = temp_encoded_tr.values
    df_test[new_feat_name] = temp_encoded_te.values

#-----------------------Build LightGBM model-----------------------
train_idx = df_train.fullVisitorId
test_idx = df_test.fullVisitorId

df_train["totals.transactionRevenue"] = df_train["totals.transactionRevenue"].astype('float').fillna(0)

train_y = df_train["totals.transactionRevenue"]
train_target = np.log1p(df_train.groupby("fullVisitorId")["totals.transactionRevenue"].sum())

df_train.drop(['fullVisitorId', 'visitId'], axis = 1, inplace = True)
df_test.drop(['fullVisitorId', 'visitId'], axis = 1, inplace = True)

#Check if dropping variables with covariate shift helps
df_train.drop(["device.browser", #better
               "trafficSource.keyword", #better
               #'trafficSource.adContent',
               #'trafficSource.adwordsClickInfo.gclId'
               ], axis = 1, inplace = True)

df_test.drop(["device.browser", #better
              "trafficSource.keyword", #better
              #'trafficSource.adContent',
              #'trafficSource.adwordsClickInfo.gclId'
              ], axis = 1, inplace = True)

y_train = np.log1p(df_train["totals.transactionRevenue"])
x_train = df_train.drop(["totals.transactionRevenue"], axis=1)
x_test = df_test.copy()

folds = KFold(n_splits=5, random_state=6)
oof_preds = np.zeros(x_train.shape[0])
sub_preds = np.zeros(x_test.shape[0])

#Train model
start = time.time()
valid_score = 0
for n_fold, (trn_idx, val_idx) in enumerate(folds.split(x_train, y_train)):
    trn_x, trn_y = x_train.iloc[trn_idx], y_train[trn_idx]
    val_x, val_y = x_train.iloc[val_idx], y_train[val_idx]

    train_data = lgb.Dataset(data=trn_x, label=trn_y)
    valid_data = lgb.Dataset(data=val_x, label=val_y)

    params = {"objective": "regression", "metric": "rmse", 'n_estimators': 10000, 'early_stopping_rounds': 100,
              "num_leaves": 30, "learning_rate": 0.01, "bagging_fraction": 0.9,
              "feature_fraction": 0.3, "bagging_seed": 0, "num_threads": 64}

    lgb_model = lgb.train(params, train_data, valid_sets=[train_data, valid_data], verbose_eval=1000)

    oof_preds[val_idx] = lgb_model.predict(val_x, num_iteration=lgb_model.best_iteration)
    oof_preds[oof_preds < 0] = 0
    sub_pred = lgb_model.predict(x_test, num_iteration=lgb_model.best_iteration) / folds.n_splits
    sub_pred[sub_pred < 0] = 0  # should be greater or equal to 0
    sub_preds += sub_pred
    print('Fold %2d RMSE : %.6f' % (n_fold + 1, np.sqrt(mean_squared_error(val_y, oof_preds[val_idx]))))
    valid_score += np.sqrt(mean_squared_error(val_y, oof_preds[val_idx]))


print('Session-level CV-score:', str(round(valid_score/folds.n_splits,4)))
print(' ')
train_pred = pd.DataFrame({"fullVisitorId":train_idx})
train_pred["PredictedLogRevenue"] = np.expm1(oof_preds)
train_pred = train_pred.groupby("fullVisitorId")["PredictedLogRevenue"].sum().reset_index()
train_pred.columns = ["fullVisitorId", "PredictedLogRevenue"]
train_pred["PredictedLogRevenue"] = np.log1p(train_pred["PredictedLogRevenue"])
train_rmse = np.sqrt(mean_squared_error(train_target, train_pred['PredictedLogRevenue']))
print('User-level score:', str(round(train_rmse, 4)))
print(' ')
end = time.time()
print('training time:', str(round((end - start)/60)), 'mins')

#Predict and write to file for submission
test_pred = pd.DataFrame({"fullVisitorId":test_idx})
test_pred["PredictedLogRevenue"] = np.expm1(sub_preds)
test_pred = test_pred.groupby("fullVisitorId")["PredictedLogRevenue"].sum().reset_index()
test_pred.columns = ["fullVisitorId", "PredictedLogRevenue"]
test_pred["PredictedLogRevenue"] = np.log1p(test_pred["PredictedLogRevenue"])
test_pred.to_csv("lgb_new_2.csv", index=False)

#Print importances
lgb.plot_importance(lgb_model, height=0.5, max_num_features=90, ignore_zero = False,
                    figsize = (12,9), importance_type ='gain')
plt.tight_layout()
plt.show()