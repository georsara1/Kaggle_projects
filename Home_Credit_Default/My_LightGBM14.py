# Results of this script:
#
# Public leaderboard
#
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.feature_selection import VarianceThreshold
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
le = LabelEncoder()

print('Importing data...')
data = pd.read_pickle('data')
test = pd.read_pickle('test')
prev = pd.read_pickle('prev')
buro = pd.read_pickle('buro')
buro_balance = pd.read_pickle('buro_balance')
credit_card = pd.read_pickle('credit_card')
POS_CASH = pd.read_pickle('pos_cash')
payments = pd.read_pickle('payments')
lgbm_submission = pd.read_csv('sample_submission.csv')

#Separate target variable
y = data['TARGET']
del data['TARGET']

#--------data and test and Feature engineering
all_data = pd.concat([data,test])
# all_data['loan_to_income'] = all_data.AMT_ANNUITY/all_data.AMT_INCOME_TOTAL
# del all_data['AMT_ANNUITY']
# del all_data['AMT_INCOME_TOTAL']
#all_data['REGION_POPULATION_RELATIVE_cut'] = pd.cut(all_data['REGION_POPULATION_RELATIVE'], 4)
#all_data['DAYS_BIRTH_cut'] = pd.cut(all_data['DAYS_BIRTH'], 3)
#all_data['DAYS_EMPLOYED_cut'] = pd.cut(all_data['DAYS_EMPLOYED'], 3)
#all_data['OWN_CAR_AGE_cut'] = pd.cut(all_data['OWN_CAR_AGE'], 2)
#all_data['EXT_SOURCE_1_cut'] = pd.cut(all_data['EXT_SOURCE_1'], 4)
#all_data['EXT_SOURCE_2_cut'] = pd.cut(all_data['EXT_SOURCE_2'], 4)
#all_data['EXT_SOURCE_3_cut'] = pd.cut(all_data['EXT_SOURCE_3'], 4)
#all_data.CODE_GENDER[all_data.CODE_GENDER == 'XNA'] = 'M'
# all_data['DAYS_EMPLOYED_PERC'] = all_data['DAYS_EMPLOYED'] / all_data['DAYS_BIRTH']
# all_data['INCOME_CREDIT_PERC'] = all_data['AMT_INCOME_TOTAL'] / all_data['AMT_CREDIT']
# all_data['INCOME_PER_PERSON'] = all_data['AMT_INCOME_TOTAL'] / all_data['CNT_FAM_MEMBERS']
# all_data['ANNUITY_INCOME_PERC'] = all_data['AMT_ANNUITY'] / all_data['AMT_INCOME_TOTAL']
#to_exclude = np.array(all_data.columns[57:71]) #40:57 (AVG), 57:71 (Mode), 71:85 (Median)
#to_exclude2 = np.array(['CNT_FAM_MEMBERS', 'AMT_GOODS_PRICE'])
#all_data = all_data.drop(to_exclude, axis = 1)
#all_data = all_data.drop(to_exclude2, axis = 1)

#----------------Buro_balance pre-processing
print('Buro Balance pre-processing...')
buro_grouped_size = buro_balance.groupby('SK_ID_BUREAU')['MONTHS_BALANCE'].size()
buro_grouped_max = buro_balance.groupby('SK_ID_BUREAU')['MONTHS_BALANCE'].max()
buro_grouped_min = buro_balance.groupby('SK_ID_BUREAU')['MONTHS_BALANCE'].min()

buro_counts = buro_balance.groupby('SK_ID_BUREAU')['STATUS'].value_counts(normalize = False)
buro_counts_unstacked = buro_counts.unstack('STATUS')
buro_counts_unstacked.columns = ['STATUS_0', 'STATUS_1','STATUS_2','STATUS_3','STATUS_4','STATUS_5','STATUS_C','STATUS_X']
buro_counts_unstacked['MONTHS_COUNT'] = buro_grouped_size
buro_counts_unstacked['MONTHS_MIN'] = buro_grouped_min
buro_counts_unstacked['MONTHS_MAX'] = buro_grouped_max
buro_counts_unstacked = buro_counts_unstacked.reset_index()

#-----------------------Buro pre-processing--------------------
print('Buro pre-processing...')
buro = buro.merge(buro_counts_unstacked, how='left', on='SK_ID_BUREAU')
del buro['SK_ID_BUREAU']

active = buro[buro['CREDIT_ACTIVE'] == 'Active']
closed = buro[buro['CREDIT_ACTIVE'] == 'Closed']
closed = closed.rename(index=str, columns={"CREDIT_ACTIVE": "CREDIT_CLOSED"})

num_aggregations = {
    'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
    'CREDIT_DAY_OVERDUE': ['max', 'mean'],
    'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
    'AMT_CREDIT_MAX_OVERDUE': ['mean'],
    'CNT_CREDIT_PROLONG': ['sum'],
    'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
    'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
    'AMT_CREDIT_SUM_OVERDUE': ['mean'],
    'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
    'DAYS_CREDIT_UPDATE': ['min', 'max', 'mean'],
    'AMT_ANNUITY': ['max', 'mean'],
    'MONTHS_COUNT': ['min'],
    'MONTHS_MIN': ['min'],
    'MONTHS_MAX': ['max']
}
bureau_agg = buro.groupby('SK_ID_CURR').agg({**num_aggregations})
bureau_agg = bureau_agg.reset_index()
bureau_agg.columns = ['_'.join(col) for col in bureau_agg.columns.values]
bureau_agg = bureau_agg.rename(index=str, columns={"SK_ID_CURR_": "SK_ID_CURR"})

active_count = active.groupby('SK_ID_CURR')['CREDIT_ACTIVE'].count()
closed_count = closed.groupby('SK_ID_CURR')['CREDIT_CLOSED'].count()

bureau_agg = bureau_agg.join(active_count, how='left', on='SK_ID_CURR')
bureau_agg = bureau_agg.join(closed_count, how='left', on='SK_ID_CURR')

#-----------------------Previous Application pre-processing--------------------
print('Previous Application pre-processing...')
#Numerical features
prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']

num_aggregations = {
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
    }

prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations}).reset_index()
prev_agg.columns = ['_'.join(col) for col in prev_agg.columns.values]
prev_agg = prev_agg.rename(index=str, columns={"SK_ID_CURR_": "SK_ID_CURR"})

#Categorical features
cnt_prev = prev[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count().reset_index()
cnt_prev = cnt_prev.rename(index=str, columns={"SK_ID_PREV": "COUNT_PREV_APP"})

approved = prev[prev['NAME_CONTRACT_STATUS'] == 'Approved']
refused = prev[prev['NAME_CONTRACT_STATUS'] == 'Refused']
approved = approved.rename(index=str, columns={"NAME_CONTRACT_STATUS": "NAME_CONTRACT_STATUS_ACC"})
refused = refused.rename(index=str, columns={"NAME_CONTRACT_STATUS": "NAME_CONTRACT_STATUS_REJ"})

approved_count = approved.groupby('SK_ID_CURR')['NAME_CONTRACT_STATUS_ACC'].count().reset_index()
refused_count = refused.groupby('SK_ID_CURR')['NAME_CONTRACT_STATUS_REJ'].count().reset_index()

prev_cat_features = [pcol for pcol in prev.columns if prev[pcol].dtype == 'object']
id_temp = prev['SK_ID_CURR'].values

prev_cat_only = prev[prev_cat_features]
prev_cat_only = pd.get_dummies(prev_cat_only)
#prev[prev_cat_features] = le.fit_transform(prev[prev_cat_features].astype(str))
prev_cat_only['SK_ID_CURR'] = id_temp

avg_prev = prev_cat_only.groupby('SK_ID_CURR').mean().reset_index()
max_prev = prev_cat_only.groupby('SK_ID_CURR').max().reset_index()
min_prev = prev_cat_only.groupby('SK_ID_CURR').min().reset_index()

#Join all new features
prev_agg = prev_agg.merge(cnt_prev, how='left', on='SK_ID_CURR')
prev_agg = prev_agg.merge(approved_count, how='left', on='SK_ID_CURR')
prev_agg = prev_agg.merge(refused_count, how='left', on='SK_ID_CURR')
prev_agg['ratio_approved'] = prev_agg['COUNT_PREV_APP']/prev_agg['NAME_CONTRACT_STATUS_ACC']

#prev_agg = prev_agg.merge(avg_prev, how='left', on='SK_ID_CURR')
#prev_agg = prev_agg.merge(max_prev, how='left', on='SK_ID_CURR')
#prev_agg = prev_agg.merge(min_prev, how='left', on='SK_ID_CURR')

del approved_count
del refused_count
del avg_prev
del max_prev
del min_prev

#-----------------------POS-CASH pre-processing--------------------
print('POS-CASH pre-processing...')

POS_CASH['NAME_CONTRACT_STATUS'] = le.fit_transform(POS_CASH['NAME_CONTRACT_STATUS'].astype(str))
nunique_status = POS_CASH[['SK_ID_CURR', 'NAME_CONTRACT_STATUS']].groupby('SK_ID_CURR').nunique()
nunique_status2 = POS_CASH[['SK_ID_CURR', 'NAME_CONTRACT_STATUS']].groupby('SK_ID_CURR').max()
nunique_status3 = POS_CASH[['SK_ID_CURR', 'NAME_CONTRACT_STATUS']].groupby('SK_ID_CURR').mean()
POS_CASH['UNIQUE_STATUS_count'] = nunique_status['NAME_CONTRACT_STATUS']
POS_CASH['NUNIQUE_STATUS_max'] = nunique_status2['NAME_CONTRACT_STATUS']
POS_CASH['NUNIQUE_STATUS_mean'] = nunique_status3['NAME_CONTRACT_STATUS']
POS_CASH.drop(['SK_ID_PREV', 'NAME_CONTRACT_STATUS'], axis=1, inplace=True)

#-----------------------Credit Card pre-processing--------------------
print('Credit Card pre-processing...')
# #FEATURE 1 - NUMBER OF LOANS PER CUSTOMER
# CCB = credit_card.copy()
# grp = CCB.groupby(by = ['SK_ID_CURR'])['SK_ID_PREV'].nunique().reset_index().rename(index = str, columns = {'SK_ID_PREV': 'NO_LOANS'})
# credit_card = credit_card.merge(grp, on = ['SK_ID_CURR'], how = 'left')
#
# #FEATURE 3 - AVG % LOADING OF CREDIT LIMIT PER CUSTOMER
# CCB = credit_card.copy()
#
# CCB['AMT_CREDIT_LIMIT_ACTUAL1'] = CCB['AMT_CREDIT_LIMIT_ACTUAL']
#
# def f(x1, x2):
#     balance = x1.max()
#     limit = x2.max()
#
#     return (balance / limit)
#
#
# # Calculate the ratio of Amount Balance to Credit Limit - CREDIT LOAD OF CUSTOMER
# # This is done for each Credit limit value per loan per Customer
# grp = CCB.groupby(by=['SK_ID_CURR', 'SK_ID_PREV', 'AMT_CREDIT_LIMIT_ACTUAL']).apply(
#     lambda x: f(x.AMT_BALANCE, x.AMT_CREDIT_LIMIT_ACTUAL1)).reset_index().rename(index=str, columns={0: 'CREDIT_LOAD1'})
# del CCB['AMT_CREDIT_LIMIT_ACTUAL1']
#
# # We now calculate the mean Credit load of All Loan transactions of Customer
# grp1 = grp.groupby(by=['SK_ID_CURR'])['CREDIT_LOAD1'].mean().reset_index().rename(index=str, columns={
#     'CREDIT_LOAD1': 'CREDIT_LOAD'})
# print(grp1.dtypes)
#
# credit_card = credit_card.merge(grp1, on=['SK_ID_CURR'], how='left')
#
# # FEATURE 4 - AVERAGE NUMBER OF TIMES DAYS PAST DUE HAS OCCURRED PER CUSTOMER
# CCB = credit_card.copy()
#
# def f1(DPD):
#     # DPD is a series of values of SK_DPD for each of the groupby combination
#     # We convert it to a list to get the number of SK_DPD values NOT EQUALS ZERO
#     x = DPD.tolist()
#     c = 0
#     for i, j in enumerate(x):
#         if j != 0:
#             c += 1
#
#     return c
#
#
# grp = CCB.groupby(by=['SK_ID_CURR',
#                       'SK_ID_PREV']).apply(lambda x: f1(x.SK_DPD)).reset_index().rename(index=str,columns={0: 'NO_DPD'})
#
# grp1 = grp.groupby(by=['SK_ID_CURR'])['NO_DPD'].mean().reset_index().rename(index=str, columns={'NO_DPD': 'DPD_COUNT'})
# credit_card = credit_card.merge(grp1, on = ['SK_ID_CURR'], how = 'left')
#
# #FEATURE 5 - AVERAGE OF DAYS PAST DUE PER CUSTOMER
# CCB = credit_card.copy()
#
# grp = CCB.groupby(by= ['SK_ID_CURR'])['SK_DPD'].mean().reset_index().rename(index = str, columns = {'SK_DPD': 'AVG_DPD'})
# credit_card = credit_card.merge(grp, on = ['SK_ID_CURR'], how = 'left')
#
# #FEATURE 6 - % of MINIMUM PAYMENTS MISSED
# CCB = credit_card.copy()
#
# def f2(min_pay, total_pay):
#     M = min_pay.tolist()
#     T = total_pay.tolist()
#     P = len(M)
#     c = 0
#     # Find the count of transactions when Payment made is less than Minimum Payment
#     for i in range(len(M)):
#         if T[i] < M[i]:
#             c += 1
#     return (100 * c) / P
#
#
# grp = CCB.groupby(by=['SK_ID_CURR']).apply(
#     lambda x: f2(x.AMT_INST_MIN_REGULARITY, x.AMT_PAYMENT_CURRENT)).reset_index().rename(index=str, columns={
#     0: 'PERCENTAGE_MISSED_PAYMENTS'})
# credit_card = credit_card.merge(grp, on=['SK_ID_CURR'], how='left')
#
# #FEATURE 7 - RATIO OF CASH VS CARD SWIPES
# CCB = credit_card.copy()
#
# grp = CCB.groupby(by = ['SK_ID_CURR'])['AMT_DRAWINGS_ATM_CURRENT'].sum().reset_index().rename(index = str, columns = {'AMT_DRAWINGS_ATM_CURRENT' : 'DRAWINGS_ATM'})
# CCB = CCB.merge(grp, on = ['SK_ID_CURR'], how = 'left')
# del grp
#
# grp = CCB.groupby(by = ['SK_ID_CURR'])['AMT_DRAWINGS_CURRENT'].sum().reset_index().rename(index = str, columns = {'AMT_DRAWINGS_CURRENT' : 'DRAWINGS_TOTAL'})
# CCB = CCB.merge(grp, on = ['SK_ID_CURR'], how = 'left')
# del grp
#
# CCB['CASH_CARD_RATIO1'] = (CCB['DRAWINGS_ATM']/CCB['DRAWINGS_TOTAL'])*100
# del CCB['DRAWINGS_ATM']
# del CCB['DRAWINGS_TOTAL']
#
# grp = CCB.groupby(by = ['SK_ID_CURR'])['CASH_CARD_RATIO1'].mean().reset_index().rename(index = str, columns ={ 'CASH_CARD_RATIO1' : 'CASH_CARD_RATIO'})
# credit_card = credit_card.merge(grp, on = ['SK_ID_CURR'], how = 'left')
#
# #FEATURE 8 - AVERAGE DRAWING PER CUSTOMER
# CCB = credit_card.copy()
#
# grp = CCB.groupby(by = ['SK_ID_CURR'])['AMT_DRAWINGS_CURRENT'].sum().reset_index().rename(index = str, columns = {'AMT_DRAWINGS_CURRENT' : 'TOTAL_DRAWINGS'})
# CCB = CCB.merge(grp, on = ['SK_ID_CURR'], how = 'left')
# del grp
#
# grp = CCB.groupby(by = ['SK_ID_CURR'])['CNT_DRAWINGS_CURRENT'].sum().reset_index().rename(index = str, columns = {'CNT_DRAWINGS_CURRENT' : 'NO_DRAWINGS'})
# CCB = CCB.merge(grp, on = ['SK_ID_CURR'], how = 'left')
# del grp
#
# CCB['DRAWINGS_RATIO1'] = (CCB['TOTAL_DRAWINGS']/CCB['NO_DRAWINGS'])*100
# del CCB['TOTAL_DRAWINGS']
# del CCB['NO_DRAWINGS']
#
# grp = CCB.groupby(by = ['SK_ID_CURR'])['DRAWINGS_RATIO1'].mean().reset_index().rename(index = str, columns ={ 'DRAWINGS_RATIO1' : 'DRAWINGS_RATIO'})
# credit_card = credit_card.merge(grp, on = ['SK_ID_CURR'], how = 'left')

credit_card['NAME_CONTRACT_STATUS'] = le.fit_transform(credit_card['NAME_CONTRACT_STATUS'].astype(str))
nunique_status = credit_card[['SK_ID_CURR', 'NAME_CONTRACT_STATUS']].groupby('SK_ID_CURR').nunique()
nunique_status2 = credit_card[['SK_ID_CURR', 'NAME_CONTRACT_STATUS']].groupby('SK_ID_CURR').max()
nunique_status3 = credit_card[['SK_ID_CURR', 'NAME_CONTRACT_STATUS']].groupby('SK_ID_CURR').mean()
credit_card['NUNIQUE_STATUS'] = nunique_status['NAME_CONTRACT_STATUS']
credit_card['NUNIQUE_STATUS2'] = nunique_status2['NAME_CONTRACT_STATUS']
credit_card['NUNIQUE_STATUS3'] = nunique_status3['NAME_CONTRACT_STATUS']
credit_card.drop(['SK_ID_PREV', 'NAME_CONTRACT_STATUS'], axis=1, inplace=True)

#-----------------------Payments pre-processing--------------------
print('Payments pre-processing...')
avg_payments = payments.groupby('SK_ID_CURR').mean()
avg_payments2 = payments.groupby('SK_ID_CURR').max()
avg_payments3 = payments.groupby('SK_ID_CURR').min()
del avg_payments['SK_ID_PREV']
del avg_payments2['SK_ID_PREV']
del avg_payments3['SK_ID_PREV']

#-------------------------Join all_data bases---------------------
print('Joining all_databases...')
all_data = all_data.merge(right=prev_agg, how='left', on='SK_ID_CURR')
all_data = all_data.merge(right=bureau_agg, how='left', on='SK_ID_CURR')
#all_data = all_data.merge(right=max_prev.reset_index(), how='left', on='SK_ID_CURR')
#all_data = all_data.merge(right=max_buro.reset_index(), how='left', on='SK_ID_CURR')
all_data = all_data.merge(POS_CASH.groupby('SK_ID_CURR').mean().reset_index(), how='left', on='SK_ID_CURR')
all_data = all_data.merge(credit_card.groupby('SK_ID_CURR').mean().reset_index(), how='left', on='SK_ID_CURR')
all_data = all_data.merge(right=avg_payments.reset_index(), how='left', on='SK_ID_CURR')
all_data = all_data.merge(right=avg_payments2.reset_index(), how='left', on='SK_ID_CURR')
all_data = all_data.merge(right=avg_payments3.reset_index(), how='left', on='SK_ID_CURR')

#One-hot encoding of categorical features in data and test sets
categorical_features = data.select_dtypes(exclude=["number"]).columns
#all_data[categorical_features] = le.fit_transform(all_data[categorical_features].astype(str))
all_data = pd.get_dummies(all_data, columns=categorical_features, dummy_na = True)

#Split again in train and test
data = all_data.iloc[:data.shape[0],:]
test = all_data.iloc[data.shape[0]:,]

#Remove features with many missing values
print('Removing features with more than 80% missing...')
test = test[test.columns[data.isnull().mean() < 0.9]]
data = data[data.columns[data.isnull().mean() < 0.9]]

#Delete customer Id
del data['SK_ID_CURR']
del test['SK_ID_CURR']

#Create train and validation set
train_x, valid_x, train_y, valid_y = train_test_split(data, y, test_size=0.1, shuffle=True, random_state=34442)

#------------------------Build LightGBM Model-----------------------
train_data=lgb.Dataset(train_x,label=train_y)
valid_data=lgb.Dataset(valid_x,label=valid_y)

#Select Hyper-Parameters
params = {'boosting_type': 'gbdt',
          'max_depth' : 10,
          'objective': 'binary',
          'nthread': 5,
          'num_leaves': 64,
          'learning_rate': 0.05,
          'max_bin': 512,
          'subsample_for_bin': 200,
          'subsample': 0.8,
          'subsample_freq': 1,
          'colsample_bytree': 0.8,
          'reg_alpha': 5,
          'reg_lambda': 10,
          'min_split_gain': 0.1,
          'min_child_weight': 1,
          'min_child_samples': 5,
          'scale_pos_weight': 1,
          'num_class' : 1,
          'metric' : 'auc'
          }

#Train model on selected parameters and number of iterations
lgbm = lgb.train(params,
                 train_data,
                 2500,
                 valid_sets=valid_data,
                 early_stopping_rounds= 40,
                 verbose_eval= 10
                 )

#Predict on test set and write to submit
predictions_lgbm_prob = lgbm.predict(test)

lgbm_submission.TARGET = predictions_lgbm_prob

lgbm_submission.to_csv('lgbm_submission.csv', index=False)

#Plot Variable Importances
#lgb.plot_importance(lgbm, max_num_features=21, importance_type='split')
