# Results of this script:
# [420]	valid_0's auc: 0.782757
# Public leaderboard 0.782
# submitted as test-komo1
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
data = pd.read_csv('application_train.csv',na_values=[365243])
test = pd.read_csv('application_test.csv',na_values=[365243])
prev = pd.read_csv('previous_application.csv',na_values=[365243])
buro = pd.read_csv('bureau.csv',na_values=[365243])
buro_balance = pd.read_csv('bureau_balance.csv')
credit_card  = pd.read_csv('credit_card_balance.csv',na_values=[365243])
POS_CASH  = pd.read_csv('POS_CASH_balance.csv',na_values=[365243])
payments = pd.read_csv('installments_payments.csv',na_values=[365243])
lgbm_submission = pd.read_csv('sample_submission.csv',na_values=[365243])

#Separate target variable
y = data['TARGET']
del data['TARGET']

#Feature engineering
all_data = pd.concat([data,test])
#all_data['loan_to_income'] = all_data.AMT_ANNUITY/all_data.AMT_INCOME_TOTAL
#all_data['REGION_POPULATION_RELATIVE_cut'] = pd.cut(all_data['REGION_POPULATION_RELATIVE'], 4)
#all_data['DAYS_BIRTH_cut'] = pd.cut(all_data['DAYS_BIRTH'], 3)
#all_data['DAYS_EMPLOYED_cut'] = pd.cut(all_data['DAYS_EMPLOYED'], 3)
#all_data['OWN_CAR_AGE_cut'] = pd.cut(all_data['OWN_CAR_AGE'], 2)
#all_data['EXT_SOURCE_1_cut'] = pd.cut(all_data['EXT_SOURCE_1'], 4)
#all_data['EXT_SOURCE_2_cut'] = pd.cut(all_data['EXT_SOURCE_2'], 4)
#all_data['EXT_SOURCE_3_cut'] = pd.cut(all_data['EXT_SOURCE_3'], 4)
#all_data.CODE_GENDER[all_data.CODE_GENDER == 'XNA'] = 'M'
all_data = all_data.drop(to_exclude2, axis = 1)

#Magic to buro_balance and join with buro dataframe
buro_grouped_size = buro_balance.groupby('SK_ID_BUREAU')['MONTHS_BALANCE'].size()
buro_grouped_max = buro_balance.groupby('SK_ID_BUREAU')['MONTHS_BALANCE'].max()
buro_grouped_min = buro_balance.groupby('SK_ID_BUREAU')['MONTHS_BALANCE'].min()

buro_counts = buro_balance.groupby('SK_ID_BUREAU')['STATUS'].value_counts(normalize = False)
buro_counts_unstacked = buro_counts.unstack('STATUS')
buro_counts_unstacked.columns = ['STATUS_0', 'STATUS_1','STATUS_2','STATUS_3','STATUS_4','STATUS_5','STATUS_C','STATUS_X']
buro_counts_unstacked['MONTHS_COUNT'] = buro_grouped_size
buro_counts_unstacked['MONTHS_MIN'] = buro_grouped_min
buro_counts_unstacked['MONTHS_MAX'] = buro_grouped_max

buro = buro.join(buro_counts_unstacked, how='left', on='SK_ID_BUREAU')

#One-hot encoding of categorical features in previous application data set
prev_cat_features = [pcol for pcol in prev.columns if prev[pcol].dtype == 'object']
prev = pd.get_dummies(prev, columns=prev_cat_features)
#prev[prev_cat_features] = le.fit_transform(prev[prev_cat_features].astype(str))

#Do weird stuff vol1
print('Doing weird stuff...')
avg_prev = prev.groupby('SK_ID_CURR').mean()
max_prev = prev.groupby('SK_ID_CURR').max()
cnt_prev = prev[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
avg_prev['nb_app'] = cnt_prev['SK_ID_PREV']
del avg_prev['SK_ID_PREV']
del max_prev['SK_ID_PREV']

#One-hot encoding of categorical features in buro data set
buro_cat_features = [bcol for bcol in buro.columns if buro[bcol].dtype == 'object']
buro = pd.get_dummies(buro, columns=buro_cat_features)
#buro[buro_cat_features] = le.fit_transform(buro[buro_cat_features].astype(str))

#Do weird stuff vol2
print('Doing weird stuff vol 2...')
avg_buro = buro.groupby('SK_ID_CURR').mean()
#max_buro = buro.groupby('SK_ID_CURR').max()
avg_buro['buro_count'] = buro[['SK_ID_BUREAU', 'SK_ID_CURR']].groupby('SK_ID_CURR').count()['SK_ID_BUREAU']
del avg_buro['SK_ID_BUREAU']
#del max_buro['SK_ID_BUREAU']

#Do weird stuff vol3
print('Doing weird stuff vol 3...')

POS_CASH['NAME_CONTRACT_STATUS'] = le.fit_transform(POS_CASH['NAME_CONTRACT_STATUS'].astype(str))
nunique_status = POS_CASH[['SK_ID_CURR', 'NAME_CONTRACT_STATUS']].groupby('SK_ID_CURR').nunique()
nunique_status2 = POS_CASH[['SK_ID_CURR', 'NAME_CONTRACT_STATUS']].groupby('SK_ID_CURR').max()
nunique_status3 = POS_CASH[['SK_ID_CURR', 'NAME_CONTRACT_STATUS']].groupby('SK_ID_CURR').mean()
POS_CASH['NUNIQUE_STATUS'] = nunique_status['NAME_CONTRACT_STATUS']
POS_CASH['NUNIQUE_STATUS2'] = nunique_status2['NAME_CONTRACT_STATUS']
POS_CASH['NUNIQUE_STATUS3'] = nunique_status3['NAME_CONTRACT_STATUS']
POS_CASH.drop(['SK_ID_PREV', 'NAME_CONTRACT_STATUS'], axis=1, inplace=True)

#Do weird stuff vol4
print('Doing weird stuff vol 4...')
credit_card['NAME_CONTRACT_STATUS'] = le.fit_transform(credit_card['NAME_CONTRACT_STATUS'].astype(str))
nunique_status = credit_card[['SK_ID_CURR', 'NAME_CONTRACT_STATUS']].groupby('SK_ID_CURR').nunique()
nunique_status2 = credit_card[['SK_ID_CURR', 'NAME_CONTRACT_STATUS']].groupby('SK_ID_CURR').max()
nunique_status3 = credit_card[['SK_ID_CURR', 'NAME_CONTRACT_STATUS']].groupby('SK_ID_CURR').mean()
credit_card['NUNIQUE_STATUS'] = nunique_status['NAME_CONTRACT_STATUS']
credit_card['NUNIQUE_STATUS2'] = nunique_status2['NAME_CONTRACT_STATUS']
credit_card['NUNIQUE_STATUS3'] = nunique_status3['NAME_CONTRACT_STATUS']
credit_card.drop(['SK_ID_PREV', 'NAME_CONTRACT_STATUS'], axis=1, inplace=True)

#Do weird stuff vol5
print('Doing weird stuff vol 5...')
avg_payments = payments.groupby('SK_ID_CURR').mean()
avg_payments2 = payments.groupby('SK_ID_CURR').max()
avg_payments3 = payments.groupby('SK_ID_CURR').min()
del avg_payments['SK_ID_PREV']

#Join all_data bases
print('Joining all_databases...')
all_data = all_data.merge(right=avg_prev.reset_index(), how='left', on='SK_ID_CURR')
all_data = all_data.merge(right=avg_buro.reset_index(), how='left', on='SK_ID_CURR')
all_data = all_data.merge(right=max_prev.reset_index(), how='left', on='SK_ID_CURR')
#all_data = all_data.merge(right=max_buro.reset_index(), how='left', on='SK_ID_CURR')
all_data = all_data.merge(POS_CASH.groupby('SK_ID_CURR').mean().reset_index(), how='left', on='SK_ID_CURR')
all_data = all_data.merge(credit_card.groupby('SK_ID_CURR').mean().reset_index(), how='left', on='SK_ID_CURR')
all_data = all_data.merge(right=avg_payments.reset_index(), how='left', on='SK_ID_CURR')
all_data = all_data.merge(right=avg_payments2.reset_index(), how='left', on='SK_ID_CURR')
all_data = all_data.merge(right=avg_payments3.reset_index(), how='left', on='SK_ID_CURR')

#One-hot encoding of categorical features in data and test sets
data = data.drop(to_exclude2, axis = 1)
categorical_features = data.select_dtypes(exclude=["number"]).columns
#all_data[categorical_features] = le.fit_transform(all_data[categorical_features].astype(str))
all_data = pd.get_dummies(all_data, columns=categorical_features)

#Split again in train and test
data = all_data.iloc[:data.shape[0],:]
test = all_data.iloc[data.shape[0]:,]

#Remove features with many missing values
print('Removing features with more than 80% missing...')
#test = test[test.columns[data.isnull().mean() < 0.85]]
#data = data[data.columns[data.isnull().mean() < 0.85]]

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

f = lgbm.feature_importance()

feat_imp = pd.DataFrame({'feature': data.columns, 'score': f})
feat_imp = feat_imp.sort_values(by = 'score', ascending=False)