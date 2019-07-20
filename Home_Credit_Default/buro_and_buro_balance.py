import pandas as pd

buro = pd.read_pickle('buro')
buro_balance = pd.read_pickle('buro_balance')


#----------------Buro_balance pre-processing
# Bureau and bureau_balance numeric features
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

#One-hot encoding of categorical features in buro data set
#buro_cat_features = [bcol for bcol in buro.columns if buro[bcol].dtype == 'object']
#buro = pd.get_dummies(buro, columns=buro_cat_features, dummy_na = True)
#buro[buro_cat_features] = le.fit_transform(buro[buro_cat_features].astype(str))

#avg_buro = buro.groupby('SK_ID_CURR').mean()
#max_buro = buro.groupby('SK_ID_CURR').max()
#avg_buro['buro_count'] = buro[['SK_ID_BUREAU', 'SK_ID_CURR']].groupby('SK_ID_CURR').count()['SK_ID_BUREAU']