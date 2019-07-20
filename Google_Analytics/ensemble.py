import pandas as pd

df_session = pd.read_csv('lgb_new_2.csv')
df_user = pd.read_csv('simple_lgb.csv')

revenue_ens = 0.2*df_user['PredictedLogRevenue']+0.8*df_session['PredictedLogRevenue']
id = df_user['fullVisitorId']

ensemble = pd.DataFrame({'fullVisitorId': id, 'PredictedLogRevenue': revenue_ens})

ensemble = ensemble[['fullVisitorId','PredictedLogRevenue']]

ensemble.to_csv('Ensemble_submission_2.csv', index=False)