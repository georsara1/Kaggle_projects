import pandas as pd

lgb1 = pd.read_csv('lgbm_submission.csv')
lgb2 = pd.read_csv('lgbm_submission2.csv')
lgb3 = pd.read_csv('lgbm_submission3.csv')
lgb4 = pd.read_csv('lgbm_submission4.csv')
lgb5 = pd.read_csv('lgbm_submission5.csv')
lgb6 = pd.read_csv('lgbm_submission6.csv')
lgb7 = pd.read_csv('lgbm_submission7.csv')

lgbm_all = pd.DataFrame({
                        'lgb1': lgb1.TARGET,
                        'lgb2': lgb2.TARGET,
                        'lgb3': lgb3.TARGET,
                        'lgb4': lgb4.TARGET,
                        'lgb5': lgb5.TARGET,
                        'lgb6': lgb6.TARGET,
                        'lgb7': lgb7.TARGET,

})

corrs = lgbm_all.corr()

lgb_ensemble = lgb1.copy()
lgb_ensemble.TARGET = (lgbm_all.lgb2+lgbm_all.lgb4+lgbm_all.lgb7)/3

lgb_ensemble.to_csv('lgbm_ensemble_submission.csv', index=False)

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
plt.hist(lgbm_all.lgb2, bins = 500, label = 'lgb2')
plt.hist(lgbm_all.lgb7, bins = 500, label = 'lgb7')
plt.hist(lgbm_all.lgb4, bins = 500, label = 'lgb4')
plt.legend()
plt.show()