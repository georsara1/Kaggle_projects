import warnings
warnings.filterwarnings('ignore')

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
#print(os.listdir("../input"))

import lightgbm as lgb
from sklearn.model_selection import *
from sklearn.metrics import mean_squared_error, make_scorer
from scipy.stats import mode, skew, kurtosis, entropy
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import seaborn as sns

import dask.dataframe as dd
from dask.multiprocessing import get

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

transact_cols = [f for f in train.columns if f not in ["ID", "target"]]
y = np.log1p(train["target"]).values

test["target"] = train["target"].mean()

cols = ['f190486d6', '58e2e02e6', 'eeb9cd3aa', '9fd594eec', '6eef030c1', '15ace8c9f',
        'fb0f5dbfe', '58e056e12', '20aa07010', '024c577b9', 'd6bb78916', 'b43a7cfd5',
        '58232a6fb', '1702b5bf0', '324921c7b', '62e59a501', '2ec5b290f', '241f0f867',
        'fb49e4212', '66ace2992', 'f74e8f13d', '5c6487af1', '963a49cdc', '26fc93eb7',
        '1931ccfdd', '703885424', '70feb1494', '491b9ee45', '23310aa6f', 'e176a204a',
        '6619d81fc', '1db387535',
        'fc99f9426', '91f701ba2', '0572565c2', '190db8488', 'adb64ff71', 'c47340d97', 'c5a231d81', '0ff32eb98'
       ]

# from: https://www.kaggle.com/dfrumkin/a-simple-way-to-use-giba-s-features-v2
def _get_leak(df, cols, lag=0):
    d1 = df[cols[:-lag-2]].apply(tuple, axis=1).to_frame().rename(columns={0: 'key'})
    d2 = df[cols[lag+2:]].apply(tuple, axis=1).to_frame().rename(columns={0: 'key'})
    d2['pred'] = df[cols[lag]]
    #d2 = d2[d2.pred != 0] ### to make output consistent with Hasan's function
    d3 = d2[~d2.duplicated(['key'], keep=False)]
    return d1.merge(d3, how='left', on='key').pred.fillna(0)


def compiled_leak_result():
    max_nlags = len(cols) - 2
    train_leak = train[["ID", "target"] + cols]
    train_leak["compiled_leak"] = 0
    train_leak["nonzero_mean"] = train[transact_cols].apply(
        lambda x: np.expm1(np.log1p(x[x != 0]).mean()), axis=1
    )

    scores = []
    leaky_value_counts = []
    leaky_value_corrects = []
    leaky_cols = []

    for i in range(max_nlags):
        c = "leaked_target_" + str(i)

        print('Processing lag', i)
        train_leak[c] = _get_leak(train_leak, cols, i)

        leaky_cols.append(c)
        train_leak = train.join(
            train_leak.set_index("ID")[leaky_cols + ["compiled_leak", "nonzero_mean"]],
            on="ID", how="left"
        )[["ID", "target"] + cols + leaky_cols + ["compiled_leak", "nonzero_mean"]]
        zeroleak = train_leak["compiled_leak"] == 0
        train_leak.loc[zeroleak, "compiled_leak"] = train_leak.loc[zeroleak, c]
        leaky_value_counts.append(sum(train_leak["compiled_leak"] > 0))
        _correct_counts = sum(train_leak["compiled_leak"] == train_leak["target"])
        leaky_value_corrects.append(_correct_counts / leaky_value_counts[-1])
        print("Leak values found in train", leaky_value_counts[-1])
        print(
            "% of correct leaks values in train ",
            leaky_value_corrects[-1]
        )
        tmp = train_leak.copy()
        tmp.loc[zeroleak, "compiled_leak"] = tmp.loc[zeroleak, "nonzero_mean"]
        scores.append(np.sqrt(mean_squared_error(y, np.log1p(tmp["compiled_leak"]).fillna(14.49))))
        print(
            'Score (filled with nonzero mean)',
            scores[-1]
        )
    result = dict(
        score=scores,
        leaky_count=leaky_value_counts,
        leaky_correct=leaky_value_corrects,
    )
    return train_leak, result

train_leak, result = compiled_leak_result()

result = pd.DataFrame.from_dict(result, orient='columns')

result.to_csv('train_leaky_stat.csv', index=False)

best_score = np.min(result['score'])
best_lag = np.argmin(result['score'])

def rewrite_compiled_leak(leak_df, lag):
    leak_df["compiled_leak"] = 0
    for i in range(lag):
        c = "leaked_target_"+str(i)
        zeroleak = leak_df["compiled_leak"]==0
        leak_df.loc[zeroleak, "compiled_leak"] = leak_df.loc[zeroleak, c]
    return leak_df

leaky_cols = [c for c in train_leak.columns if 'leaked_target_' in c]
train_leak = rewrite_compiled_leak(train_leak, best_lag)
train_leak[['ID']+leaky_cols+['compiled_leak']].head()

train_res = train_leak[leaky_cols+['compiled_leak']].replace(0.0, np.nan)
train_res.to_csv('train_leak.csv', index=False)


def compiled_leak_result_test(max_nlags):
    test_leak = test[["ID", "target"] + cols]
    test_leak["compiled_leak"] = 0
    test_leak["nonzero_mean"] = test[transact_cols].apply(
        lambda x: np.expm1(np.log1p(x[x != 0]).mean()), axis=1
    )

    scores = []
    leaky_value_counts = []
    # leaky_value_corrects = []
    leaky_cols = []

    for i in range(max_nlags):
        c = "leaked_target_" + str(i)

        print('Processing lag', i)
        test_leak[c] = _get_leak(test_leak, cols, i)

        leaky_cols.append(c)
        test_leak = test.join(
            test_leak.set_index("ID")[leaky_cols + ["compiled_leak", "nonzero_mean"]],
            on="ID", how="left"
        )[["ID", "target"] + cols + leaky_cols + ["compiled_leak", "nonzero_mean"]]
        zeroleak = test_leak["compiled_leak"] == 0
        test_leak.loc[zeroleak, "compiled_leak"] = test_leak.loc[zeroleak, c]
        leaky_value_counts.append(sum(test_leak["compiled_leak"] > 0))
        # _correct_counts = sum(train_leak["compiled_leak"]==train_leak["target"])
        # leaky_value_corrects.append(_correct_counts/leaky_value_counts[-1])
        print("Leak values found in test", leaky_value_counts[-1])
        # print(
        #    "% of correct leaks values in train ",
        #    leaky_value_corrects[-1]
        # )
        # tmp = train_leak.copy()
        # tmp.loc[zeroleak, "compiled_leak"] = tmp.loc[zeroleak, "nonzero_mean"]
        # scores.append(np.sqrt(mean_squared_error(y, np.log1p(tmp["compiled_leak"]).fillna(14.49))))
        # print(
        #    'Score (filled with nonzero mean)',
        #    scores[-1]
        # )
    result = dict(
        # score=scores,
        leaky_count=leaky_value_counts,
        # leaky_correct=leaky_value_corrects,
    )
    return test_leak, result

test_leak, test_result = compiled_leak_result_test(max_nlags=38)

test_result = pd.DataFrame.from_dict(test_result, orient='columns')

test_result.to_csv('test_leaky_stat.csv', index=False)

test_leak = rewrite_compiled_leak(test_leak, best_lag)

test_res = test_leak[leaky_cols+['compiled_leak']].replace(0.0, np.nan)
test_res.to_csv('test_leak.csv', index=False)

test_leak.loc[test_leak["compiled_leak"]==0, "compiled_leak"] = test_leak.loc[test_leak["compiled_leak"]==0, "nonzero_mean"]

#submission
sub = test[["ID"]]
sub["target"] = test_leak["compiled_leak"]
sub.to_csv(f"baseline_sub_lag_{best_lag}.csv", index=False)




