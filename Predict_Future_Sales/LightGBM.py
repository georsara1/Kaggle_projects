import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


train_df = pd.read_csv('sales_train_v2.csv')
test_df = pd.read_csv('test.csv')
submission_df = pd.read_csv('sample_submission.csv')

train_df.item_price.hist(bins=100)

print(train_df.head())