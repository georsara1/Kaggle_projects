
#Import needed libraries
import pandas as pd
from pandas import pivot_table
import numpy as np


#Import dataset
reserve_air = pd.read_csv('air_reserve.csv')
reserve_hpg = pd.read_csv('hpg_reserve.csv')

submission_df = pd.read_csv('sample_submission.csv')
store_id_relation = pd.read_csv('store_id_relation.csv')

#Data pre-processing
submission_ids = submission_df.id.str[:20]


#Change the order of columns in store_id_relation
store_id_relation = store_id_relation[['hpg_store_id','air_store_id']]

#Create dictionary to replace hpg with air restaurants
restaurant_dictionary = pd.Series(store_id_relation.air_store_id.values,index=store_id_relation.hpg_store_id).to_dict()
reserve_hpg['hpg_store_id'] = reserve_hpg['hpg_store_id'].map(restaurant_dictionary)

#Keep only non-null values of ID column (only 'air' restaurants)
reserve_hpg = reserve_hpg[reserve_hpg.hpg_store_id.notnull()]



