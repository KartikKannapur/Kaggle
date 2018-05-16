# #Python Libraries
import pandas as pd
import time
import gc
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb


start_time = time.time()

df_train = pd.read_csv("data/train.csv", skiprows=160000000, nrows=21000000)
df_train.columns = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'attributed_time', 'is_attributed']
print("Time taken to load the data: {} ".format(time.time() - start_time))
z
y = df_train['is_attributed']
df_train.drop(['is_attributed', 'attributed_time'], axis=1, inplace=True)



df_train['click_time'] = pd.to_datetime(df_train['click_time']).dt.date
df_train['click_time'] = df_train['click_time'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)

df_test = pd.read_csv("data/test.csv")
gc.collect()

df_test['click_time'] = pd.to_datetime(df_test['click_time']).dt.date
df_test['click_time'] = df_test['click_time'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)



sub = pd.DataFrame()
sub['click_id'] = df_test['click_id']
df_test.drop('click_id', axis=1, inplace=True)
gc.collect()

start_time = time.time()

params = {'eta': 0.1, 
          'max_depth': 4, 
          'subsample': 0.9, 
          'colsample_bytree': 0.7, 
          'colsample_bylevel':0.7,
          'min_child_weight':100,
          'alpha':4,
          'objective': 'binary:logistic', 
          'eval_metric': 'auc', 
          'random_state': 42, 
          'silent': True}
          
x1, x2, y1, y2 = train_test_split(df_train, y, test_size=0.1, random_state=42)

watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]
model = xgb.train(params, xgb.DMatrix(x1, y1), 270, watchlist, maximize=True, verbose_eval=100)

print("Time taken for XGBoost Training: {} ".format(time.time() - start_time))

sub['is_attributed'] = model.predict(xgb.DMatrix(df_test), ntree_limit=model.best_ntree_limit)
sub.to_csv('model_3_xbg_submission_v4.csv',index=False)