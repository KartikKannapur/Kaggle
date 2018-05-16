# #Python Libraries
import pandas as pd
import time
import gc
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb


start_time = time.time()

df_train_features = pd.read_csv("data/train.csv", skiprows=160000000, nrows=21000000)
df_train_features.columns = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'attributed_time', 'is_attributed']
print("Time taken to load the data: {} ".format(time.time() - start_time))

y = df_train_features['is_attributed']
df_train_features.drop(['is_attributed', 'attributed_time'], axis=1, inplace=True)


# #Feature - Frequency Encoding
start_time = time.time()
arr_feature = ["ip", "app", "device", "os", "channel"]

for ele in arr_feature:
     df_train_features[ele + "_freq_encoding"] = df_train_features.groupby(ele)[ele].transform('count')
     gc.collect()

print("Time taken for Feature Engineering - Frequency Encoding: {} ".format(time.time() - start_time))
# -------------------------------------------------#
# #Feature - Datetime Extraction
start_time = time.time()

df_train_features["click_time"] = pd.to_datetime(df_train_features["click_time"])

df_train_features["click_time_hour"] = df_train_features["click_time"].dt.hour.astype(int)
df_train_features["click_time_minute"] = df_train_features["click_time"].dt.minute.astype(int)
gc.collect()
df_train_features["click_time_dayofweek"] = df_train_features["click_time"].dt.dayofweek.astype(int)
df_train_features["click_time"] = df_train_features["click_time"].dt.day.astype(int)
gc.collect()
print("Time taken for Feature Engineering - Datetime Extraction: {} ".format(time.time() - start_time))
# -------------------------------------------------#
# #Feature - Combine device and os
start_time = time.time()
df_train_features['device_os'] = df_train_features['device'].astype(str) + "_" +df_train_features['os'].astype(str)
df_train_features["device_os"] = df_train_features["device_os"].astype('category').cat.codes
print("Time taken for Feature Engineering - Combine device and os: {} ".format(time.time() - start_time))
gc.collect()

# #Feature - Combine app and os
start_time = time.time()
df_train_features['app_os'] = df_train_features['app'].astype(str) + "_" +df_train_features['os'].astype(str)
df_train_features["app_os"] = df_train_features["app_os"].astype('category').cat.codes
print("Time taken for Feature Engineering - Combine app and os: {} ".format(time.time() - start_time))
gc.collect()

# #Feature - Combine app and device_os
start_time = time.time()
df_train_features['app_deviceos'] = df_train_features['app'].astype(str) + "_" +df_train_features['device_os'].astype(str)
df_train_features["app_deviceos"] = df_train_features["app_deviceos"].astype('category').cat.codes
print("Time taken for Feature Engineering - Combine app and device_os: {} ".format(time.time() - start_time))
gc.collect()

# #Feature - Combine app and channel
start_time = time.time()
df_train_features['app_channel'] = df_train_features['app'].astype(str) + "_" +df_train_features['channel'].astype(str)
df_train_features["app_channel"] = df_train_features["app_channel"].astype('category').cat.codes
print("Time taken for Feature Engineering - Combine app and channel: {} ".format(time.time() - start_time))
gc.collect()

# #Feature - Combine os and channel
start_time = time.time()
df_train_features['os_channel'] = df_train_features['os'].astype(str) + "_" +df_train_features['channel'].astype(str)
df_train_features["os_channel"] = df_train_features["os_channel"].astype('category').cat.codes
print("Time taken for Feature Engineering - Combine os and channel: {} ".format(time.time() - start_time))
gc.collect()
# -------------------------------------------------#
# #Feature - Rank Transformation
start_time = time.time()
df_train_features['ip_rank'] = df_train_features['ip_freq_encoding'].rank(method='dense')
df_train_features['app_rank'] = df_train_features['app_freq_encoding'].rank(method='dense')
df_train_features['device_rank'] = df_train_features['device_freq_encoding'].rank(method='dense')
df_train_features['os_rank'] = df_train_features['os_freq_encoding'].rank(method='dense')
df_train_features['channel_rank'] = df_train_features['channel_freq_encoding'].rank(method='dense')
gc.collect()
print("Time taken for Feature Engineering - Rank Transformation: {} ".format(time.time() - start_time))



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
          
x1, x2, y1, y2 = train_test_split(df_train_features, y, test_size=0.1, random_state=42)

watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]
model = xgb.train(params, xgb.DMatrix(x1, y1), 270, watchlist, maximize=True, verbose_eval=100)

print("Time taken for XGBoost Training: {} ".format(time.time() - start_time))


df_test = pd.read_csv("data/test.csv")
gc.collect()

df_test['click_time'] = pd.to_datetime(df_test['click_time']).dt.date
df_test['click_time'] = df_test['click_time'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)



sub = pd.DataFrame()
sub['click_id'] = df_test['click_id']
df_test.drop('click_id', axis=1, inplace=True)
gc.collect()


# #Feature - Frequency Encoding
start_time = time.time()
arr_feature = ["ip", "app", "device", "os", "channel"]

for ele in arr_feature:
     df_test[ele + "_freq_encoding"] = df_test.groupby(ele)[ele].transform('count')
     gc.collect()

print("Time taken for Feature Engineering - Frequency Encoding: {} ".format(time.time() - start_time))

# #Feature - Convert to Datetime
df_test["click_time"] = pd.to_datetime(df_test["click_time"])
print("Time taken for Feature Engineering - Convert to Datetime: {} ".format(time.time() - start_time))
gc.collect()

# #Feature - Datetime Extraction
start_time = time.time()
df_test["click_time_hour"] = df_test["click_time"].dt.hour.astype(int)
df_test["click_time_minute"] = df_test["click_time"].dt.minute.astype(int)
df_test["click_time_dayofweek"] = df_test["click_time"].dt.dayofweek.astype(int)
df_test["click_time"] = df_test["click_time"].dt.day.astype(int)
gc.collect()
print("Time taken for Feature Engineering - Datetime Extraction: {} ".format(time.time() - start_time))

# #Feature - Combine device and os
start_time = time.time()
df_test['device_os'] = df_test['device'].astype(str) + "_" +df_test['os'].astype(str)
df_test["device_os"] = df_test["device_os"].astype('category').cat.codes

print("Time taken for Feature Engineering - Combine device and os: {} ".format(time.time() - start_time))
gc.collect()

# #Feature - Combine app and os
start_time = time.time()
df_test['app_os'] = df_test['app'].astype(str) + "_" +df_test['os'].astype(str)
df_test["app_os"] = df_test["app_os"].astype('category').cat.codes
print("Time taken for Feature Engineering - Combine app and os: {} ".format(time.time() - start_time))
gc.collect()

# #Feature - Combine app and device_os
start_time = time.time()
df_test['app_deviceos'] = df_test['app'].astype(str) + "_" +df_test['device_os'].astype(str)
df_test["app_deviceos"] = df_test["app_deviceos"].astype('category').cat.codes
print("Time taken for Feature Engineering - Combine app and device_os: {} ".format(time.time() - start_time))
gc.collect()

# #Feature - Combine app and channel
start_time = time.time()
df_test['app_channel'] = df_test['app'].astype(str) + "_" +df_test['channel'].astype(str)
df_test["app_channel"] = df_test["app_channel"].astype('category').cat.codes
print("Time taken for Feature Engineering - Combine app and channel: {} ".format(time.time() - start_time))
gc.collect()

# #Feature - Combine os and channel
start_time = time.time()
df_test['os_channel'] = df_test['os'].astype(str) + "_" +df_test['channel'].astype(str)
df_test["os_channel"] = df_test["os_channel"].astype('category').cat.codes
print("Time taken for Feature Engineering - Combine os and channel: {} ".format(time.time() - start_time))
gc.collect()

start_time = time.time()

# #Feature - Rank Transformation
df_test['ip_rank'] = df_test['ip_freq_encoding'].rank(method='dense')
df_test['app_rank'] = df_test['app_freq_encoding'].rank(method='dense')
df_test['device_rank'] = df_test['device_freq_encoding'].rank(method='dense')
df_test['os_rank'] = df_test['os_freq_encoding'].rank(method='dense')
df_test['channel_rank'] = df_test['channel_freq_encoding'].rank(method='dense')

print("Time taken for Feature Engineering - Rank Transformation: {} ".format(time.time() - start_time))
gc.collect()


sub['is_attributed'] = model.predict(xgb.DMatrix(df_test), ntree_limit=model.best_ntree_limit)
sub.to_csv('model_3_xbg_submission_v4.csv',index=False)