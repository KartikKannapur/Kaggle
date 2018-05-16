# #Python Libraries
import pandas as pd
import time
import gc
from sklearn.model_selection import train_test_split
import xgboost as xgb

my_model_name = "model_1.model"
# #Prepare the Test Dataset
df_test = pd.read_csv("data/test_100000.csv")

submission = pd.DataFrame()
submission['click_id'] = df_test['click_id']
gc.collect()
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

start_time = time.time()

# #Feature - Rank Transformation
df_test['ip_rank'] = df_test['ip_freq_encoding'].rank(method='dense')
df_test['app_rank'] = df_test['app_freq_encoding'].rank(method='dense')
df_test['device_rank'] = df_test['device_freq_encoding'].rank(method='dense')
df_test['os_rank'] = df_test['os_freq_encoding'].rank(method='dense')
df_test['channel_rank'] = df_test['channel_freq_encoding'].rank(method='dense')

print("Time taken for Feature Engineering - Rank Transformation: {} ".format(time.time() - start_time))
gc.collect()

start_time = time.time()

xgb_booster = xgb.Booster()
model_iter = xgb_booster.load_model(my_model_name)
submission['is_attributed'] = model_iter.predict(xgb.DMatrix(df_test), ntree_limit=model.best_ntree_limit)
submission.to_csv('model_1_xbg_submission_v2.csv',index=False)