# #Python Libraries
import pandas as pd
import time
import gc
from sklearn.model_selection import train_test_split
import xgboost as xgb

# #Load the training data
chunksize = 9000000
iteration = 0
my_model_name = "model_1.model"
for chunk in pd.read_csv("data/train.csv", chunksize=chunksize):
     start_time = time.time()
     df_train_features = chunk
     print("Time taken to load the data: {} ".format(time.time() - start_time))
     # -------------------------------------------------#
     # #For computational purposes only
     y = df_train_features['is_attributed']
     df_train_features.drop(['is_attributed'], axis=1, inplace=True)

     df_train_features.drop(["attributed_time"], axis=1, inplace=True)
     gc.collect()
     # -------------------------------------------------#
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
     # -------------------------------------------------#
     # # Model Building - XGBoost
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

     x1, x2, y1, y2 = train_test_split(df_train_features, y, test_size=0.2, random_state=42)
     gc.collect()

     watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]
     # params.update({'process_type': 'update','updater': 'refresh','refresh_leaf': True})

     if iteration == 0:
          model_iter = xgb.train(params, xgb.DMatrix(x1, y1), 270, watchlist, maximize=True, verbose_eval=100)
          # model_iter.save_model(my_model_name)
          iteration += 1

     else:
          params.update({'process_type': 'update','updater': 'refresh','refresh_leaf': True})
          model_iter = xgb.train(params, xgb.DMatrix(x1, y1), 270, watchlist, maximize=True, verbose_eval=100, xgb_model=model_iter)
     print("Time taken for XGBoost Training: {} ".format(time.time() - start_time))

     gc.collect()


# #Prepare the Test Dataset
df_test = pd.read_csv("data/test.csv")

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

submission['is_attributed'] = model_iter.predict(xgb.DMatrix(df_test), ntree_limit=model_iter.best_ntree_limit)
submission.to_csv('model_1_xbg_submission_v4.csv',index=False)