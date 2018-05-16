# Diary of a Data Scientist

### Notes, Questions and Hypothesis:

1. The `train_sample.csv` contains 100,000 randomly-selected rows from the training data. `attributed_time` is the only column with missing data. In the `train_sample.csv` dataset, 99.773% of the data was missing! `attributed_time` represents the time of the app download, after the user clicked on the ad. 
Hypothesis: Could this imply that in 99.773% of transactions, the app was not downloaded? 
Result: Yes. Only 227 of 100,000 transactions had 1's; indicating that the app was downloaded. This is certainly a class imbalance problem!

2. `ip` seems to be a multimodal distribution.


Q. ip addresses of the transactions that lead to an app download? Frequency of clicks - total, prior to app download, post the app download? Patterns?

### Feautures:
ip, frequency_encoding, greater_than_1_counts, rank_transformation
app, frequency_encoding, rank_transformation
device, os, device_os combination, frequency_encoding, rank_transformation
channel_id, frequency_encoding, rank_transformation
click_time - datetime, hour, minute, second, AM/PM
download_time - If NAN, download_time = 0, download_time_diff = 0; Else download_time = attributed_time, download_time_diff = (attributed_time - latest(click_time))

Completed

click_time - datetime, hour, minute, second, AM/PM
download_time - If NAN, download_time = 0, download_time_diff = 0; Else download_time = attributed_time, download_time_diff = (attributed_time - latest(click_time))

y - is_attributed

### Submissions:

1. 03-11-2018 | Model: XGBoost - Standard Parameters | Rank: 338 | AUC: 0.9116
2. 03-11-2018 | Model: XGBoost - Standard Parameters | Rank: 331 | AUC: 0.9218
3. 03-11-2018 | Model: XGBoost - Standard Parameters | Rank: 305 | AUC: 0.9484
4. 03-12-2018 | Model: XGBoost - Standard Parameters | Rank: 318 | AUC: 0.9501
5. 03-12-2018 | Model: XGBoost - Standard Parameters, Kaggle LB Training Subset, Custom Features | Rank: 272 | AUC: 0.9529
6. 03-12-2018 | Model: Ensemble_1 | Rank: 128 | AUC: 0.9608
7. 03-13-2018 | Model: Ensemble_2 | Rank: 164 | AUC: 0.9619
8. 03-13-2018 | Model: Ensemble_3 - Sub Mix | Rank: 86 | AUC: 0.9639
9. 03-13-2018 | Model: Ensemble_3 - Sub Mix Kartik - 'xgb  ':  .02,'ftrl1':  .04, 'nn   ':  .04, 'lgb  ':  .60, 'usam ':  .05, 'means':  .10, 'ftrl2':  .06| Rank: 63 | AUC: 0.9640
10. 03-13-2018 | Model: Ensemble_4 - Sub Mix Kartik - 'xgb  ':  .10,'ftrl1':  .04, 'nn   ':  .05, 'lgb  ':  .60, 'usam ':  .05, 'means':  .7, 'ftrl2':  .09| Rank: 69 | AUC: 0.9642
11. 03-13-2018 | Model: Ensemble_5 - Sub Mix Kartik - Same as above without the Neural Network | Rank: 53 | AUC: 0.9651
12. 03-13-2018 | Model: Ensemble_6 - Sub Mix Kartik - Same as above with modified weights | Rank: 50 | AUC: 0.9653
13. 03-13-2018 | Model: LightGBM with in-built class imablance handling | Rank: ~400 | AUC: 0.97 | Poor performance on test data. Likely overfit.
14. 03-14-2018 | Model: Blended Model with Parameter Tuning | Rank: 10 | AUC: 0.9678 
15. 03-16-2018 | Model: Blended Model with Parameter Tuning | Rank: 22 | AUC: 0.9682


### 03-08-2018

1. Downloaded and unzipped the data.
2. Checking for class imbalance properties.
3. `ip` address is not unique across all the transactions.
4. Creating a primary list of features
5. Distribution of `ip` addresses

### 03-09-2018

1. Reduced the size of the train dataset by 20%, using the command: sed  s/2017-11-//g < train.csv > train_reduced.csv

### 03-11-2018

1. Model submissions with various training dataset sizes
2. Batch training of XGBoost

### 03-12-2018

1. Used the Kaggle LB Training data subset, to build the model + My custom generated features. Model submission - Rank: 272 | AUC: 0.9529
2. Ensemble predictions (Kaggle LB Training data subset, to build the model + My custom generated features) + (Kaggle - App Mean Model) with equal weights - Rank: 128 | AUC: 0.9608

### 03-13-2018

1. Model Blending - Experiments

### 03-14-2018

1. Attempting model blending using GDM + RF + LR. Class imbalance not yet taken into account. SMOTE to be inplemented with model blending next. Need to contrast LightGBM (with in-built imbalance handling) vs. (LightGBM + SMOTE).
2. Kartik: Model Blending - Experiments - Top 10 on the Public Leaderboard 

### 03-[15-16]-2018

1. Kartik: Model Blending - Experiments - Top 10 on the Public Leaderboard 
2. Kartik: Model Blending - Experiments - Top 30 on the Public Leaderboard 