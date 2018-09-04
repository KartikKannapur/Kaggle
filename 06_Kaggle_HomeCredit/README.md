# Diary of a Data Scientist

### Notes, Questions and Hypothesis:

1. 

### Feautures:

### Hyperparameters for XGBoost - Current Best 0.777:

LB: 0.779
xgb_params = {
 'alpha': 3.160842634951819,
 'booster ': 'gbtree',
 'colsample_bytree': 0.7,
 'eta': 0.1604387053222455,
 'eval_metric': 'auc',
 'gamma': 0.6236454630290655,
 'lambda': 4.438488456929287,
 'max_depth': 4,
 'min_child_weight': 9,
 'n_thread': -1,
 'objective': 'binary:logistic',
 'seed': 42,
 'silent': 0,
 'subsample': 0.8
}

### Submissions:

MICE + XGBoost Starter | 0.729
XGBoost Starter | 0.729
XGBoost on entire Training set | 0.738
XGBoost with manual Hyperparameter Tuning on entire Training set v0 | 0.742
XGBoost with manual Hyperparameter Tuning on entire Training set v1 | 0.743
XGBoost with manual Hyperparameter Tuning on entire Training set v2 | 0.746
XGBoost df_application + df_installments_payments | 0.756
XGBoost df_application + df_installments_payments + df_pos_cash_balance | 0.765
XGBoost df_application + df_installments_payments + df_pos_cash_balance + df_previous_application | 0.772
XGBoost with all the datasets | 0.773
XGBoost with all the datasets + Hyperparameter Optimization with Hyperopt (Notebook 08) | 0.779
LightGBM with all datasets + new features | 0.785, 0.788, 0.790
LightGBM(best model 0.790) + Genetic Programming(best model 0.785) - blender | 0.795


### 05-20-2018
1. Basic EDA
2. Decided to fork a LigthGBM stater to get the pipeline running
3. First submission gave me an AUC of 0.739

### 05-21-2018
1. In order to verify and exclude the obvious cases, I built two models - 1) Removing the rows where ALL the observations were missing - Result: No such row existed. This implies that we can run kNN imputation or similar missing value imputation techniques; 2) Removing the rows from the training set where ATLEAST one observation in a row was missing. This reduced the data significantly and the model had poor performance. In conclusion, missing value imputation needs to be performed.
2. Trying out the library - 'fancyimpute'. 
3. Due to setup issues, had to install 'ecos' via a wheel file, prior to installing fancyimpute.

### 05-22-2018
1. Missing value imputation using fancyimpute - MICE
2. XGBoost Starter post missing value imputation - 0.729
3. To test my hypothesis - did missing value imputation via MICE cause my model to perform better, I ran a model with the original data without any imputation. This gave me an accuracy of 0.729 as well. This implies that either there is a bug in my code or that XGBoost performs missing value imputation the same way as fancyimpute does with MICE. 

### 05-23-2018
1. NOTE: Missing values are handled by default in XGBoost.
2. The distribution of the TARGET variable i.e. the variable to predict, seems to have a good spread. No class imbalance to deal with here.
3. Implemented PCA on the train and test datasets and ran the XGBoost model. I got an accuracy of 0.579. This is probably because there is no correlation between the variables/features in the dataset. The pricipal compoenents that is generated are poor and therefore the model performs poorly as well
4. Hyperparameter tuning on XGBoost does not seem to be improving my scores as well.

### 05-24-2018
1. The mistake I was doing, was running my code only on 70% of the test set. I ran it on the entire thing and my accuracy went up to 0.738.
2. Hyperparameter optimization helps!!! I manually tuned a few paramters and my score went up to 0.742

### 05-25-2018
1. Hyperparameter optimization - to change the existing parameters. This gave me an improvement of 0.001. Although this is minor, it certainly a good sign that hyperparameter optimization is helpful.
2. Newer hyperparameters obtained via `hyperopt`, lead me to 0.736
3. I increased 270 and 1000, 1500 and the model overfit. Poor performance on the leaderboard. Probably a good metric here is the `n_estimators` parameter in XGBoost, that specifies early stopping. It dosen't make sense to set parameters way past this! This should be able to handle overfitting.
4. Performed feature extraction on the `df_installments_payments` table and added that to the model. This gave me an accuracy of 0.756

### 05-26-2018
1. Performed feature extraction on the `df_pos_cash_balance` table and added that to the model. This gave me an accuracy of 0.765
2. Performed feature extraction on the `df_previous_application` table and added that to the model. This gave me an accuracy of 0.772

### 05-26-2018
1. Performed feature extraction on the `df_bureau` and `df_bureau_balance` table and added that to the model. This gave me an accuracy of 0.773

### 05-28-2018
1. Note: While combining the datasets `df_bureau` and `df_bureau_balance`,
I have aggreated over each SK_ID_BUREAU of `df_bureau_balance`; and since each SK_ID_CURR in `df_bureau` corresponds to mutliple values of SK_ID_BUREAU, I have taken the mean of the means, max of the max and min of the min.
2. Performed Hyperparameter Optimization with Hyperopt, for XGBoost with all the datasets. This gave me an accuracy of 0.779
3. WOW - Interestingly enough, I changed max_depth from 4 to 5 and that caused a drop in my accuracy from 0.779 to 0.773

### 05-30-2018
1. Adding these features got the accuracy down from 0.779 to 0.775 - 
'K_CREDIT_ACTIVE_ACTIVE', 'K_CREDIT_ACTIVE_BADDEBT', 'K_CREDIT_ACTIVE_CLOSED', 'K_CREDIT_ACTIVE_SOLD'
2. 'K_AMT_DIFF' seems to be a good feature (source: kaggle discussions)
3. New Features - Added mean values to `df_pos_cash_balance` + New Features for the main `df_application_train` table: K_APP_CREDIT_TO_INCOME_RATIO, K_APP_ANNUITY_TO_INCOME_RATIO, K_APP_ANNUITY_TO_CREDIT_RATIO, K_APP_GOODSPRICE_TO_CREDIT_RATIO, K_APP_INCOME_EDUCATION, K_APP_INCOME_EDUCATION_FAMILY, K_DAYS_BIRTH_TO_EMPLOYED_RATIO + Removed all the created features from point 1. This gave me an accuracy of 0.785

### 05-31-2018
1. Added the code for the LightGBM Model. Model + Stratified 5-fold CV
2. `df_installments_payments` table seems to have important features. I have added the following features - K_INST_DAYS_DIFF_TO_COUNT_RATIO
K_NUM_INSTALMENT_NUMBER_SUM_TO_COUNT_RATIO, K_DAYS_ENTRY_PAYMENT_MAX, K_DAYS_ENTRY_PAYMENT_VAR, K_AMT_PAYMENT_MEAN, K_AMT_PAYMENT_MAX, K_AMT_PAYMENT_MIN, K_AMT_PAYMENT_VAR.This gave me an accuracy of 0.788
3. Downloaded and setup MongoDB + Studio3T Client

### 06-01-2018
1. Added new features - K_CNT_INSTALMENT_MATURE_CUM_MAX_TO_COUNT_RATIO, K_CREDIT_UTILIZATION_VAR
2. Removed K_* generated features that had a feature importance score of zero.
This gave me an accuracy of 0.790

### 06-03-2018
1. K_APP_GOODSPRICE_TO_INCOME_RATIO and K_APP_GOODSPRICE_TO_ANNUITY_RATIO reduced the accuracy of the model on the LB
2. Different combinations of features - K_DAYS_BIRTH_TO_REGISTRATION_RATIO, K_DAYS_EMPLOYED_TO_REGISTRATION_RATIO with the ones above. None of them improved my model
3. I also tried `categorical_feature` encoding while calling the ligthgbm model and this did not help as well.
4. Variance based approach to feature selection does not help either.
5. I set max_bin as a parameter to a really large value and that reduced my score to 0.783. I have kind of hit a plateau now, not really sure on how to progress.

### 06-05-2018, 06-06-2018, 06-07-2018
1. Tried running the data via DataRobot; best model had an AUC of 0.787
2. First attempt at Model Blending - AUC 0.795. Blended with a genetic programming model
3. As in all Kaggle competitions (and all machine learning problems, for that matter), the most important first step is to get a validation set-up that matches the test set.

### 06-13-2018
1. Decreased the learning_rate from 0.12 to 0.02 gradually. This made my CV score exactly equal to my LB score.

### 06-15-2018, 06-16-2018, 06-17-2018
1. Added the following features -  K_APP_GOODSPRICE_TO_INCOME_RATIO, K_APP_GOODSPRICE_TO_ANNUITY_RATIO, DAYS_LAST_DUE, DAYS_TERMINATION, DAYS_FIRST_DRAWING, DAYS_FIRST_DUE, DAYS_LAST_DUE_1ST_VERSION, DAYS_EMPLOYED, K_APP_CREDITTOINCOME_TO_DAYSEMPLOYED_RATIO, K_APP_ANNUITYTOINCOME_TO_DAYSEMPLOYED_RATIO, K_APP_ANNUITYTOCREDIT_TO_DAYSEMPLOYED_RATIO, K_APP_INCOME_PER_FAMILY, 'K_NAME_CONTRACT_STATUS_APPROVED', 'K_NAME_CONTRACT_STATUS_REFUSED',
2. Single model LightGBM is at 0.7909 i.e. increased by 0.009 :P


1. Add K_AMT_ANNUITY_VAR as a feature + minor variations to hyperparameters
2. Single model LightGBM is at LB 0.792



### Next Steps:
###### Feature Generation: Sum, Difference, Ratio, Product, Mean, Min, Max, Var, Range

using median instead of mean


One hot encoding instead of label encoding

As I have written, I remove all of standard features excluding ext_sources and add 17 more to achieve 0.7572 score on CV. = https://www.kaggle.com/c/home-credit-default-risk/discussion/57958#337671





1. Combine CREDIT_ACTIVE category + CREDIT_CURRENCY + CREDIT_TYPE
2. New features for pos_cash_balance_table - NAME_CONTRACT_STATUS
0. Features for df_credit_balance

How much did the Customer load a Credit line?
How many times did the Customer miss the minimum payment?
What is the average number of days did Customer go past due date?
What fraction of minimum payments were missed?
CUSTOMER BEHAVIOUR PATTERNS

Cash withdrawals VS Overall Spending ratio
Average number of drawings per customer - Total Drawings / Number of Drawings

https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html

This means that recent data is more relevant than old data.

https://github.com/slundberg/shap

Can we identify some of the anonimized features?

Confusion matrix for analysis?

https://www.kaggle.com/c/donorschoose-application-screening/kernels?sortBy=scoreDescending&group=everyone&pageSize=20&competitionId=8426
https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion
https://www.kaggle.com/c/allstate-claims-severity
https://www.kaggle.com/c/predicting-red-hat-business-value

    + AVERAGE NUMBER OF DAYS BETWEEN SUCCESSIVE PAST APPLICATIONS FOR EACH CUSTOMER
    INTERPRETING CREDIT_DAYS_ENDDATE
NEGATIVE VALUE - Credit date was in the past at time of application( Potential Red Flag !!! )
POSITIVE VALUE - Credit date is in the future at time of application ( Potential Good Sign !!!!)

+ The Ratio of Total Debt to Total Credit for each Customer
+ OVERDUE OVER DEBT RATIO
+ AVERAGE NUMBER OF LOANS PROLONGED

http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html

https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/kernels

3.  FTRL
    - https://www.kaggle.com/anttip/talkingdata-wordbatch-fm-ftrl-lb-0-9769
    - https://www.kaggle.com/titericz/giba-darragh-ftrl-rerevisited
    - https://github.com/anttttti/Wordbatch
    - https://github.com/alexeygrigorev/libftrl-python
    - https://github.com/fmfn/FTRLp



Genetic Programming for feature engineering


Check if missing values have been encoded as other values in any of the tables like - 365243
