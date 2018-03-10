# #Python Libraries
import numpy as np
import scipy as sp
import pandas as pd
import statsmodels
import pandas_profiling

%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns

import os
import sys
import time
import requests
import datetime

import missingno as msno

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
# #X_trainval, X_test, y_trainval, y_test = train_test_split(X, y)
# #X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval)
from sklearn.model_selection import GridSearchCV

# #Spark
import os
import sys
spark_home = os.environ.get('SPARK_HOME', None)

if not spark_home:
    raise ValueError('SPARK_HOME environment variable is not set')

sys.path.insert(0, os.path.join(spark_home, 'python'))
sys.path.insert(0, os.path.join(spark_home, 'C:/Users/karti/Spark/spark-2.3.0-bin-hadoop2.7/python/lib/py4j-0.10.6-src.zip'))

filename=os.path.join(spark_home, 'python/pyspark/shell.py')
exec(compile(open(filename, "rb").read(), filename, 'exec'))
