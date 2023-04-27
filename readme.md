# CA-Projects-template-submission-repo
*The objective of this project is to analyze the sales data of a store and build a regression model to predict future sales. The data for this project is obtained from a retail store that sells various products, such as food, clothing, electronics, and home appliances.*

## Summary
| Code      | Name        | Published Article |  Deployed App |
|-----------|-------------|:-------------:|------:|
| LP2 | Store Sales -- Time Series Forecasting |  [Best article of the world](https://medium.com/@acheampongfrancis95/predictive-analytics-for-grocery-sales-forecasting-a-case-study-of-favorita-stores-b9c7e89549fe) | [Best app of the world](https://app.powerbi.com/groups/me/reports/edca594b-66bb-4578-b416-e93d01c74ddc/ReportSection) |

## Project Description
This is a time series forecasting problem. In this project, you will predict store sales on data from Corporation Favorita, a large Ecuadorian-based grocery retailer. 
Specifically, we will build a machine learning model that accurately predicts the unit sales for thousands of items sold at different Favorita stores.
The Favorita Grocery Sales Forecasting dataset is a fascinating collection of data that provides a great opportunity for analysis and prediction. In this article, we will take a deep dive into the dataset, explore its various attributes, and analyze the sales patterns to build a robust sales forecasting model and answer some pertinent question on the dataset.

## Setup


from scipy import stats
from scipy.stats import pearsonr

from scipy.stats import chi2_contingency
import plotly.express as px

from scipy.stats import ttest_ind
from datetime import datetime


# Importing the relevant libraries
import IPython.display
import json
import squarify
%matplotlib inline
import missingno as msno
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
color = sns.color_palette()

# D3 modules
from IPython.core.display import display, HTML, Javascript
from string import Template

# Other packages
import os
import re
#display all columns and rows 
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_squared_log_error

import warnings
warnings.filterwarnings('ignore')

## App Executio
...

## Author
ACHEAMPONG Francis
