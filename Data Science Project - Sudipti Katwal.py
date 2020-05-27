#!/usr/bin/env python
# coding: utf-8

# #   Final Project - Retail Data Analytics
# ## Prediction Poblem - Sales Forecasting
# 
# 
# **Data Science** -  **March 29, 2020**
# 
# #### SUDIPTI KATWAL

# ## Data Collection & Cleaning

# Data Source:
# 
# Kaggle: https://www.kaggle.com/manjeetsingh/retaildataset

# In[83]:


# Import Python packages

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from bokeh.plotting import figure, output_notebook, show
from bokeh.models import ColumnDataSource, NumeralTickFormatter, CategoricalColorMapper
from bokeh.models.tools import HoverTool
from bokeh.palettes import Spectral6, RdBu3
from bokeh.transform import factor_cmap
import statsmodels.formula.api as smf
import statsmodels.api as sm  
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf  
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from sklearn import model_selection, metrics
from sklearn.metrics import mean_squared_error
import itertools
import warnings

get_ipython().run_line_magic('matplotlib', 'inline')
output_notebook()


# The following few steps involve importing the data and exploring the dataset

# In[7]:


# Imported first file in the dataset

sales = pd.read_csv("D:/Learning/Brainstation Data Science/Project/retaildataset\sales data-set.csv")
sales.shape


# In[8]:


sales.info()


# In[9]:


sales.describe()


# In[10]:


sales.head()


# In[11]:


# Imported the second file in the dataset

stores = pd.read_csv("D:/Learning/Brainstation Data Science/Project/retaildataset\stores data-set.csv")
stores.shape


# In[12]:


stores.info()


# In[13]:


stores.describe()


# In[14]:


stores.head()


# In[94]:


# Merged sales & stores datasets into a unique dataframe using a left join with 'Stores' as the unique key

df = pd.merge(sales, stores, how = 'left', on = 'Store')
df.head()


# In[16]:


# Checked the data types for all columns in the dataset 

df.dtypes


# In[17]:


# Changed the data type for 'Date' column to datetime

df['Date'] = pd.to_datetime(df['Date'])
df.dtypes


# In[18]:


df.describe()


# In[19]:


df.info()


# In[20]:


# There are 45 unique stores, 81 Departments and 3 Types of Stores

df.nunique()


# In[21]:


# No null values found in the data

df.isnull().sum()


# In[22]:


# No duplicate values found in the data

df[['Date','Store','Dept','Weekly_Sales']].duplicated().sum()


# In[23]:


# There are 3 different 'Types' of Stores

df['Type'].unique()


# In[24]:


# Transformed the 'IsHoliday' column to int data type from boolean

df['IsHoliday'] = df['IsHoliday'].astype(int)
df.head()


# The dataset was pretty clean with no null or duplicated values.

# ## Exploratory Data Analysis

# In[25]:


# Checked the correlation between various variables in the data frame. No 2 variables seem to have a high correlation.
# The highest correlation of all vairables is found between Weekly_Sales & Size of the Store which is 0.24 (very low)

df.corr()


# In[26]:


# Plotted the correlation using seaborn in a heatmap.
# Previous finding substantiated by the heatmap - very low correlation between various variables.
# Store number & Dept number are categorical variables / labels for the purpose of analysis, since they don't represent
# numerical information that can be used for analysis

sns.heatmap(df.corr())


# In[27]:


# Plotted all vairables in a pairplot to look for linear / polynomial relationships 

sns.pairplot(df)


# In[28]:


# Used aggregation to calculate total weekly sales for all stores

df_sales_all_stores = df.groupby(by = ['Date'], as_index=False)[['Weekly_Sales']].sum()
df_sales_all_stores.head()


# In[30]:


# Plotted Date & total weekly sales for all stores (calculated above). It looks like the sales are seasonal, with higher sales
# during the Holiday period - Nov to January

source = ColumnDataSource(df_sales_all_stores)
p = figure(plot_width=900, plot_height=300, title='Total Weekly Sales for all Stores', x_axis_label='Date',
           y_axis_label='Weekly Sales', x_axis_type="datetime")

# add a line renderer and a hover tool

p.line(x='Date', y='Weekly_Sales', source=source, color='navy', alpha=0.8, line_width=2)
hover = HoverTool()
hover.tooltips = [("Totals", "@Weekly_Sales{0,0}")]
p.add_tools(hover)

p.left[0].formatter.use_scientific = False
p.yaxis.formatter=NumeralTickFormatter(format="0 a")

# show the results

show(p)


# In[31]:


# Sorted dates in ascending order to find if there exists a weekly frequency in sales data?

a = df.groupby(by='Date', as_index=False)[['Weekly_Sales']].sum()
a.sort_values(by='Date', ascending = True)
a[:5]

# Finding: The dates are random with no fixed frequency. Also, the data for January 2010 begins on Jan 10th (mid-month)
# *WE WILL USE THIS INFORMATION LATER DURING DATA MODELLING*


# In[32]:


a[-5:]

# Finding: The data for December 2012 is avaiable only until Dec 10th week (mid-month)
# *WE WILL USE THIS INFORMATION LATER DURING DATA MODELLING*


# In[33]:


# Used aggregation to calculate Average sales by store size (since there was a weak correlation between
# Weekly_Sales & Size of the store). It looks like generaly the sales increase with the size of the store, however, it's
# not a linear graph, since some smaller and medium size stores have higher sales than some larger stores.

df_avg_sales_by_store_size = df.groupby(by='Size', as_index=False)[['Weekly_Sales']].mean()


source = ColumnDataSource(df_avg_sales_by_store_size)
p = figure(plot_width=900, plot_height=300, title='Average Sales by Store Size', x_axis_label='Store Size',
           y_axis_label='Sales')

# add a vertical bar renderer
p.vbar(x='Size', top='Weekly_Sales', source=source, width = 500, color='firebrick')

# add a hover tool
hover = HoverTool()
hover.tooltips = [("Average Sales", "@Weekly_Sales{int}")]
p.add_tools(hover)

p.left[0].formatter.use_scientific = False
p.yaxis.formatter=NumeralTickFormatter(format="0 a")
p.xaxis.formatter=NumeralTickFormatter(format="0 a")

# show the results
show(p)


# In[34]:


# Using aggregation to calculate average sales by Store & Store Size

df_avg_sales_by_store = df.groupby(by = ['Store','Size'], as_index=False)[['Weekly_Sales']].mean()
df_avg_sales_by_store.sort_values(by='Weekly_Sales', ascending=False).head()

# We see that the Top 5 Stores by Average Sales are Store number 19, 3, 13, 12 and 1 (in descending order)


# In[38]:


# Plotting Average Sales by Store number

source = ColumnDataSource(df_avg_sales_by_store)
p = figure(plot_width=900, plot_height=500, title='Average Sales by Store', y_axis_label='Store Number',
           x_axis_label='Sales')

# add a horizontal bar renderer
p.hbar(y='Store', right='Weekly_Sales', source=source, height=0.5, left=0, color="navy")

# add a hover tool
hover = HoverTool()
hover.tooltips = [("Store Number", "@Store{int}"),("Average Sales", "@Weekly_Sales{0,0}"),("Store Size","@Size{0 a}")]
p.add_tools(hover)

# format the axis
p.left[0].formatter.use_scientific = False
p.yaxis.formatter=NumeralTickFormatter(format="0,0")
p.xaxis.formatter=NumeralTickFormatter(format="0 a")

# show the results
show(p)


# In[39]:


# Using aggregation calculated Average Sales by Dept and plotted it using bokeh

df_avg_sales_by_dept = df.groupby(by=['Dept'], as_index=False)[['Weekly_Sales']].mean()


source = ColumnDataSource(df_avg_sales_by_dept)
p = figure(plot_width=900, plot_height=300, title='Average Sales by Dept', x_axis_label='Dept Number', y_axis_label='Sales')

# add a horizontal bar renderer
p.vbar(x='Dept', top='Weekly_Sales', source=source, width = 0.5, color='firebrick')

# add a hover tool
hover = HoverTool()
hover.tooltips = [("Dept","@Dept{int}"),("Average Sales", "@Weekly_Sales{0,0}")]
p.add_tools(hover)

p.left[0].formatter.use_scientific = False
p.yaxis.formatter=NumeralTickFormatter(format="0 a")
p.xaxis.formatter=NumeralTickFormatter(format="0 a")

# show the results
show(p)


# In[40]:


# Using aggregation to calculate Average Sales for Holiday / Not Holiday to see the effect of Holidays on Weekly Sales

df_avg_sales_isholiday = df.groupby(by=['IsHoliday'], as_index=False)[['Weekly_Sales']].mean()
df_avg_sales_isholiday.head()


# In[41]:


# Plotting average sales against IsHoliday shows that the average sales during Holidays are higher compared to sales during
# no holiday (1 indicates Holiday and 0 indicates no holiday)

source = ColumnDataSource(df_avg_sales_isholiday)
p = figure(plot_height=200, title='Average Sales by Holiday', x_axis_label='Avarage Sales',
           y_axis_label='Is Holiday')

# add horizonal bar renderers
p.hbar(y='IsHoliday', right='Weekly_Sales', source=source, height = 0.2, left=0, color='firebrick')

# add a hover tool
hover = HoverTool()
hover.tooltips = [("Is Holiday","@IsHoliday{int}"),("Average Sales", "@Weekly_Sales{int}")]
p.add_tools(hover)

p.left[0].formatter.use_scientific = False
p.yaxis.formatter=NumeralTickFormatter(format="0 a")
p.xaxis.formatter=NumeralTickFormatter(format="0 a")

# show the results
show(p)


# In[42]:


# Using aggregation calculated Average sales and Average Store Size by Store Type

df_avg_sales_store_type = df.groupby(by=['Type'], as_index = False)[['Size','Weekly_Sales']].mean()
df_avg_sales_store_type

# Stores belonging to Type A are the largest and have the highest sales, followed by Type B and Type C (Store Size as well as
# Average Sales)


# In[43]:


# Plotted Average Sales by Store Type, with 'Average Store Size' in hover details

source = ColumnDataSource(df_avg_sales_store_type)
type = source.data['Type'].tolist()

p=figure(x_range=type, plot_height=300, toolbar_location=None, title="Average Sales by Store Type", x_axis_label='Store Type',
           y_axis_label='Average Sales')

# add vertical bars
p.vbar(x='Type', top='Weekly_Sales', width=0.5, source=source, legend='Type',
       line_color='white', fill_color=factor_cmap('Type', palette=Spectral6, factors=type))

# add hover information
hover = HoverTool()
hover.tooltips = [("Average Store Size","@Size{0,0}"),("Average Sales", "@Weekly_Sales{0,0}")]
p.add_tools(hover)

p.left[0].formatter.use_scientific = False
p.yaxis.formatter=NumeralTickFormatter(format="0 a")
p.xgrid.grid_line_color = None

# show results
show(p)


# ## Data Modelling

# ### Linear Regression Modelling

# In[45]:


# Using statsmodels package to build a simple linear regression model to test a linear relationship between
# the dependent variable (Weekly Sales) and the independent variable (Date)

reg_model1 = smf.ols("Weekly_Sales ~  Date", data = df).fit()
model1.summary()

# From the OLS regression results we see that the R-squared value is extremely low (<<1), hence, indicating that
# this linear regression model is not a good fit for the data


# In[47]:


# Using statsmodels package to build a simple linear regression model to test a linear relationship between
# the dependent variable (Weekly Sales) and the independent variable (Store Size). Store Size and Weekly Sales had
# the higest correlation among any 2 variables in this data (0.24 - Low correlation)

reg_model2 = smf.ols("Weekly_Sales ~  Size", data = df).fit()
reg_model2.summary()

# From the OLS regression results, we see that the R-squared value is still very low (<<1) indicating that this model is
# not a good fit for the data


# In[49]:


# Using statsmodels package to build a multiple linear regression model to test a linear relationship between
# the dependent variable (Weekly Sales) and the independent variables (Store Size & IsHoliday)

reg_model3 = smf.ols("Weekly_Sales ~  Size + IsHoliday", data = df).fit()
reg_model3.summary()

# From the OLS regression results, we see that the R-squared value is again very low (<<1) indicating that this model is also
# not a good fit for the data


# In[50]:


# Using statsmodels package to build a polynomial regression model to test a polynomial relationship between
# the dependent variable (Weekly Sales) and the independent variable (Store Size)

reg_model4 = smf.ols("Weekly_Sales ~  + np.power(Size, 2) + np.power(Size,3) + np.power(Size,4) + np.power(Size,5)",
                     data=df).fit()
reg_model4.summary()

# The R-squared value continues to be very low (<<1), indicating that a regression model might not be a good fit for this data


# In[52]:


# Plotted model 2 & model 4 in a graph using matplotlib.pyplot
# model 2 - Simple linear regression between Weekly Sales & Store Size to predict Weekly Sales
# model 4 - Polynomial (quintic) regression between Weekly Sales & Store Size to predict Weekly Sales

weekly_sales_df = pd.DataFrame({"Size": np.arange(0, 300000, 1000)})
weekly_sales_df['Linear Prediction Size Variable'] = reg_model2.predict(weekly_sales_df)
weekly_sales_df['Polynomial Prediction Size Variable'] = reg_model4.predict(weekly_sales_df)

plt.figure(figsize=(15,5))
plt.scatter(df['Size'], df['Weekly_Sales'])
plt.plot(weekly_sales_df['Size'], weekly_sales_df['Linear Prediction Size Variable'], color = 'red')
plt.plot(weekly_sales_df['Size'], weekly_sales_df['Polynomial Prediction Size Variable'], color = 'purple')

# The graphs substantiate that the regression models are not a good fit for this dataset


# ### SARIMA Modelling - Time Series Forecasting

# Seasonal Autoregressive Integrated Moving Average Model

# Since Regression models were not a good fit for this dataset, and the dataset seemed to have seasonality based on the graph, I looked at other options for time series forecasting, and came across SARIMA modelling

# In[53]:


# Created a new dataframe df1 which aggregates Weekly Sales by Date

df1 = df.groupby(by='Date')[["Weekly_Sales"]].sum()
df1.index = pd.to_datetime(df1.index)
df1.head()


# In[54]:


# Checked the data types in the new dataframe

df1.dtypes


# In[55]:


# As we found previously that the dates do not have a fixed frequency, we will resample the data to Monthly frequency,
# since time series forecasting using ARIMA / SARIMA models requires the data to have a fixed frequency

monthly_df1 = df1.Weekly_Sales.resample('M').sum()
monthly_df1 = pd.DataFrame(monthly_df1)
monthly_df1.rename(columns = {'Weekly_Sales':'Monthly_Sales'}, inplace = True) 
monthly_df1.head()


# In[56]:


# Plotting the total monthly sales for all stores (calculated above)

source = ColumnDataSource(monthly_df1)
p = figure(plot_width=900, plot_height=300, title='Total Monthly Sales for all Stores', x_axis_label='Date',
           y_axis_label='Monthly Sales', x_axis_type="datetime")

# add a line renderer
p.line(x='Date', y='Monthly_Sales', source=source, line_width=2)

# Add hover details
hover = HoverTool()
hover.tooltips = [("Total Sales", "@Monthly_Sales{0 a}")]
p.add_tools(hover)

p.left[0].formatter.use_scientific = False
p.yaxis.formatter=NumeralTickFormatter(format="0 a")

# show the results
show(p)


# In[57]:


monthly_df1.describe()


# In[58]:


# We also found during the exploratory analysis phase that the data for Jan 2010 & Dec 2012 starts/ends mid-month, which is 
# why we will discard Jan 2010 & Dec 2012 data
# Nov 2012 data looks like an outlier too, so we will discard that as well

monthly_df1 = monthly_df1[1:-2]
monthly_df1


# In[59]:


# Plotting the total monthly sales for all stores (calculated above), after discarding Jan 2010, Nov 2012 & Dec 2012 data

source = ColumnDataSource(monthly_df1)
p = figure(plot_width=900, plot_height=300, title='Total Monthly Sales for all Stores', x_axis_label='Date',
           y_axis_label='Monthly Sales', x_axis_type="datetime")

# add a line renderer
p.line(x='Date', y='Monthly_Sales', source=source, line_width=2)

# Add hover details
hover = HoverTool()
hover.tooltips = [("Total Sales", "@Monthly_Sales{0 a}")]
p.add_tools(hover)

p.left[0].formatter.use_scientific = False
p.yaxis.formatter=NumeralTickFormatter(format="0 a")

# show the results
show(p)


# In[60]:


# Using statsmodel seasonal_decompose to visualize seasonality and trend. Since the data is monthly, freq = 12

decomposition = seasonal_decompose(monthly_df1, freq=12)  
fig = plt.figure()  
fig = decomposition.plot()  
fig.set_size_inches(15, 8)


# In[61]:


# Created a copy of monthly_df1 to test if the data is stationary (data is not a function of time) using the adfuller test

monthly_df2 = monthly_df1.copy()
result1 = adfuller(monthly_df2['Monthly_Sales'], autolag='AIC')
print('ADF Statistic:{}'.format(result1[0]))
print('p-value:{}'.format(result1[1]))
print('#Lags Used:{}'.format(result1[2]))
print('Number of observations used:{}'.format(result1[3]))
for key, value in result1[4].items():
    print('Critial Values:')
    print(f'   {key}, {value}')
    
# The results of Dickey-Fuller test show that the ADF statistic is higher than all 3 critical values, so we can say that
# the data is NOT stationary


# In[62]:


# Testing the stationarity of the data after one order of differencing

monthly_df2['first_difference'] = monthly_df2 - monthly_df2.shift(1)

result2 = adfuller(monthly_df2['first_difference'].dropna(inplace=False), autolag='AIC')
print('ADF Statistic:{}'.format(result2[0]))
print('p-value:{}'.format(result2[1]))
print('#Lags Used:{}'.format(result2[2]))
print('Number of observations used:{}'.format(result2[3]))
for key, value in result2[4].items():
    print('Critial Values:')
    print(f'   {key}, {value}')
    
# The results of Dickey-Fuller test show that the ADF statistic is lower than 5% critical value, so we can say with
# 95% confidence that the first order differenced data is stationary


# In[63]:


# Added a new column with second order for differencing

monthly_df2['second_difference'] = monthly_df1.diff(2)
monthly_df2.head()


# In[64]:


# Testing the stationarity of the data after two orders of differencing

monthly_df2['second_difference'] = monthly_df2 - monthly_df2.shift(2)

result3 = adfuller(monthly_df2['second_difference'].dropna(inplace=False), autolag='AIC')
print('ADF Statistic:{}'.format(result3[0]))
print('p-value:{}'.format(result3[1]))
print('#Lags Used:{}'.format(result3[2]))
print('Number of observations used:{}'.format(result3[3]))
for key, value in result3[4].items():
    print('Critial Values:')
    print(f'   {key}, {value}')
    
# The results of Dickey-Fuller test show that the ADF statistic is lower than 1% critical value, so we can say with
# 99% confidence that the data is stationary. Since the first order of differencing gave a near stationary data with
# 95% confidence, the second order of difference will lead to overfitting of the model


# In[67]:


# Plotted the monthly sales data with first order differenced and second order differenced data

monthly_df2.plot(figsize=(15,5))
plt.show()


# As shown in the Ad-fuller test, the data is stationary after first order differencing. However, the same will be substantiated below in the estimation of the parameters for SARIMAX model using itertools

# In[69]:


# Using itertools to iterate through various combinations of p, d & q parameters to determine the combination that yields
# the lowest AIC value
# The AIC measures how well a model fits the data while taking into account the overall complexity of the model

warnings.filterwarnings("ignore")

# Define the p, d and q parameters to take any value between 0 and 2
p = d = q = range(0, 2)

# Generate all different combinations of p, q and q triplets
pdq = list(itertools.product(p, d, q))

# Generate all different combinations of seasonal p, q and q triplets
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(monthly_df1,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

            results = mod.fit()

            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue
            
# We see that the values (1,1,1)x(1,1,0)12 for (p,d,q)x(P,D,Q)S parameters yields the lowest AIC value


# In[74]:


# Using the paramter values calculated above, generated the data model

final_mod = sm.tsa.statespace.SARIMAX(monthly_df1, order=(1, 1, 1), seasonal_order=(1, 1, 0, 12), enforce_invertibility=False)
final_res = mod.fit()
print(final_res.summary())


# In[75]:


# Plotted the data and forecast using the SARIMA data model

monthly_df3 = monthly_df1.copy()
monthly_df3['forecast'] = final_res.predict(start = '2010-10-31', end= '2012-12-31')  
monthly_df3[['Monthly_Sales', 'forecast']].plot(figsize=(15, 5))


# ### Model Validation

# In[76]:


# Define Train & Test Data sets

tr_start, tr_end = '2010-01-01','2012-03-31'
te_start, te_end = '2012-04-01', '2012-12-31'
train = monthly_df1['Monthly_Sales'][tr_start:tr_end].dropna()
test = monthly_df1['Monthly_Sales'][te_start:te_end].dropna()


# In[103]:


model_train = sm.tsa.statespace.SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 0, 12), enforce_invertibility=False)
res_train = model_train.fit()
print(res_train.summary())

prediction = res_train.predict(tr_end, te_end)[3:]
print(res_train.predict(tr_end, te_end)[1:])

print('SARIMA Model MSE:{}'.format(mean_squared_error(test, pred)))

plt.figure(figsize=(15,5), dpi=100)
plt.plot(train, label='training')
plt.plot(test, label='actual')
plt.plot(pred, label='forecast')
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()


# The Mean Square Error as calculated above is very high, which indicates that the model needs further fitting

# In[109]:


# Added new monthly time periods to the data frame

pred_monthly_df1 = monthly_df1.copy()

from dateutil.relativedelta import relativedelta, MO
start = datetime.strptime("2012-12-31", "%Y-%m-%d")
date_list = [start + relativedelta(months=x) for x in range(0,12)]
future = pd.DataFrame(index=date_list, columns= monthly_df1.columns)
pred_monthly_df1 = pd.concat([pred_monthly_df1, future])

pred_monthly_df1[-5:]


# In[118]:


# Predicted sales for the next 12 months

forecast = res_train.predict(start = '2012-12-31', end= '2013-12-31')  
print(forecast)
pred_monthly_df1['forecast'] = res_train.predict(start = '2012-10-31', end= '2013-12-31')  
pred_monthly_df1[['Monthly_Sales', 'forecast']].ix[-24:].plot(figsize=(15, 5))


# # Thank you!
