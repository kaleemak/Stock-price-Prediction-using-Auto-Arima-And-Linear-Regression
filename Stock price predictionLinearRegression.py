#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#stock price prediction
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#for scalling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))


# In[ ]:


#read the dataset
df = pd.read_csv('NSE-BSE.csv')
df.head()


# In[ ]:


#setting the index with date value
df['Date'] = pd.to_datetime(df.Date,format ='%Y-%m-%d')
df.index = df['Date']
#sort the data in ascending order at rowwise
sort = df.sort_index(ascending =True,axis =0)
#create a seperate dataset that only contain the data an dclose column
newdf = pd.DataFrame(index =range(0,len(df)),columns = ['Date','Close'])
#fill the newdf with the sort values
for i in range(len(sort)):
    newdf['Date'][i]=sort['Date'][i]
    newdf['Close'][i]=sort['Close'][i]


# In[ ]:


#plot the actual data
plt.figure(figsize=(16,8))
plt.plot(df['Close'],label='Previous histort Record')


# In[ ]:


newdf.head(5)


# In[ ]:


# Apart from this, we can add our own set of features that we believe would be relevant for the predictions. For instance, my hypothesis is that the first and last days of the week could potentially affect the closing price of the stock far more than the other days. So I have created a feature that identifies
# whether a given day is Monday/Friday or Tuesday/Wednesday/Thursday. This can be done using the following lines of code:


# In[ ]:


#create features
from fastai.structured import  add_datepart
add_datepart(newdf, 'Date')
newdf.drop('Elapsed', axis=1, inplace=True)  #elapsed will be the time stamp


# In[ ]:


#create a new column[mon-fri]:monday to friday
import sys
newdf['mon-fri'] = 0
for i in range(0,len(newdf)):
    if(newdf['Dayofweek'][i]==0 or newdf['Dayofweek'][i]==4):
        newdf['mon-fri'][i] =1
    else:
        newdf['mon-fri'][i] =0


# In[ ]:


#generate the train test splits
train = newdf[:400]
test = newdf[400:]
#for training
x_train = train.drop('Close',axis=1)
y_train = train['Close']
x_test = test.drop('Close',axis = 1)
y_test = test['Close']


# In[ ]:


#prepare the model
from sklearn.linear_model import LinearRegression
lin = LinearRegression()
lin.fit(x_train,y_train)


# In[ ]:


#make prediction
preds =lin.predict(x_test)


# In[ ]:


preds


# In[ ]:


#calculate the root mean square error
from sklearn.metrics import mean_squared_error
from math import sqrt

rms = sqrt(mean_squared_error(y_test, preds))
rms


# In[ ]:


#now plot thhe graph
test['Predictions'] = 0
test['Predictions'] = preds
plt.figure(figsize=(20,10))
test.index = newdf[400:].index
train.index = newdf[:400].index
plt.plot(train['Close'])
plt.plot(test[['Close','Predictions']])


# In[ ]:


# #Auto ARIMA
# ARIMA is a very popular statistical method for time series forecasting. ARIMA models take into account the past values to predict the future values. There are three important parameters in ARIMA:

# p (past values used for forecasting the next value)
# q (past forecast errors used to predict the future values)
# d (order of differencing)
# Parameter tuning for ARIMA consumes a lot of time. So we will use auto ARIMA which automatically selects the best combination of (p,q,d) that provides the least error. To read more about how auto ARIMA works, refer to this article:


# In[ ]:


#implementation
from pmdarima.arima import auto_arima
#train anad test
train =df[:400]
test =df[400:]
training = train['Close']
testing = test['Close']
#auto-arima model
model = auto_arima(training, start_p=1, start_q=1,max_p=3, max_q=3, m=12,start_P=0, seasonal=True,d=1, D=1, trace=True,error_action='ignore',suppress_warnings=True)
model.fit(training)


# In[ ]:


#forcasting by arima
forcasting = model.predict(n_periods=10)
forecast = pd.DataFrame(forcasting,index = test.index,columns=['Prediction'])


# In[ ]:


#calculate the rms
#calculate the root mean square error
from sklearn.metrics import mean_squared_error
from math import sqrt

rms = sqrt(mean_squared_error(testing, forcasting))
rms


# In[ ]:


#plot the graph
plt.figure(figsize=(16,10))
plt.plot(train['Close'])
plt.plot(test['Close'])
plt.plot(forecast['Prediction'])


# In[ ]:




