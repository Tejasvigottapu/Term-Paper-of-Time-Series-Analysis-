#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/tirthamutha/repository/blob/main/TimeSeries.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[ ]:


#Final


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd


# In[ ]:


data=pd.read_csv(r"/content/DailyDelhiClimateTrain.csv");data


# In[ ]:


testdata=pd.read_csv(r"/content/DailyDelhiClimateTest.csv")


# In[ ]:


testdata


# In[ ]:


#CONVERTING DATE TO DATE TIME OBJECT

data['data']=pd.to_datetime(data['date'],format='%Y-%m')


# In[ ]:


#CONVERTING DATE TO DATE TIME OBJECT

testdata['testdata']=pd.to_datetime(testdata['date'],format='%Y-%m')


# In[ ]:


sns.lineplot(x='date',y='meantemp',data=data)


# In[ ]:


from pandas.plotting import autocorrelation_plot


# In[ ]:


autocorrelation_plot(data['meantemp'])


# In[ ]:


from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf


# In[ ]:


plot_acf(data['meantemp'])
plt.show()


# In[ ]:


plot_pacf(data['meantemp'])
plt.show()


# In[ ]:


decomposition=seasonal_decompose(data['meantemp'],model='additive',period=12)
decomposition.plot()
plt.show()


# In[ ]:


dftest = adfuller(data.meantemp, autolag = 'AIC')
print("1. ADF : ",dftest[0])
print("2. P-Value : ", dftest[1])
print("3. Num Of Lags : ", dftest[2])
print("4. Num Of Observations Used For ADF Regression and Critical Values Calculation :", dftest[3])
print("5. Critical Values :")
for key, val in dftest[4].items():
    print("\t",key, ": ", val)


# In[ ]:


data_diff=data['meantemp'].diff(periods=350)


# In[ ]:


dftest = adfuller(data_diff.dropna(), autolag = 'AIC')
print("1. ADF : ",dftest[0])
print("2. P-Value : ", dftest[1])
print("3. Num Of Lags : ", dftest[2])
print("4. Num Of Observations Used For ADF Regression and Critical Values Calculation :", dftest[3])
print("5. Critical Values :")
for key, val in dftest[4].items():
    print("\t",key, ": ", val)


# In[ ]:


import pmdarima as pmd


# In[ ]:


model=pmd.auto_arima(data['meantemp'],start_p=1,start_q=1,test='adf',m=12,seasonal=True,trace=True)


# In[ ]:


sarima=SARIMAX(data['meantemp'],order=(1,1,1),seasonal_order=(1,0,1,12))
predicted=sarima.fit().predict();predicted


# In[ ]:


plt.figure(figsize=(20,6))
plt.plot(data['meantemp'],label='Actual')
plt.plot(predicted,label='Predicted')
plt.legend()


# In[ ]:


resid=data['meantemp']-predicted;resid
mae=abs(resid.mean());mae


# In[ ]:


plt.hist(resid)
plt.show()
#residuals follow Normal distribution


# In[ ]:


(resid**2).mean() #mse


# In[ ]:


sarima1=SARIMAX(testdata['meantemp'],order=(1,1,1),seasonal_order=(1,0,1,12))
predicted1=sarima1.fit().predict();predicted1


# In[ ]:


plt.figure(figsize=(20,6))
plt.plot(testdata['meantemp'],label='Actual')
plt.plot(predicted1,label='Predicted')
plt.legend()


# In[ ]:


resid1=testdata['meantemp']-predicted1;resid1
mae1=abs(resid1.mean());mae1


# In[ ]:


plt.hist(resid1)
plt.show()


# In[ ]:


import scipy.stats


# In[ ]:


resid11=scipy.stats.boxcox(abs(resid1))
plt.hist(resid11)
plt.show()


# In[ ]:


(resid1**2).mean() #mse


# In[ ]:


model=pmd.auto_arima(data['meantemp'],start_p=1,start_q=1,test='adf',m=12,seasonal=True,trace=True)


# In[ ]:


pred=model.predict(n_periods=12);pred


# In[ ]:


plt.plot(data['meantemp'])
plt.plot(pred)


# In[ ]:


model.fit(testdata['meantemp'])
pred1=model.predict(n_periods=12);pred1


# In[ ]:


plt.plot(testdata['meantemp'])
plt.plot(pred1)


# In[ ]:


type(predicted)


# In[ ]:


temp=pd.Series(data['meantemp']);temp


# In[ ]:


type(temp)


# In[ ]:


temp1=pd.concat([predicted,pred]);temp1


# In[ ]:


plt.figure(figsize=(20,6))
plt.plot(temp,label='Actual')
plt.plot(temp1,label='Predicted')
plt.legend()


# In[ ]:


newtemp=testdata['meantemp']
newertemp=pd.concat([predicted1,pred1]);newertemp


# In[ ]:


plt.figure(figsize=(20,6))
plt.plot(testdata['meantemp'],label='Actual')
plt.plot(newertemp,label='Predicted')
plt.legend()


# In[ ]:




