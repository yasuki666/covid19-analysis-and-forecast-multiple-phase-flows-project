import pandas as pd
from fbprophet import Prophet
pred = pd.read_csv("../kaggle3/covid-19-all.csv")
pred.head()

pred = pred.fillna(0)
predgrp = pred.groupby("Date")[["Confirmed","Recovered","Deaths"]].sum().reset_index()
predgrp.head()

predgrp.tail()

pred_cnfrm = predgrp.loc[:,["Date","Confirmed"]]
pred_cnfrm.shape

pr_data = pred_cnfrm
pr_data.columns = ['ds','y']
pr_data.head()

m=Prophet()
m.fit(pr_data)
future=m.make_future_dataframe(periods=15)   #预测15天
forecast=m.predict(future)
forecast.head().T

y_ped = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
y_ped.head()

m.plot(forecast,xlabel='Date',ylabel='Confirmed Count', uncertainty=True) #预测

m.plot_components(forecast)   # 成分分析，转换序列输出

#sklearn 误差
from sklearn.metrics import mean_squared_error # 均方误差
from sklearn.metrics import mean_absolute_error # 平方绝对误差

y = y_ped["yhat"].loc[0:91]
(y-pred_cnfrm["Confirmed"]).head()
mean_squared_error(pred_cnfrm["Confirmed"],y
mean_absolute_error(pred_cnfrm["Confirmed"],y)

#numpy 误差
import numpy as np
mse_test=np.sum((y-pred_cnfrm["Confirmed"])**2)/len(y) 
mse_test

rmse_test=mse_test ** 0.5
rmse_test

mae_test=np.sum(abs(y-pred_cnfrm["Confirmed"]))/len(y) 
mae_test