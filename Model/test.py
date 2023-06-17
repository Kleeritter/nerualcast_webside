import matplotlib.pyplot as plt
from funcs.visualer_funcs import lstm_uni, skill_score
import pandas as pd
import xarray as xr
import datetime
import seaborn as sns
import numpy as np
from funcs.trad import p_ro,sarima

# Passe die folgenden Variablen entsprechend an
forecast_var="temp" #Which variable to forecast
window_size=24*7*4 #How big is the window for training
forecast_horizon=24 #How long to cast in the future
forecast_year=2022 #Which year to forecast
dt = datetime.datetime(forecast_year,1,1,0,0) #+ datetime.timedelta(hours=window_size)
dtl=datetime.datetime(forecast_year -1 ,12,31,23)
#print(dt +datetime.timedelta(hours=8760))
dtlast= dtl - datetime.timedelta(hours=window_size+24)
nc_path = '../Data/einer/Messwerte_'+str(forecast_year)+'.nc' # Replace with the actual path to your NetCDF file
nc_path_last = '../Data/einer/Messwerte_'+str(forecast_year-1)+'.nc'
data = xr.open_dataset(nc_path)#.to_dataframe()#["index">dt]
datalast= xr.open_dataset(nc_path_last)
print(data.to_dataframe().iloc[-1])
data=xr.concat([datalast,data],dim="index").to_dataframe()
start_index_forecast = data.index.get_loc(dtlast)
start_index_visual = data.index.get_loc(dt)
forecast_data=data[start_index_forecast:-1]
visual_data=data[start_index_visual:-1]
#print(data[:start_index_visual+1])
#print(data)
datus= xr.open_dataset(nc_path)
print(forecast_data[0:forecast_data.index.get_loc(dt)+1])
