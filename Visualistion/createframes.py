import matplotlib.pyplot as plt
from funcs.visualer_funcs import lstm_uni, skill_score,multilstm_full,tft
import pandas as pd
import xarray as xr
import datetime
import seaborn as sns
import numpy as np
from funcs.trad import p_ro,sarima
from tqdm import tqdm

# Passe die folgenden Variablen entsprechend an
forecast_var="temp" #Which variable to forecast
window_size=24*7*4 #How big is the window for training
forecast_horizon=24 #How long to cast in the future
forecast_year=2022 #Which year to forecast
dt = datetime.datetime(forecast_year,1,1,0,0) #+ datetime.timedelta(hours=window_size)
dtl=datetime.datetime(forecast_year -1 ,12,31,23)
#print(dt +datetime.timedelta(hours=8760))
dtlast= dtl - datetime.timedelta(hours=window_size-1)
nc_path = '../Data/stunden/'+str(forecast_year)+'_resample_stunden.nc' # Replace with the actual path to your NetCDF file
nc_path_last = '../Data/stunden/'+str(forecast_year-1)+'_resample_stunden.nc'
data = xr.open_dataset(nc_path)#.to_dataframe()#["index">dt]
datalast= xr.open_dataset(nc_path_last)
#print(data.to_dataframe().iloc[-1])
data=xr.concat([datalast,data],dim="index").to_dataframe()
start_index_forecast = data.index.get_loc(dtlast)
start_index_visual = data.index.get_loc(dt)
forecast_data=data[start_index_forecast:]
#print(forecast_data)
visual_data=data[start_index_visual:]
#print(data[:start_index_visual+1])
#print(data)
datus= xr.open_dataset(nc_path)
#print(forecast_data[0:forecast_data.index.get_loc(dt)+1])
univariant_model_path = '../Model/output/lstm_uni/'+forecast_var+'optimierter.pth' # Replace with the actual path to your model
multivariant_model_path = '../Model/output/lstm_multi/'+forecast_var+'_unoptimiert.pth' # Replace with the actual path to your model
tft_model_path='../Model/output/tft/'+forecast_var+'._unoptimiert.pth' # Replace with the actual path to your model




learning_rate =0.00005
weight_decay = 0.0001
hidden_size = 32
optimizer= "Adam"
num_layers=1
dropout=0
#print("koksnot")
start_index_test = forecast_data.index.get_loc(dt)

references=np.load("reference_new.npy").flatten()

#print(forecast_data.iloc[-1])


predicted_temp_tft=[]


def fullsarima():
    import datetime
    import xarray as xr
    import numpy as np
    var_list=["wind_dir_50"]#, "Geneigt CM-11", 'temp', "press_sl", "humid", "diffuscmp11", "globalrcmp11", "gust_10", "gust_50","rain", "wind_10", "wind_50"]

    empty_lists = {var: [] for var in var_list}


    referencor=[]
    i=0

    for var in var_list:
        print(var)
        for window, last_window in tqdm(zip(range(window_size, len(forecast_data.index.tolist()), forecast_horizon),
                                       range(0, len(forecast_data.index.tolist()) - window_size,
                                             forecast_horizon))):
            refenrence = sarima.sarima(forecast_data[forecast_var][last_window:window])
            empty_lists[var].append(refenrence)

    df = pd.DataFrame(empty_lists)

    # Setze den Index auf "Datum"
    df = df.set_index(pd.to_datetime(visual_data.index.tolist()), inplace=False)
    output_file = "forecast_sarima.nc"
    df=xr.Dataset.from_dataframe(df)
    df.to_netcdf(output_file)
    return




def forecast_lstm_uni():
    predicted_temp_uni = []
    for window, last_window in zip(range(window_size, len(forecast_data.index.tolist()), forecast_horizon),
                                   range(0, len(forecast_data.index.tolist()) - window_size,
                                         forecast_horizon)):

        predictions = lstm_uni(univariant_model_path, forecast_data[forecast_var], start_index=last_window,
                               end_index=window)  # .insert(0, Messfrühling[0:24]),
        predicted_temp_uni.append(predictions)

    data = pd.DataFrame({
        'temp': np.array(predicted_temp_uni).flatten(),

    })
    output_file = "forecast_lstm_uni.nc"
    df = data.set_index(pd.to_datetime(visual_data.index.tolist()), inplace=False)
    df.index.name = "Datum"
    df=xr.Dataset.from_dataframe(df)
    df.to_netcdf(output_file)




def forecast_lstm_multi():
    predicted_temp_multi = []
    for window, last_window in zip(range(window_size, len(forecast_data.index.tolist()), forecast_horizon),
                                   range(0, len(forecast_data.index.tolist()) - window_size,
                                         forecast_horizon)):
        predictions_multi = multilstm_full(multivariant_model_path, forecast_data, start_idx=last_window,
                                           end_idx=window, forecast_var=forecast_var)
        predicted_temp_multi.append(predictions_multi)
    data = pd.DataFrame({
        'temp': np.array(predicted_temp_multi).flatten(),

    })
    output_file = "forecast_lstm_multi.nc"
    df = data.set_index(pd.to_datetime(visual_data.index.tolist()), inplace=False)
    df.index.name = "Datum"
    df=xr.Dataset.from_dataframe(df)
    df.to_netcdf(output_file)

def forecast_tft():
    predicted_temp_multi = []
    for window, last_window in zip(range(window_size, len(forecast_data.index.tolist()), forecast_horizon),
                                   range(0, len(forecast_data.index.tolist()) - window_size,
                                         forecast_horizon)):
        predictions_multi = tft(tft_model_path, forecast_data, start_idx=last_window,
                                           end_idx=window, forecast_var=forecast_var)
        predicted_temp_multi.append(predictions_multi)
    data = pd.DataFrame({
        'temp': np.array(predicted_temp_multi).flatten(),

    })
    output_file = "forecast_tft.nc"
    df = data.set_index(pd.to_datetime(visual_data.index.tolist()), inplace=False)
    df.index.name = "Datum"
    df=xr.Dataset.from_dataframe(df)
    df.to_netcdf(output_file)





#fullsarima()
#forecast_lstm_uni()
#forecast_lstm_multi()

forecast_tft()