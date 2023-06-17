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
nc_path = '../Data/stunden/'+str(forecast_year)+'_resample_stunden.nc' # Replace with the actual path to your NetCDF file
nc_path_last = '../Data/stunden/'+str(forecast_year-1)+'_resample_stunden.nc'
data = xr.open_dataset(nc_path)#.to_dataframe()#["index">dt]
datalast= xr.open_dataset(nc_path_last)
#print(data.to_dataframe().iloc[-1])
data=xr.concat([datalast,data],dim="index").to_dataframe()
start_index_forecast = data.index.get_loc(dtlast)
start_index_visual = data.index.get_loc(dt)
forecast_data=data[start_index_forecast:-1]
visual_data=data[start_index_visual:-1]
#print(data[:start_index_visual+1])
#print(data)
datus= xr.open_dataset(nc_path)
#print(forecast_data[0:forecast_data.index.get_loc(dt)+1])
univariant_model_path = '../Model/output/lstm_uni/'+forecast_var+'unoptimiert.pth' # Replace with the actual path to your model
multivariant_model_path = '../Model/output/lstm_multi/'+forecast_var+'.pth' # Replace with the actual path to your model





learning_rate =0.00005
weight_decay = 0.0001
hidden_size = 32
optimizer= "Adam"
num_layers=1
dropout=0
print("koksnot")
start_index_test = forecast_data.index.get_loc(dt)

references=np.load("reference.csv.npy").flatten()
referencor=[]
print(len(references))
#print(forecast_data.iloc[-1])
predicted_temp_uni =[]
skills=[]
i=0
for window, last_window,refup in zip(range(window_size,len(forecast_data.index.tolist()),forecast_horizon),range(0,len(forecast_data.index.tolist())-window_size,forecast_horizon),range(24,len(references)-48,24)):
    #print(forecast_data["temp"][last_window:window])
    predictions=lstm_uni(univariant_model_path,forecast_data[forecast_var],start_index=last_window,end_index=window)#.insert(0, Messfrühling[0:24]),
    predicted_temp_uni.append(predictions)
    refenrence=references[last_window:refup]#sarima.sarima(forecast_data[forecast_var][last_window:window])
    #referencor.append((refenrence))
    print(refenrence)
    actual=forecast_data[forecast_var][window:window+forecast_horizon]
    skills.append(skill_score(actual_values=actual,prediction=predictions,reference_values=refenrence))
    #print(i)
    i+=1



#np.save(file="reference.csv",arr=np.array(referencor))
print("skills: ",len(np.array(skills)))
print("visual: ",len(visual_data.index.tolist()))
print("mittlerer Skillwert: ", np.mean(np.array(skills)), " Abweichung: ", np.std(np.array(skills)))
sns.set_theme(style="darkgrid")
#print(predicted_temp_uni)
# Load an example dataset with long-form data
visualer=pd.DataFrame({
    #'Datum': visual_data.index.tolist(),
    'Datum': np.arange(0,363,1),
    #'Messdaten': visual_data[forecast_var].tolist(),
    #'Univariantes-LSTM' :np.array(predicted_temp_uni).flatten(),
    'SKILL': np.array(skills).flatten(),

})

# Plot the responses for different events and regions
sns.lineplot(x="Datum", y="SKILL",
             data=visualer)
plt.show()