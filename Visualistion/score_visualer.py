import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import datetime
import seaborn as sns
import numpy as np
from funcs.trad import p_ro,sarima
from funcs.visualer_funcs import skill_score
# Passe die folgenden Variablen entsprechend an
forecast_var="temp" #Which variable to forecast
window_size=24*7*4 #How big is the window for training
forecast_horizon=24 #How long to cast in the future
forecast_year=2022 #Which year to forecast
dt = datetime.datetime(forecast_year,1,1,0,0) #+ datetime.timedelta(hours=window_size)



nc_path = '../Data/stunden/'+str(forecast_year)+'_resample_stunden.nc' # Replace with the actual path to your NetCDF file

references=np.load("reference_new.npy").flatten()
lstm_uni_path="forecast_lstm_uni.nc"
lstm_multi_path="forecast_lstm_multi.nc"
tft_path="forecast_tft.nc"
lstm_uni=xr.open_dataset(lstm_uni_path).to_dataframe()
lstm_multi=xr.open_dataset(lstm_multi_path).to_dataframe()
tft=xr.open_dataset(tft_path).to_dataframe()
data = xr.open_dataset(nc_path).to_dataframe()


skills=[]
i=0


for start,end in zip(range(0,len(lstm_uni)-forecast_horizon),range(forecast_horizon,len(lstm_uni))):
    actual_values=data[forecast_var][start:end]
    reference_values=references[start:end]
    prediction=tft[forecast_var][start:end]
    skill=skill_score(actual_values=actual_values,reference_values=reference_values,prediction=prediction)
    skills.append(skill)
    if skill<-15:
        print("reference: ",reference_values)
        print("real: ", actual_values)
        print("forecast: ", prediction)





#np.save(file="reference.csv",arr=np.array(referencor))
print("skills: ",len(np.array(skills)))
print("mittlerer Skillwert: ", np.mean(np.array(skills)), " Abweichung: ", np.std(np.array(skills)))
sns.set_theme(style="darkgrid")

visualerdata=pd.DataFrame({
    'Datum': data.index.tolist(),
    #'Datum': np.arange(0,363,1),
    'Messdaten': data[forecast_var].tolist(),
    'Univariantes-LSTM' :lstm_uni[forecast_var],
    'Multivariantes-LSTM' :lstm_multi[forecast_var],#[:-25],
    'TFT' :tft[forecast_var],
    #'SKILL': np.array(skills).flatten(),
    'SARIMA' : references[:-24]

})

skiller=pd.DataFrame({
    'Datum': data.index.tolist()[:-24],
    #'Datum': np.arange(0,363,1),
    #'Messdaten': data[forecast_var].tolist(),
    #'Univariantes-LSTM' :lstm_uni[forecast_var],
    #'Multivariantes-LSTM' :lstm_multi[forecast_var],#[:-25],
    #'TFT' :np.array(predicted_temp_tft).flatten(),
    'SKILL': np.array(skills).flatten(),
    #'SARIMA' : references[:-24]

})

# Plot the responses for different events and regions
#sns.lineplot(x="Datum", y="SKILL",             data=skiller)

sns.lineplot(x="Datum", y='value', hue='variable',data=pd.melt(visualerdata, ['Datum']))
plt.show()