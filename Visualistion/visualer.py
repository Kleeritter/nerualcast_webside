import seaborn as sns
import matplotlib.pyplot as plt
from Visualistion.visualer_funcs import lstm_uni, multilstm_full, start_index_test,start_index_real,end_index_test,end_index_real, conv
from datetime import datetime
import pandas as pd
import itertools
from trad.sarima import sarima
from trad.p_ro import pp
import xarray as xr
# Passe die folgenden Variablen entsprechend an
forecast_var="temp" #Which variable to forecast
window_size=24*7*4 #How big is the window for training
forecast_horizon=24 #How long to cast in the future
forecast_year=2022 #Which year to forecast


nc_path = '../Data/stunden/'+str(forecast_year)+'_resample_stunden.nc' # Replace with the actual path to your NetCDF file
univariant_model_path = '../Model/output/lstm_uni/'+forecast_var+'.pth' # Replace with the actual path to your model
multivariant_model_path = '../Model/output/lstm_multi/'+forecast_var+'.pth' # Replace with the actual path to your model
data = xr.open_dataset(nc_path)
wintertag = pd.to_datetime(str(forecast_year)+"-12-16 00:00")  # Datum im datetime64-Format
sommmertag=pd.to_datetime(str(forecast_year)+"-07-16 00:00")
frühlingstag = pd.to_datetime(str(forecast_year)+"-04-16 00:00")  # Datum im datetime64-Format
herbsttag=pd.to_datetime(str(forecast_year)+"-10-16 00:00")


time_prog=data["index"]  #Zeitachsenwerte
singledata_sommer= data[str(forecast_var)]
data_full=xr.open_dataset(nc_path)[['temp',"press_sl","humid","Geneigt CM-11","gust_10","gust_50", "rain", "wind_10", "wind_50"]]
data_fuller=xr.open_dataset(nc_path)[['temp',"press_sl","humid","Geneigt CM-11","gust_10","gust_50", "rain", "wind_10", "wind_50","wind_dir_50"]]
data_ligth=xr.open_dataset(nc_path)[['temp',"press_sl","humid"]]


Messsomer =data[str(forecast_var)][start_index_real(nc_path,sommmertag):end_index_real(nc_path,sommmertag,forecast_horizon=forecast_horizon)].values.reshape((24+forecast_horizon,))
Messwinter =data[str(forecast_var)][start_index_real(nc_path,wintertag):end_index_real(nc_path,wintertag,forecast_horizon=forecast_horizon)].values.reshape((24+forecast_horizon,))
Messfrühling=data[str(forecast_var)][start_index_real(nc_path,frühlingstag):end_index_real(nc_path,frühlingstag,forecast_horizon=forecast_horizon)].values.reshape((24+forecast_horizon,))
Messherbst=  data[str(forecast_var)][start_index_real(nc_path,herbsttag):end_index_real(nc_path,herbsttag,forecast_horizon=forecast_horizon)].values.reshape((24+forecast_horizon,))

sommer=pd.DataFrame({
    'Datum': time_prog[start_index_real(nc_path,sommmertag):end_index_real(nc_path,sommmertag,forecast_horizon=forecast_horizon)],
    'Messdaten' : Messsomer,
    'Univariantes LSTM': list(itertools.chain(Messsomer[0:24],lstm_uni(univariant_model_path,singledata_sommer,start_index=start_index_test(nc_path,sommmertag,window_size=window_size,forecast_horizon=forecast_horizon),end_index=end_index_test(nc_path,sommmertag)))),#.insert(0, Messsomer[0]),
    #'Multivariantes LSTM_3': list(itertools.chain(Messsomer[0:24],multilstm_light("output/lstm_model_multi_3var.pth",data_ligth,start_idx=start_index_test(nc_path,sommmertag),end_idx=end_index_test(nc_path,sommmertag)))),#.insert(0, Messsomer[0:24]),
    'Multivariantes LSTM': list(itertools.chain(Messsomer[0:24],multilstm_full(multivariant_model_path,data_fuller,start_idx=start_index_test(nc_path,sommmertag,forecast_horizon=forecast_horizon,window_size=window_size),end_idx=end_index_test(nc_path,sommmertag,forecast_horizon=forecast_horizon,window_size=window_size),window_size=window_size))),#.insert(0, Messsomer[0:24])
    'SARIMA' : list(itertools.chain(Messsomer[0:24],sarima(nc_path,sommmertag,forecast_var=forecast_var))),
    'PROPHET' : list(itertools.chain(Messsomer[0:24],pp(nc_path,sommmertag,forecast_var=forecast_var)))
    #'CONV':list(itertools.chain(Messsomer[0:24],conv("output/lstm_model_conv.pth",data_fuller,start_idx=start_index_test(nc_path,sommmertag),end_idx=end_index_test(nc_path,sommmertag))))
})

winter=pd.DataFrame({
    'Datum': time_prog[start_index_real(nc_path,wintertag):end_index_real(nc_path,wintertag)],#pd.date_range(start='2023-01-01', periods=100, freq='D'),
    'Messdaten' : Messwinter,
    'Univariantes LSTM': list(itertools.chain(Messwinter[0:24],lstm_uni(univariant_model_path,singledata_sommer,start_index=start_index_test(nc_path,wintertag,forecast_horizon=forecast_horizon,window_size=window_size),end_index=end_index_test(nc_path,wintertag,forecast_horizon=forecast_horizon,window_size=window_size)))),#.insert(0, Messwinter[0:24]),
    #'Multivariantes LSTM_3':list(itertools.chain(Messwinter[0:24],multilstm_light("output/lstm_model_multi_3var.pth",data_ligth,start_idx=start_index_test(nc_path,wintertag),end_idx=end_index_test(nc_path,wintertag)))),#.insert(0, Messwinter[0:24]),
    'Multivariantes LSTM':list(itertools.chain(Messwinter[0:24],multilstm_full(multivariant_model_path,data_fuller,start_idx=start_index_test(nc_path,wintertag,window_size=window_size),end_idx=end_index_test(nc_path,wintertag),window_size=window_size))),#.insert(0, Messwinter[0:24])
    'SARIMA' : list(itertools.chain(Messwinter[0:24],sarima(nc_path,wintertag,forecast_var=forecast_var))),
    'PROPHET' : list(itertools.chain(Messwinter[0:24],pp(nc_path,wintertag,forecast_var=forecast_var))),
    #'CONV':list(itertools.chain(Messwinter[0:24],conv("output/lstm_model_conv.pth",data_fuller,start_idx=start_index_test(nc_path,wintertag),end_idx=end_index_test(nc_path,wintertag))))
})

herbst=pd.DataFrame({
    'Datum': time_prog[start_index_real(nc_path,herbsttag):end_index_real(nc_path,herbsttag)],#pd.date_range(start='2023-01-01', periods=100, freq='D'),
    'Messdaten' :Messherbst,
    'Univariantes LSTM': list(itertools.chain(Messherbst[0:24],lstm_uni(univariant_model_path,singledata_sommer,start_index=start_index_test(nc_path,herbsttag,forecast_horizon=forecast_horizon),end_index=end_index_test(nc_path,herbsttag,window_size=window_size,forecast_horizon=forecast_horizon)))),#.insert(0, Messherbst[0:24]),
    #'Multivariantes LSTM_3':list(itertools.chain(Messherbst[0:24],multilstm_light("output/lstm_model_multi_3var.pth",data_ligth,start_idx=start_index_test(nc_path,herbsttag),end_idx=end_index_test(nc_path,herbsttag)))),#.insert(0, Messherbst[0:24]),
    'Multivariantes LSTM': list(itertools.chain(Messherbst[0:24],multilstm_full(multivariant_model_path,data_fuller,start_idx=start_index_test(nc_path,herbsttag,window_size=window_size),end_idx=end_index_test(nc_path,herbsttag),window_size=window_size))),
    'SARIMA' : list(itertools.chain(Messherbst[0:24],sarima(nc_path,herbsttag,forecast_var=forecast_var))),
    'PROPHET' : list(itertools.chain(Messherbst[0:24],pp(nc_path,herbsttag,forecast_var=forecast_var))),
    #'CONV':list(itertools.chain(Messherbst[0:24],conv("output/lstm_model_conv.pth",data_fuller,start_idx=start_index_test(nc_path,herbsttag),end_idx=end_index_test(nc_path,herbsttag))))
})

frühling=pd.DataFrame({
    'Datum': time_prog[start_index_real(nc_path,frühlingstag):end_index_real(nc_path,frühlingstag)],#pd.date_range(start='2023-01-01', periods=100, freq='D'),
    'Messdaten' :Messfrühling ,
    'Univariantes LSTM': list(itertools.chain(Messfrühling[0:24],lstm_uni(univariant_model_path,singledata_sommer,start_index=start_index_test(nc_path,frühlingstag,forecast_horizon=forecast_horizon,window_size=window_size),end_index=end_index_test(nc_path,frühlingstag,forecast_horizon=forecast_horizon,window_size=window_size)))),#.insert(0, Messfrühling[0:24]),
    #'Multivariantes LSTM_3':list(itertools.chain(Messfrühling[0:24],multilstm_light("output/lstm_model_multi_3var.pth",data_ligth,start_idx=start_index_test(nc_path,frühlingstag),end_idx=end_index_test(nc_path,frühlingstag)))),#.insert(0, Messfrühling[0:24]),
    'Multivariantes LSTM': list(itertools.chain(Messfrühling[0:24],multilstm_full(multivariant_model_path,data_fuller,start_idx=start_index_test(nc_path,frühlingstag,window_size=window_size),end_idx=end_index_test(nc_path,frühlingstag),window_size=window_size))),
    'SARIMA' : list(itertools.chain(Messfrühling[0:24],sarima(nc_path,frühlingstag,forecast_var=forecast_var))),
    'PROPHET' : list(itertools.chain(Messfrühling[0:24],pp(nc_path,frühlingstag,forecast_var=forecast_var))),
    #'CONV':list(itertools.chain(Messfrühling[0:24],conv("output/lstm_model_conv.pth",data_fuller,start_idx=start_index_test(nc_path,frühlingstag),end_idx=end_index_test(nc_path,frühlingstag))))
})

fig, axs= plt.subplots(2,2,figsize=(12,8))
sns.set(style="darkgrid")  # Setze den Stil des Plots auf "darkgrid"
#sns.scatterplot(x='Datum', y='Univariantes LSTM', data=frühling, ax=axs[0, 0])
sns.lineplot(x='Datum', y='value', hue='variable', data=pd.melt(frühling, ['Datum']),markers=True, style="variable", ax=axs[0,0])
sns.lineplot(x='Datum', y='value', hue='variable', data=pd.melt(sommer, ['Datum']),markers=True, style="variable",ax=axs[0,1])
sns.lineplot(x='Datum', y='value', hue='variable', data=pd.melt(herbst, ['Datum']),markers=True, style="variable", ax=axs[1,0])
sns.lineplot(x='Datum', y='value', hue='variable', data=pd.melt(winter, ['Datum']),markers=True, style="variable", ax=axs[1,1])

# Diagramm anzeigen
plt.tight_layout()
plt.show()

#sns.lineplot(x='Datum', y='value', hue='variable', data=pd.melt(herbst, ['Datum']),markers=True, style="variable")
#plt.show()