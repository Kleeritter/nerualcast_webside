import torch
import seaborn as sns
import xarray as xr
import matplotlib.pyplot as plt
from visualer_funcs import lstm_uni, multilstm_full,multilstm_light,start_index_test,start_index_real,end_index_test,end_index_real
from datetime import datetime
import pandas as pd
# Passe den Dateipfad entsprechend an


datetime_str = '07/16/22 00:00:00'

datetime_object = datetime.strptime(datetime_str, '%m/%d/%y %H:%M:%S')
# Passe den Dateipfad entsprechend an
nc_path = '../Data/stunden/2022_resample_stunden.nc'
import xarray as xr

data = xr.open_dataset(nc_path)
wintertag = pd.to_datetime("2022-01-16 00:00")  # Datum im datetime64-Format
sommmertag=pd.to_datetime("2022-07-16 00:00")
frühlingstag = pd.to_datetime("2022-04-16 00:00")  # Datum im datetime64-Format
herbsttag=pd.to_datetime("2022-10-16 00:00")

#print(dataf.index.get_loc(gesuchtes_datum))
# Indexbereich für den 16. Juli 2011 ermitteln


# Den von Ihnen gesuchten Indexbereich erhalten Sie mit:
#print(start_index["index"])
#print(data["index"])
#print(data["index"][np.datetime64('2022-07-16T00:00:00')])#["2022-01-01 00:00"])
#time_real= data["index"][start_index_real-1:end_index_real]
time_prog=data["index"]#[start_index_real:end_index_real]
#real_values_sommer = data['temp'][start_index_real-1:end_index_real].values.reshape((25,))
singledata_sommer= data['temp']#[start_index_test:end_index_test].values
data_full=xr.open_dataset(nc_path)[['temp',"press_sl","humid","Geneigt CM-11","gust_10","gust_50", "rain", "wind_10", "wind_50"]]
data_ligth=xr.open_dataset(nc_path)[['temp',"press_sl","humid"]]
##predicted_values.append(predicted_value.item())
# Passe die Achsenbezeichnungen und das Layout entsprechend an
sns.set_theme(style="darkgrid")
#plt.figure(figsize=(10, 6))
#plt.ylim(273, 300)
#sns.lineplot(x=time_real, y=real_values, label='Real values')
#sns.lineplot(x=time, y=test("output/lstm_model_old.pth",real_valueser), label='V1')
#sns.lineplot(x=time_prog, y=test("output/lstm_model_frisch.pth",singledata_sommer), label='V2')
#sns.lineplot(x=time_prog, y=multilstm_light("output/lstm_model_multi_3var.pth",data_ligth,start_idx=start_index_test,end_idx=end_index_test), label='Multi V1')
#sns.lineplot(x=time_prog, y=multilstm_full("output/lstm_model_multi_9var.pth",data_full,start_idx=start_index_test,end_idx=end_index_test), label='Multi V2')
#plt.xlabel('Zeit')
#plt.ylabel('Temperatur in K')
#plt.title('Vorhersagequalität des LSTM-Modells')
#plt.legend()
#plt.show()
Messsomer =data['temp'][start_index_real(nc_path,sommmertag):end_index_real(nc_path,sommmertag)].values.reshape((25,))
Messwinter =data['temp'][start_index_real(nc_path,wintertag):end_index_real(nc_path,wintertag)].values.reshape((25,))
Messfrühling=data['temp'][start_index_real(nc_path,frühlingstag):end_index_real(nc_path,frühlingstag)].values.reshape((25,))
Messherbst=  data['temp'][start_index_real(nc_path,herbsttag):end_index_real(nc_path,herbsttag)].values.reshape((25,))

print(Messfrühling)#[Messsomer[0]]+lstm_uni("output/lstm_model_frisch.pth",singledata_sommer,start_index=start_index_test(nc_path,sommmertag),end_index=end_index_test(nc_path,sommmertag))) #.insert(0, Messsomer[0]))

sommer=pd.DataFrame({
    'Datum': time_prog[start_index_real(nc_path,sommmertag):end_index_real(nc_path,sommmertag)],#pd.date_range(start='2023-01-01', periods=100, freq='D'),
    'Messdaten' : Messsomer,
    'Univariantes LSTM': [Messsomer[0]]+lstm_uni("output/lstm_model_frisch.pth",singledata_sommer,start_index=start_index_test(nc_path,sommmertag),end_index=end_index_test(nc_path,sommmertag)),#.insert(0, Messsomer[0]),
    'Multivariantes LSTM_3':[Messsomer[0]]+multilstm_light("output/lstm_model_multi_3var.pth",data_ligth,start_idx=start_index_test(nc_path,sommmertag),end_idx=end_index_test(nc_path,sommmertag)),#.insert(0, Messsomer[0]),
    'Multivariantes LSTM_9': [Messsomer[0]]+multilstm_full("output/lstm_model_multi_9var.pth",data_full,start_idx=start_index_test(nc_path,sommmertag),end_idx=end_index_test(nc_path,sommmertag))#.insert(0, Messsomer[0])
})

winter=pd.DataFrame({
    'Datum': time_prog[start_index_real(nc_path,wintertag):end_index_real(nc_path,wintertag)],#pd.date_range(start='2023-01-01', periods=100, freq='D'),
    'Messdaten' : Messwinter,
    'Univariantes LSTM': [Messwinter[0]]+lstm_uni("output/lstm_model_frisch.pth",singledata_sommer,start_index=start_index_test(nc_path,wintertag),end_index=end_index_test(nc_path,wintertag)),#.insert(0, Messwinter[0]),
    'Multivariantes LSTM_3':[Messwinter[0]]+multilstm_light("output/lstm_model_multi_3var.pth",data_ligth,start_idx=start_index_test(nc_path,wintertag),end_idx=end_index_test(nc_path,wintertag)),#.insert(0, Messwinter[0]),
    'Multivariantes LSTM_9': [Messwinter[0]]+multilstm_full("output/lstm_model_multi_9var.pth",data_full,start_idx=start_index_test(nc_path,wintertag),end_idx=end_index_test(nc_path,wintertag))#.insert(0, Messwinter[0])
})

herbst=pd.DataFrame({
    'Datum': time_prog[start_index_real(nc_path,herbsttag):end_index_real(nc_path,herbsttag)],#pd.date_range(start='2023-01-01', periods=100, freq='D'),
    'Messdaten' :Messherbst,
    'Univariantes LSTM': [Messherbst[0]]+lstm_uni("output/lstm_model_frisch.pth",singledata_sommer,start_index=start_index_test(nc_path,herbsttag),end_index=end_index_test(nc_path,herbsttag)),#.insert(0, Messherbst[0]),
    'Multivariantes LSTM_3':[Messherbst[0]]+multilstm_light("output/lstm_model_multi_3var.pth",data_ligth,start_idx=start_index_test(nc_path,herbsttag),end_idx=end_index_test(nc_path,herbsttag)),#.insert(0, Messherbst[0]),
    'Multivariantes LSTM_9': [Messherbst[0]]+multilstm_full("output/lstm_model_multi_9var.pth",data_full,start_idx=start_index_test(nc_path,herbsttag),end_idx=end_index_test(nc_path,herbsttag))#.insert(0, Messherbst[0])
})

frühling=pd.DataFrame({
    'Datum': time_prog[start_index_real(nc_path,frühlingstag):end_index_real(nc_path,frühlingstag)],#pd.date_range(start='2023-01-01', periods=100, freq='D'),
    'Messdaten' :Messfrühling ,
    'Univariantes LSTM': [Messfrühling[0]]+lstm_uni("output/lstm_model_frisch.pth",singledata_sommer,start_index=start_index_test(nc_path,frühlingstag),end_index=end_index_test(nc_path,frühlingstag)),#.insert(0, Messfrühling[0]),
    'Multivariantes LSTM_3':[Messfrühling[0]]+multilstm_light("output/lstm_model_multi_3var.pth",data_ligth,start_idx=start_index_test(nc_path,frühlingstag),end_idx=end_index_test(nc_path,frühlingstag)),#.insert(0, Messfrühling[0]),
    'Multivariantes LSTM_9': [Messfrühling[0]]+multilstm_full("output/lstm_model_multi_9var.pth",data_full,start_idx=start_index_test(nc_path,frühlingstag),end_idx=end_index_test(nc_path,frühlingstag))#.insert(0, Messfrühling[0])
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