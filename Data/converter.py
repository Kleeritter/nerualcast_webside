import pandas as pd
import pandas.errors
from netCDF4 import Dataset, date2num
from tqdm import tqdm
import numpy as np
import xarray as xr
import os
import numpy as np
import glob
from converterfuncs import pressurereduction, pressreduction_international, kelvinize, csvreader,dewpointer
from multiprocessing import Pool
pd.set_option('display.max_columns', None)
filelist= glob.glob("C:/Users/alexa/Downloads/2Alexander/2Alexander/herrenhausen/2007/*.csv")
filelistdach=glob.glob("C:/Users/alexa/Downloads/2Alexander/2Alexander/dach/2007/*.csv")
filelistsonic=glob.glob("C:/Users/alexa/Downloads/2Alexander/2Alexander/sonic/2007/*.txt")
fn = 'testa.nc'
try:
    os.remove(fn)
except:
    print("file doesnt exist")
    pass

print("Beginne Einlesen")
data = csvreader(filelist)

datadach =csvreader(filelistdach)
datasonic= csvreader(filelistsonic)

dataall=data.join(datadach)#pd.concat([data, datadach], axis=1)
dataall=dataall.drop_duplicates()
#print(dataall[dataall.index.duplicated()])
#print(dataall.loc[3658800])
print(datasonic.columns)
datasonic.rename(columns={'  Temperatur':'sonicTemp'},inplace=True)
dataall=dataall.join(datasonic)

#print(dataall.loc[25445700])
dataall.rename(columns={'   Druck': 'Druck',' Feuchte':'Feuchte','   Regen':'Regen' }, inplace=True)
#print(dataall['Druck'])
dataall=dataall.drop_duplicates(subset=['Druck','Feuchte','Temperatur','Regen','Pyranometer'])
print(dataall[dataall.index.duplicated()])

#Calculations:
##Pressure
print("Calculating Starting")

#for i in tqdm(dataall.index):
        #print(i)
 #       dataall.at[i, 'Temperatur']=kelvinize(dataall.at[i,'Temperatur'])
        #dataall.at[i,'Druck']= pressreduction_international( dataall.at[i,'Druck'],51, dataall.at[i,'Temperatur'])
tempus=dataall['Temperatur']+273.15
temp=pd.DataFrame({'Temperatur':tempus})
dataall.update(temp)
druckus=dataall['Druck']
drucka=[]
for i,j in zip( druckus,tempus):
    drucka.append(pressreduction_international(i,51, j))

drucku=pd.DataFrame({'Druck_reduziert':drucka, 'Time':dataall.index}).set_index('Time')
dataall=dataall.join(drucku)
#print(drucku)
tls=dataall['Psychro T']
tlfs=dataall['Psychro Tf']
taus=[]
for i,j in zip(tls,tlfs):
    taus.append(dewpointer(i,j))
tauu=pd.DataFrame({'Td':taus, 'Time':dataall.index}).set_index('Time')
print(tauu.head())
dataall=dataall.join(tauu)
print("Calculating finished")

print(dataall.head())
#print(len(dataall))
#print(dataall.iloc[3658800])
#print(dataall["Druck"])
#dimensions
#print(dataall)
print("Starte NetCDF")
ds = Dataset(fn, 'w', format='NETCDF4')
time = ds.createDimension('time', None)
long= ds.createDimension('long', 1)
lat= ds.createDimension('lat', 1)
ds.description="Test"
ds.title = "Test data"

#variables
times = ds.createVariable('time', np.int64, ('time',))
times.units = 'minutes since 01-01-2007'
times.long_name= 'time'
feuchte = ds.createVariable('humid', 'f4', ('time','long','lat',))
feuchte.units = 'percent'
feuchte.long_name ='relative humidity_2m'

temperature=ds.createVariable('temp', 'f4', ('time','long','lat',))
temperature.units ='K'
temperature.long_name = 'air_temperature_2m'
#
druck=ds.createVariable('press', 'f4', ('time','long','lat',))
druck.units ='hPa'
druck.long_name = 'air_pressure_at_sea_level'

regen= ds.createVariable('rain', 'f4', ('time','long','lat',))
regen.units ='mm'

windu =ds.createVariable('wind_10', 'f4', ('time','long','lat',))
windu.units ='m/s'
windu.long_name = 'wind_speed_10m'

wingu =ds.createVariable('gust_10', 'f4', ('time','long','lat',))
wingu.units ='m/s'
wingu.long_name = 'wind_speed_of_gust_10m'

td=ds.createVariable('dewpoint', 'f4', ('time','long','lat',))
td.units ='K'
td.long_name = 'dew_point_temperature(calculated)'
#Data
times[:]= dataall.index.tolist() #date2num(data['         Datum/Zeit'],time_unit_out)
feuchte[:,0,0]=dataall["Feuchte"]
temperature[:,0,0] = dataall["Temperatur"]
druck[:,0,0] = dataall["Druck"]*100
windu[:,0,0]=dataall["Wind (m/s)"]
wingu[:,0,0]=dataall["Windmax (m/s)"]
regen[:,0,0]=dataall["Regen"]
td[:,0,0]=dataall["Td"]
ds.close()

