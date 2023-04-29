import pandas as pd
import pandas.errors
from netCDF4 import Dataset, date2num
from tqdm import tqdm
import numpy as np
import xarray as xr
import os
import numpy as np
import glob
from converterfuncs import pressurereduction, pressreduction_international, kelvinize

filelist= glob.glob("C:/Users/alexa/Downloads/2Alexander/2Alexander/herrenhausen/2007/*.csv")
print(filelist)
fn = 'testa.nc'
try:
    os.remove(fn)
except:
    pass
frames=[]
stringlist=str(np.arange(0,len(filelist)))
for i,j in zip(filelist,stringlist):

    j =pd.read_csv(i,sep=";")
    if '       Datum/Zeit' in j.columns:
            j.rename(columns={'       Datum/Zeit':'Time'}, inplace=True)
    elif '        Date/Time' in j.columns:
            j.rename(columns={'        Date/Time': 'Time'}, inplace=True)

    elif '         Datum/Zeit' in j.columns:
            j.rename(columns={'         Datum/Zeit': 'Time'}, inplace=True)
    else:
        pass
    print(i)
    try:
        j['Time'] = pd.to_datetime( j['Time'], format='%d.%m.%y %H:%M:%S')

    except :
        j['Time'] = pd.to_datetime( j['Time'], format='%d.%m.%Y %H:%M:%S')
    j['Time'] = (j['Time'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
    frames.append(j)

data = pd.concat(frames, ignore_index=True)
data.rename(columns={'   Druck': 'Druck',' Feuchte':'Feuchte','   Regen':'Regen' }, inplace=True)
#xdata=xr.DataArray(data)
#xdata.to_netcdf(path=fn)
#print(xdata)
print(data.head())
#print(data[data.duplicated(subset=['Time'],keep=False)])
#Calculations:
##Pressure
print("Calculating Starting")

for i in  np.arange(0, len(data)):
        data.at[i, 'Temperatur']=kelvinize(data.at[i,'Temperatur'])
        data.at[i,'Druck']= pressreduction_international( data.at[i,'Druck'],51, data.at[i,'Temperatur'])

print("Calculating finished")
print(data.head())
#dimensions
ds = Dataset(fn, 'w', format='NETCDF4')
time = ds.createDimension('time', None)
long= ds.createDimension('long', 1)
lat= ds.createDimension('lat', 1)
ds.description="Test"
ds.title = "Test data"

#variables
times = ds.createVariable('time', np.float64, ('time',))
times.units = 'seconds since 01-01-1970'
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

#Data
times[:]= data['Time'] #date2num(data['         Datum/Zeit'],time_unit_out)

feuchte[:,0,0]=data["Feuchte"]
temperature[:,0,0] = data["Temperatur"]
druck[:,0,0] = data["Druck"]*100
print(len(feuchte))
print(ds)
ds.close()

