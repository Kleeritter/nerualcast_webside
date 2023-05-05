import pandas as pd
import pandas.errors
from netCDF4 import Dataset, date2num
#from tqdm import tqdm
import numpy as np
import xarray as xr
import os
import numpy as np
import glob
from converterfuncs import pressurereduction, pressreduction_international, kelvinize, csvreader,dewpointer
from multiprocessing import Pool
pd.set_option('display.max_columns', None)
"""
try:
    filelist= glob.glob("C:/Users/alexa/Downloads/2Alexander/2Alexander/herrenhausen/2007/*.csv")
    filelistdach=glob.glob("C:/Users/alexa/Downloads/2Alexander/2Alexander/dach/2007/*.csv")
    filelistsonic=glob.glob("C:/Users/alexa/Downloads/2Alexander/2Alexander/sonic/2007/*.txt")
except:
"""

filelist= glob.glob("/home/alex/Dokumente/herrenhausen/2007/*.csv")
filelistdach=glob.glob("/home/alex/Dokumente/dach/2007/*.csv")
filelistsonic=glob.glob("/home/alex/Dokumente/sonic/2007/*.txt")
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

datasonic.rename(columns={'  Temperatur':'sonicTemp'},inplace=True)
dataall=dataall.join(datasonic)

#print(dataall.loc[25445700])
dataall.rename(columns={'   Druck': 'Druck',' Feuchte':'Feuchte','   Regen':'Regen' }, inplace=True)
#print(dataall['Druck'])
dataall=dataall.drop_duplicates(subset=['Druck','Feuchte','Temperatur','Regen','Pyranometer'])
print(dataall.columns)

#Calculations:
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
dataall=dataall.fillna(9999)
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
ds.description="Herrenhausen Daten"
ds.title = "Herrenhausen 2007"

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
druck.units ='Pa'
druck.long_name = 'air_pressure_measured'

regen= ds.createVariable('rain', 'f4', ('time','long','lat',))
regen.units ='mm'
regen.long_name ='precipitation'

windu =ds.createVariable('wind_10', 'f4', ('time','long','lat',))
windu.units ='m/s'
windu.long_name = 'wind_speed_10m'

wingu =ds.createVariable('gust_10', 'f4', ('time','long','lat',))
wingu.units ='m/s'
wingu.long_name = 'wind_speed_of_gust_10m'

windo =ds.createVariable('wind_50', 'f4', ('time','long','lat',))
windo.units ='m/s'
windo.long_name = 'wind_speed_50m'

wingo =ds.createVariable('gust_50', 'f4', ('time','long','lat',))
wingo.units ='m/s'
wingo.long_name = 'wind_speed_of_gust_50m'

windiro= ds.createVariable('wind_dir_50', np.int64, ('time','long','lat',))
windiro.units ='Degrees'
windiro.long_name='wind_from_direction_50m'

td=ds.createVariable('dewpoint', 'f4', ('time','long','lat',))
td.units ='K'
td.long_name = 'dew_point_temperature_calculated'

tpl=ds.createVariable('ptd', 'f4', ('time','long','lat',))
tpl.units ='K'
tpl.long_name = 'psychro_temperature_dry'

tpf=ds.createVariable('ptm', 'f4', ('time','long','lat',))
tpf.units ='K'
tpf.long_name = 'psychro_temperature_moist'

drucksl=ds.createVariable('press_sl', 'f4', ('time','long','lat',))
drucksl.units ='Pa'
drucksl.long_name = 'air_pressure_at_sea_level_calculated'

temperatures=ds.createVariable('temps', 'f4', ('time','long','lat',))
temperatures.units ='K'
temperatures.long_name = 'air_temperature_sonic'

korona=ds.createVariable('korona', 'f4', ('time','long','lat',))
korona.units ='mV'
korona.long_name = 'korona'

globals=ds.createVariable('globalrcm11', 'f4', ('time','long','lat',))
globals.units ='W/m2'
globals.long_name = 'global_radiation_cm-11'

globalsp=ds.createVariable('globalrcmp11', 'f4', ('time','long','lat',))
globalsp.units ='W/m2'
globalsp.long_name = 'global_radiation_cmp-11'

diffuscmp=ds.createVariable('diffuscmp11', 'f4', ('time','long','lat',))
globalsp.units ='W/m2'
globalsp.long_name = 'diffuse_radiation_cmp-11'

lnet=ds.createVariable('L_net', 'f4', ('time','long','lat',))
lnet.units ='W/m2'
lnet.long_name = 'L_net'

ld=ds.createVariable('L_dwn', 'f4', ('time','long','lat',))
ld.units ='W/m2'
ld.long_name = 'L_dwn'

pycm=ds.createVariable('pyr_CM3', 'f4', ('time','long','lat',))
pycm.units ='W/m2'
pycm.long_name = 'Pyranometer_CM3'

pycme=ds.createVariable('pyr_CM11', 'f4', ('time','long','lat',))
pycme.units ='W/m2'
pycme.long_name = 'Pyranometer_CM11'

pyc=ds.createVariable('pyr', 'f4', ('time','long','lat',))
pyc.units ='W/m2'
pyc.long_name = 'Pyranometer'

cotwo=ds.createVariable('co2_sensor', 'f4', ('time','long','lat',))
cotwo.units = 'Tics'
cotwo.long_name = 'co2_sensor'

cotwopp=ds.createVariable('co2_sensor_ppm', 'f4', ('time','long','lat',))
cotwopp.units = 'PPM'
cotwopp.long_name = 'co2_sensor_ppm'



#Data
times[:]= dataall.index.tolist() #date2num(data['         Datum/Zeit'],time_unit_out)
feuchte[:,0,0]=dataall["Feuchte"]
temperature[:,0,0] = dataall["Temperatur"]
temperatures[:,0,0] = dataall["sonicTemp"]+273.15
druck[:,0,0] = dataall["Druck"]*100
drucksl[:,0,0] = dataall["Druck_reduziert"]*100
windu[:,0,0]=dataall["Wind (m/s)"]
wingu[:,0,0]=dataall["Windmax (m/s)"]
windo[:,0,0]=dataall[" Wind"]
wingo[:,0,0]=dataall[" Peak "]
windiro[:,0,0]=dataall[" Richtung"]
regen[:,0,0]=dataall["Regen"]*0.01
td[:,0,0]=dataall["Td"]+273.15
tpl[:,0,0] = dataall["Psychro T"]+273.15
tpf[:,0,0] = dataall["Psychro Tf"]+273.15
try:
    korona[:,0,0]=dataall["Korona (mV)"]
except KeyError:
    pass
try:
    globals[:,0,0]=dataall["Globalstrahlung"]
except KeyError:
    try:
        globals[:, 0, 0] = dataall["Global CM-11 (W/m2)"]
    except:
        pass

try:
    diffuscmp[:,0,0]=dataall["CMP-11 Diffus (W/m2)"]
except KeyError:
    try:
        diffuscmp[:, 0, 0] = dataall["Diffus CMP-11(W/m2)"]
    except:
        pass


try:
    globalsp[:,0,0]=dataall["Global CMP-11 (W/m2)"]
except KeyError:
    pass

try:
    lnet[:,0,0]=dataall["   L net"]
except KeyError:
    pass
try:
    ld[:,0,0]=dataall["   L dwn"]
except KeyError:
    pass
try:
    pycm[:,0,0]=dataall["Pyranometer CM3"]
except KeyError:
    pass
try:
    pycme[:,0,0]=dataall["Pyranometer CM-11"]
except KeyError:
    pass
try:
    pyc[:,0,0]=dataall["Pyranometer"]
except KeyError:
    pass
try:
    cotwo[:,0,0]=dataall["CO2 Sensor"]
except KeyError:
    pass
try:
    cotwopp[:,0,0]=dataall[" CO2 ppm"]
except KeyError:
    pass
ds.close()

