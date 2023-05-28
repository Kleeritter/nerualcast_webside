import pandas as pd
import pandas.errors
from netCDF4 import Dataset, date2num
#from tqdm import tqdm
import numpy as np
import xarray as xr
import os
import numpy as np
import glob
from converterfuncs import pressurereduction, pressreduction_international, kelvinize, csvreader,dewpointer,dewpt,dew_point,dew_pointa
from multiprocessing import Pool
import datetime
pd.set_option('display.max_columns', None)
"""
try:
    filelist= glob.glob("C:/Users/alexa/Downloads/2Alexander/2Alexander/herrenhausen/2010/*.csv")
    filelistdach=glob.glob("C:/Users/alexa/Downloads/2Alexander/2Alexander/dach/2010/*.csv")
    filelistsonic=glob.glob("C:/Users/alexa/Downloads/2Alexander/2Alexander/sonic/2010/*.txt")
except:
"""
def koks(year):
    filelist= glob.glob("/home/alex/Dokumente/herrenhausen/"+str(year)+"/*.csv")
    filelistdach=glob.glob("/home/alex/Dokumente/dach/"+str(year)+"/*.csv")
    filelistsonic=glob.glob("/home/alex/Dokumente/sonic/"+str(year)+"/*.txt")


    #print("Beginne Einlesen")
    data = csvreader(filelist)

    datadach =csvreader(filelistdach)
    datasonic= csvreader(filelistsonic)

    dataall=data.join(datadach)#pd.concat([data, datadach], axis=1)
    #print(dataall.columns)
    dataall.rename(columns={' Feuchte':'Feuchte','   Regen':'Regen' ,'  Temperatur':'Temperatur'  }, inplace=True)
    dataall=dataall.drop_duplicates()
    #print(dataall[dataall.index.duplicated()])
    #print(dataall.loc[3658800])

    datasonic.rename(columns={'Temperatur':'sonicTemp'},inplace=True)
    #print(datasonic.columns)
    dataall=dataall.join(datasonic)

    #print(dataall.loc[25445700])


    #dataall.rename(columns={'Druck (hPa)': 'Druck',' Feuchte':'Feuchte','   Regen':'Regen' }, inplace=True)
       # pass
    #print(dataall['Druck'])
    #print(dataall.columns)
    dataall=dataall.drop_duplicates(subset=['Druck','Feuchte','Temperatur','Regen'])
    dataall= dataall.apply(pd.to_numeric, errors='coerce')
    print(dataall.index[dataall.index.duplicated()])
    #dataall['Temperatur'] = pd.to_numeric(dataall['Temperatur'], errors='coerce')
    #print(dataall.columns)

   #print(dataall["Temperatur"])
    string_found = False

    # Durchlaufen Sie die Liste und prüfen Sie jedes Element, ob es ein String ist

    #Calculations:
    #print("Calculating Starting")
    #for i in tqdm(dataall.index):
            #print(i)
     #       dataall.at[i, 'Temperatur']=kelvinize(dataall.at[i,'Temperatur'])
            #dataall.at[i,'Druck']= pressreduction_international( dataall.at[i,'Druck'],51, dataall.at[i,'Temperatur'])

    tempuss=dataall['Temperatur']+273.15
    #temp=pd.DataFrame({'Temperatur':tempus})
    #dataall.update(temp)
    druckus=dataall['Druck']
    #print(druckus)
    drucka=[]
    for i,j in zip( druckus,tempuss):
        drucka.append(pressreduction_international(i,51, j))

    drucku=pd.DataFrame({'Druck_reduziert':drucka, 'Time':dataall.index}).set_index('Time')
    dataall=dataall.join(drucku)
    #print(drucku)
    tls=dataall['PsychroT']
    tlfs=dataall['PsychroTf']
    rs=dataall['Feuchte']*100
    #print(tempus)
    taus=[]
    taupt= []
    tempus=dataall['Temperatur']#+273.15
    for i,j in zip(tls,tlfs):
        taus.append(dewpointer(i,j))
        #taupt.append(dew_point(i,j))
    for i,j in zip(tempus,rs):
        taupt.append(dew_pointa(i,j))
    #print(len(druckus),len(tempuss))
    #print(len(dataall["Temperatur"]),len(dataall.index))
    #print(dataall["Temperatur"].isna().sum())
    #print(dataall[pd.to_numeric(dataall['Temperatur'], errors='coerce').isna()])
    #print(dataall.head())
    tauu=pd.DataFrame({'Td':taus, 'Time':dataall.index}).set_index('Time')
    tauupt=pd.DataFrame({'Tdp':taupt, 'Time':dataall.index}).set_index('Time')
    #print(tauu.head())
    dataall=dataall.join(tauu)
    dataall=dataall.join(tauupt)
    #dataall=dataall.fillna(9999)
    #print("Calculating finished")

    #print(dataall.head())
    #print(len(dataall))
    #print(dataall.iloc[3658800])
    #print(dataall["Druck"])
    #dimensions
    #print(dataall)
    koko= str(datetime.datetime.fromtimestamp(dataall.index[0]).strftime('%Y'))
    fn = "Messwerte_"+koko +".nc"
    try:
        os.remove(fn)
    except:
        #print("file doesnt exist")
        pass
    #print("Starte NetCDF")
    ds = Dataset(fn, 'w', format='NETCDF4')
    #sd=ds.createGroup('Stationsdaten')
    #dd=ds.createGroup('Dachdaten')
    #ssd=ds.createGroup('Sonic')
    time = ds.createDimension('time', None)
    long= ds.createDimension('long', 1)
    lat= ds.createDimension('lat', 1)
    ds.description="Herrenhausen Daten"
    ds.title = "Herrenhausen "+koko

    #variables
    times = ds.createVariable('time', np.int64, ('time',), zlib=True)
    times.units = 'seconds since 1970-01-01 00:00:00'
    times.long_name= 'time'
    feuchte = ds.createVariable('humid', 'f4', ('time',),zlib=True,shuffle=True )
    feuchte.units = 'percent'
    feuchte.long_name ='relative humidity_2m'

    temperature=ds.createVariable('temp', 'f4', ('time',),zlib=True,shuffle=True  )
    temperature.units ='K'
    temperature.long_name = 'air_temperature_2m'
    #
    druck=ds.createVariable('press', 'f4', ('time',),zlib=True,shuffle=True )
    druck.units ='Pa'
    druck.long_name = 'air_pressure_measured'

    regen= ds.createVariable('rain', 'f4', ('time',),zlib=True,shuffle=True )
    regen.units ='mm'
    regen.long_name ='precipitation'

    windu =ds.createVariable('wind_10', 'f4', ('time',),zlib=True,shuffle=True )
    windu.units ='m/s'
    windu.long_name = 'wind_speed_10m'

    wingu =ds.createVariable('gust_10', 'f4', ('time',),zlib=True,shuffle=True )
    wingu.units ='m/s'
    wingu.long_name = 'wind_speed_of_gust_10m'

    windo =ds.createVariable('wind_50', 'f4', ('time',),zlib=True,shuffle=True )
    windo.units ='m/s'
    windo.long_name = 'wind_speed_50m'

    wingo =ds.createVariable('gust_50', 'f4', ('time',),zlib=True,shuffle=True )
    wingo.units ='m/s'
    wingo.long_name = 'wind_speed_of_gust_50m'

    windiro= ds.createVariable('wind_dir_50', np.int64, ('time',),zlib=True,shuffle=True )
    windiro.units ='Degrees'
    windiro.long_name='wind_from_direction_50m'

    #td=ds.createVariable('dewpoint', 'f4', ('time',),zlib=True,shuffle=True )
    #td.units ='K'
    #td.long_name = 'dew_point_temperature_calculated'
    #

    dtd=ds.createVariable('dewpoint_calc', 'f4', ('time',),zlib=True,shuffle=True )
    dtd.units ='K'
    dtd.long_name = 'dew_point_temperature_calculated'

    tpl=ds.createVariable('ptd', 'f4', ('time',),zlib=True,shuffle=True )
    tpl.units ='K'
    tpl.long_name = 'psychro_temperature_dry'

    tpf=ds.createVariable('ptm', 'f4', ('time',),zlib=True,shuffle=True )
    tpf.units ='K'
    tpf.long_name = 'psychro_temperature_moist'

    drucksl=ds.createVariable('press_sl', 'f4', ('time',),zlib=True,shuffle=True )
    drucksl.units ='Pa'
    drucksl.long_name = 'air_pressure_at_sea_level_calculated'

    temperatures=ds.createVariable('temps', 'f4', ('time',),zlib=True,shuffle=True )
    temperatures.units ='K'
    temperatures.long_name = 'air_temperature_sonic'

    korona=ds.createVariable('korona', 'f4', ('time',),zlib=True,shuffle=True )
    korona.units ='mV'
    korona.long_name = 'korona'

    globals=ds.createVariable('globalrcm11', 'f4', ('time',),zlib=True,shuffle=True )
    globals.units ='W/m2'
    globals.long_name = 'global_radiation_cm-11'

    globalsp=ds.createVariable('globalrcmp11', 'f4', ('time',),zlib=True,shuffle=True )
    globalsp.units ='W/m2'
    globalsp.long_name = 'global_radiation_cmp-11'

    diffuscmp=ds.createVariable('diffuscmp11', 'f4', ('time',),zlib=True,shuffle=True )
    globalsp.units ='W/m2'
    globalsp.long_name = 'diffuse_radiation_cmp-11'

    lnet=ds.createVariable('L_net', 'f4', ('time',),zlib=True,shuffle=True )
    lnet.units ='W/m2'
    lnet.long_name = 'L_net'

    ld=ds.createVariable('L_dwn', 'f4', ('time',),zlib=True,shuffle=True )
    ld.units ='W/m2'
    ld.long_name = 'L_dwn'

    pycm=ds.createVariable('pyr_CM3', 'f4', ('time',),zlib=True,shuffle=True )
    pycm.units ='W/m2'
    pycm.long_name = 'Pyranometer_CM3'

    pycme=ds.createVariable('pyr_CM11', 'f4', ('time',),zlib=True,shuffle=True )
    pycme.units ='W/m2'
    pycme.long_name = 'Pyranometer_CM11'

    pyc=ds.createVariable('pyr', 'f4', ('time',),zlib=True,shuffle=True )
    pyc.units ='W/m2'
    pyc.long_name = 'Pyranometer'

    pyg=ds.createVariable('Geneigt CM-11', 'f4', ('time',),zlib=True,shuffle=True )
    pyg.units ='W/m2'
    pyg.long_name = 'Geneigt CM-11'

    cotwo=ds.createVariable('co2_sensor', 'f4', ('time',),zlib=True,shuffle=True )
    cotwo.units = 'Tics'
    cotwo.long_name = 'co2_sensor'

    cotwopp=ds.createVariable('co2_sensor_ppm', 'f4', ('time',),zlib=True,shuffle=True )
    cotwopp.units = 'PPM'
    cotwopp.long_name = 'co2_sensor_ppm'

    tempamudis=ds.createVariable('temp_amudis', 'f4', ('time',),zlib=True,shuffle=True )
    tempamudis.units = 'K'
    tempamudis.long_name = 'temperature_amudis_box'

    tempbent=ds.createVariable('temp_bentham', 'f4', ('time',),zlib=True,shuffle=True )
    tempbent.units = 'K'
    tempbent.long_name = 'temperature_bentham_box'


    #Data
    times[:]= dataall.index.tolist() #date2num(data['         Datum/Zeit'],time_unit_out)
    feuchte[:]=dataall["Feuchte"]
    temperature[:] = dataall["Temperatur"]+273.15
    temperatures[:] = dataall["sonicTemp"]+273.15
    druck[:] = dataall["Druck"]*100
    drucksl[:] = dataall["Druck_reduziert"]*100
    windu[:]=dataall["Wind(m/s)"]
    wingu[:]=dataall["Windmax(m/s)"]
    windo[:]=dataall["Wind"]
    wingo[:]=dataall["Peak"]
    windiro[:]=dataall["Richtung"]
    regen[:]=dataall["Regen"]*0.01
    #td[:]=dataall["Td"]+273.15
    dtd[:]=dataall["Tdp"]+273.15
    tpl[:] = dataall["PsychroT"]+273.15
    tpf[:] = dataall["PsychroTf"]+273.15
    try:
        korona[:]=dataall["Korona(mV)"]
    except KeyError:
        pass
    try:
        globals[:]=dataall["Globalstrahlung"]
    except KeyError:
        try:
            globals[:] = dataall["GlobalCM-11(W/m2)"]
        except:
            pass

    try:
        diffuscmp[:]=dataall["CMP-11Diffus(W/m2)"]
    except KeyError:
        try:
            diffuscmp[:] = dataall["DiffusCMP-11(W/m2)"]
        except:
            pass


    try:
        globalsp[:]=dataall["GlobalCMP-11(W/m2)"]
    except KeyError:
        pass

    try:
        lnet[:]=dataall["Lnet"]
    except KeyError:
        pass
    try:
        ld[:]=dataall["Ldwn"]
    except KeyError:
        pass
    try:
        pycm[:]=dataall["PyranometerCM3"]
    except KeyError:
        pass
    try:
        pycme[:]=dataall["PyranometerCM-11"]
    except KeyError:
        pass
    try:
        pyc[:]=dataall["Pyranometer"]
    except KeyError:
        pass
    try:
        cotwo[:]=dataall["CO2Sensor"]
    except KeyError:
        pass
    try:
        cotwopp[:]=dataall["CO2ppm"]
    except KeyError:
        pass
    try:
        tempamudis[:]=dataall["TempAMUDISBox"]+273.15
    except KeyError:
        pass
    try:
        tempamudis[:]=dataall["TempBenthamBox"]+273.15
    except KeyError:
        pass

    try:
        pyg[:]=dataall["GeneigtCM-11(W/m2)"]+273.15
    except KeyError:
        pass




years=np.arange(2007,2023,1)

#with Pool() as pool:
    #pool.map(koks, years)

for year in years:
   koks(year)

#koks(2007)