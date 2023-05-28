import xarray as xr
import pandas as pd
import glob
import numpy as np
# Öffnen der NetCDF-Datei und Auswählen der Variablen, die resampled werden sollen
filelist=sorted(glob.glob("/home/alex/Downloads/Dator/*.nc"))
#years=np.arange(2022,2007,-1)
years= np.arange(2007,2023,1)
#print(len(filelist))
def resample_zehner(filelist,years):
    for i in range(len(filelist)):
        print(years[i])
        ds = xr.open_dataset(filelist[i])
        time_index = pd.to_datetime(ds['time'].values, unit='s')
        #values =xr.Dataset(coords=dict(time=ds["time"].resample(time="10T",origin="epoch")))
        vars=["humid","temp","press","press_sl","dewpoint_calc","ptd","ptm","wind_10","wind_50","wind_dir_50","gust_10","gust_50","rain","globalrcm11"]
        values= ds[vars].isel(time=time_index.minute % 10 ==0) #time_index.minute%10==0
        #print(values.head())
        # Schleife durch alle Variablen im xarray.Dataset-Objekt
        for var_name, var in values.variables.items():
            # Resampling der Variablen auf stündliche Werte
            if var_name != "time" and  values[var_name].isnull().all() != True :
                #print(var_name)
                # Konvertieren des numerischen Zeitstempels in ein Datums- und Zeitformat
                #ds['time'] = xr.decode_cf(ds['time'])
                # Resampling auf stündliche Werte
                if var_name == "rain":
                    hourly_var = ds[var_name].resample(time='10T',origin="epoch").sum()
                    #ds[var_name] = hourly_var
                    values[var_name]=hourly_var
                    #values.assign(var_name= hourly_var)
                elif var_name == "wind_dir_50":
                    hourly_var = ds[var_name].resample(time='10T',origin="epoch").mean()
                    #ds[var_name] = hourly_var
                    values[var_name]=xr.where(hourly_var<0,0,hourly_var)
                    #values.assign(var_name= hourl

                else:
                    hourly_var = ds[var_name].resample(time='10T',origin="epoch").mean()
                    ds[var_name] = hourly_var
                    values[var_name]=hourly_var

        #            values.assign(var_name=hourly_var)

        # Speichern des resultierenden xarray.Dataset-Objekts in einer neuen NetCDF-Datei
        values.to_netcdf("zehner/"+str(years[i])+'_resample_zehner.nc')#.asfreq(freq='10T', method='pad')

        ds.close()
    return


def resample_stunden(filelist, years):
    for i in range(len(filelist)):
        print(years[i])
        print(filelist[i])
        ds = xr.open_dataset(filelist[i])
        print(len(ds["time"]))
        time_index = pd.to_datetime(ds['time'].values, unit='s')
        # values =xr.Dataset(coords=dict(time=ds["time"].resample(time="10T",origin="epoch")))
        vars = ["humid", "temp", "press", "press_sl", "dewpoint_calc", "ptd", "ptm", "wind_10", "wind_50",
                "wind_dir_50", "gust_10", "gust_50", "rain", "globalrcm11"]
        values = ds[vars].isel(time=time_index.minute % 60 == 0)  # time_index.minute%10==0
        #print(values.head())
        # Schleife durch alle Variablen im xarray.Dataset-Objekt
        for var_name, var in values.variables.items():
            # Resampling der Variablen auf stündliche Werte
            if var_name != "time" and values[var_name].isnull().all() != True:
                # print(var_name)
                # Konvertieren des numerischen Zeitstempels in ein Datums- und Zeitformat
                # ds['time'] = xr.decode_cf(ds['time'])
                # Resampling auf stündliche Werte
                if var_name == "rain":
                    hourly_var = ds[var_name].resample(time='1H', origin="epoch").sum()
                    # ds[var_name] = hourly_var
                    values[var_name] = hourly_var
                    # values.assign(var_name= hourly_var)

                else:
                    hourly_var = ds[var_name].resample(time='1H', origin="epoch").mean()
                    ds[var_name] = hourly_var
                    values[var_name] = hourly_var

        #            values.assign(var_name=hourly_var)

        # Speichern des resultierenden xarray.Dataset-Objekts in einer neuen NetCDF-Datei
        values.to_netcdf("stunden/" + str(years[i]) + '_resample_stunden.nc')  # .asfreq(freq='10T', method='pad')

        ds.close()
    return

for i in range(len(filelist)):
    print(years[i])
    print(filelist[i])
    ds = xr.open_dataset(filelist[i])
    print(len(ds["time"]))
#resample_zehner(filelist,years)
#resample_stunden(filelist,years)