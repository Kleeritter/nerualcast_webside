import xarray as xr
import pandas as pd
# Öffnen der NetCDF-Datei und Auswählen der Variablen, die resampled werden sollen
ds = xr.open_dataset('Messwerte_2007.nc')
time_index = pd.to_datetime(ds['time'].values, unit='s')
#values =xr.Dataset(coords=dict(time=ds["time"].resample(time="10T",origin="epoch")))
vars=["humid","temp","press","press_sl","dewpoint_calc","ptd","ptm","wind_10","wind_50","wind_dir_50","gust_10","gust_50","rain","globalrcm11"]
values= ds[vars].isel(time=time_index.minute % 60 ==0) #time_index.minute%10==0
print(values.head())
# Schleife durch alle Variablen im xarray.Dataset-Objekt
for var_name, var in values.variables.items():
    # Resampling der Variablen auf stündliche Werte
    if var_name != "time" and  values[var_name].isnull().all() != True :
        print(var_name)
        # Konvertieren des numerischen Zeitstempels in ein Datums- und Zeitformat
        #ds['time'] = xr.decode_cf(ds['time'])
        # Resampling auf stündliche Werte
        if var_name == "rain":
            hourly_var = ds[var_name].resample(time='1H',origin="epoch").sum()
            #ds[var_name] = hourly_var
            values[var_name]=hourly_var
            #values.assign(var_name= hourly_var)

        else:
            hourly_var = ds[var_name].resample(time='1H',origin="epoch").mean()
            ds[var_name] = hourly_var
            values[var_name]=hourly_var

#            values.assign(var_name=hourly_var)

# Speichern des resultierenden xarray.Dataset-Objekts in einer neuen NetCDF-Datei
values.to_netcdf('test_resample_stunde.nc')#.asfreq(freq='10T', method='pad')

ds.close()

