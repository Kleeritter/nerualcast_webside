import xarray as xr
import pandas as pd
import glob
import numpy as np
from scipy.interpolate import interp1d
# Öffnen der NetCDF-Datei und Auswählen der Variablen, die resampled werden sollen
filelist=sorted(glob.glob("/home/alex/PycharmProjects/nerualcast/Data/einer/*.nc"))
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
        #print(ds)
        time_index = pd.to_datetime(ds['time'].values, unit='s')
        # values =xr.Dataset(coords=dict(time=ds["time"].resample(time="10T",origin="epoch")))
        vars = ["humid", "temp", "press", "press_sl", "dewpoint_calc", "ptd", "ptm", "wind_10", "wind_50",
                "wind_dir_50", "gust_10", "gust_50", "rain","Geneigt CM-11","globalrcm11","globalrcmp11"]# "globalrcm11"]
        values = ds[vars].isel(time=time_index.minute % 60 == 0)  # time_index.minute%10==0
        print(len(values["time"]))
        start_date = str(years[i])+'-01-01'
        end_date = str(years[i])+'-12-31'
        hourly_range = pd.date_range(start=start_date, end=end_date, freq='H')
        dfs = pd.DataFrame(index=hourly_range)
        #dfs = dfs.join(values.to_dataframe())#pd.concat([dfs,values.to_dataframe()],join="inner")
        #print(dfs)
        # Schleife durch alle Variablen im xarray.Dataset-Objekt
        for var_name, var in values.variables.items():
            # Resampling der Variablen auf stündliche Werte
            if var_name != "time" and values[var_name].isnull().all() != True and var_name=="temp":
                # Resampling auf stündliche Werte
                if var_name == "rain":
                    hourly_var = ds[var_name].resample(time='1H', origin="epoch").sum()
                    # ds[var_name] = hourly_var
                    values[var_name] = hourly_var
                    #df[var_name]=df.join(values[var_name]).asfreq('H')
                    # values.assign(var_name= hourly_var)
                elif var_name== "wind_dir_50":
                    hourly_var = ds[var_name].resample(time='1H', origin="epoch").mean()
                    hourly_var[hourly_var < 0] = 0
                    ds[var_name] = hourly_var
                    values[var_name] = hourly_var
                else:
                    hourly_var = ds[var_name].resample(time='1H', origin="epoch").mean()
                    ds[var_name] = hourly_var
                    values[var_name] = hourly_var
                    #df[var_name]=df.join(values[var_name]).asfreq('H')#pd.Series(values[var_name], index=hourly_range).reindex(df.index)

        #            values.assign(var_name=hourly_var)

        # Speichern des resultierenden xarray.Dataset-Objekts in einer neuen NetCDF-Datei
        dfs = dfs.join(values.to_dataframe())#, left_index=True, right_index=True)#dfs.join(values).asfreq('H')
        print(dfs)
        groups = dfs.groupby(pd.Grouper(freq='D'))
        missing_days = [group for group, group_df in groups if len(group_df) - group_df['temp'].count() > 3]
        #print(dfs.loc[missing_days[0]])
        # Approximiere fehlende Stundenwerte (max. 3 aufeinanderfolgende Stunden)
        #for day in missing_days:
         #   day_df = dfs.loc[day]
          #  #f = interp1d(day_df.index, values['temp'], kind='linear')
           # #day_df['temp'] =f(day_df.index)
             #day_df['temp'].interpolate(limit=3)
            #day_df = day_df.interpolate(method='linear')#,limit=3)
            #dfs.loc[day] = day_df

        # Entferne ganze Tage mit größeren Lücken
        df_cleaned = dfs.interpolate(method='linear')#dfs.drop(missing_days)
        df_cleaned.loc[df_cleaned['wind_dir_50'] < 0, 'wind_dir_50'] = 0   #     print(len(df_cleaned.index.tolist()))
        df_cleaned.to_xarray().to_netcdf("stunden/" + str(years[i]) + '_resample_stunden.nc')  # .asfreq(freq='10T', method='pad')

        ds.close()
    return

#for i in range(len(filelist)):
    #print(years[i])
    #print(filelist[i])
    #ds = xr.open_dataset(filelist[i])
    #print(len(ds["time"]))
#resample_zehner(filelist,years)
#resample_stunden(filelist,years)
resample_stunden(["einer/Messwerte_2016.nc","einer/Messwerte_2017.nc","einer/Messwerte_2018.nc","einer/Messwerte_2019.nc","einer/Messwerte_2020.nc","einer/Messwerte_2021.nc","einer/Messwerte_2022.nc"],[2016,2017,2018,2019,2020,2021,2022])