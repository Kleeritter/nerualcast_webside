import glob


def normalize(folder):
    from sklearn import preprocessing
    import xarray as xr
    import numpy as np
    import glob
    years=np.arange(2022,2007,-1)
    from sklearn.impute import SimpleImputer
    filelist = glob.glob(str(folder)+"/*.nc")
    #print(glob.glob("zehner/*nc"))
    for i in range(len(filelist)):
        ds = xr.open_dataset(filelist[i])

        for var_name, var in ds.variables.items():
            print(var_name)
            if var_name == "wind_dir_50":
                print("windir")
                data = ds["wind_dir_50"].values.reshape(-1, 1)  # .reshape(-1,1)
                # Windrichtungen auf den Bereich von 0 bis 360 Grad normalisieren
                normalized_directions = data % 360

                # Windrichtungen in numerische Werte umwandeln (0 bis (Anzahl der einzigartigen Richtungen - 1))
                # Windrichtungen in numerische Werte umwandeln (0 bis 7)
                numeric_directions = (normalized_directions / 45).astype(int) % 8

                encoder = preprocessing.OneHotEncoder(sparse=False, categories="auto")
                enc = encoder.fit_transform(numeric_directions.reshape(-1, 1))
                print(len(enc[0]))
                print(enc[-1])
                variable_names = [f'Windrichtung_{i * 45}' for i in range(8)]
                separate_variables = {name: (['time'], enc[:, i]) for i, name in enumerate(variable_names)}
                ds.update(separate_variables)
            elif var_name!= "time":
             #   pass

                data= ds[var_name].values.reshape(-1,1)
                nan_count = np.isnan(data).sum()

                print("Anzahl der NaN-Werte:", nan_count, var_name)
                #print(np.squeeze(data).ndim)
                #imputer = SimpleImputer(strategy='mean')
                if nan_count >0 :
                    #print(np.squeeze(data))
                    data = np.nan_to_num(data, nan=np.nanmean(np.squeeze(data))).reshape(-1,1)
                normalizedata=preprocessing.normalize(data,axis=0)
                normalized_data_array = xr.DataArray(normalizedata.flatten(), coords={'time': ds['time']}, dims=['time'])

                #print(normalized_data_array)
    # Erstellen eines neuen xarray-Datasets mit den normalisierten Daten
                ds[var_name]= normalized_data_array

            else:
                pass
            ds.to_netcdf(str(folder)+"/normal/" + str(years[i])+'_normal_'+str(folder)+'.nc')  # .asfreq(freq='10T', method='pad')


        ds.close()
    return


normalize("zehner")