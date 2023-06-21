def sarima(data):
    import pandas as pd
    import statsmodels.api as sm
    #nc_path = file#'../Data/stunden/2022_resample_stunden.nc'
    import xarray as xr

    #df = xr.open_dataset(nc_path).to_dataframe()

    #startindex= df[forecast_var].index.get_loc(gesuchtes_datum)-(wind)
    #endindex= df[forecast_var].index.get_loc(gesuchtes_datum)
    #temperature = df[forecast_var][startindex:endindex]

    train= data
    #print(train)
    train.index = pd.DatetimeIndex(train.index.values,
                                   freq="H")
    # Erstellen und fitten Sie das SARIMA-Modell
    order = (0, 1, 1)  # Beispielwerte für AR, I, MA
    seasonal_order = (0, 1, 1, 24)  # Beispielwerte für saisonale AR, I, MA, Saisonlänge
    model = sm.tsa.statespace.SARIMAX(train, order=order, seasonal_order=seasonal_order)
    model_fit = model.fit(disp=0)

    # Generieren Sie Vorhersagen für die nächsten 4 Wochen (24 Stunden pro Tag * 7 Tage pro Woche * 4 Wochen)
    forecast = model_fit.get_forecast(steps=24)#*7*4)

    # Extrahieren Sie die Vorhersageergebnisse
    forecast_values = forecast.predicted_mean
    confidence_interval = forecast.conf_int()

    return forecast_values.values

def fullsarima(forecast_var="temp"):
    import datetime
    import xarray as xr
    import numpy as np
    # Passe die folgenden Variablen entsprechend an
    #forecast_var = "temp"  # Which variable to forecast
    window_size = 24 * 7 * 4  # How big is the window for training
    forecast_horizon = 24  # How long to cast in the future
    forecast_year = 2022  # Which year to forecast
    dt = datetime.datetime(forecast_year, 1, 1, 0, 0)  # + datetime.timedelta(hours=window_size)
    dtl = datetime.datetime(forecast_year - 1, 12, 31, 23)
    dtlast = dtl - datetime.timedelta(hours=window_size + 24)
    nc_path = '../../../Data/stunden/' + str(
        forecast_year) + '_resample_stunden.nc'  # Replace with the actual path to your NetCDF file
    nc_path_last = '../../../Data/stunden/' + str(forecast_year - 1) + '_resample_stunden.nc'
    data = xr.open_dataset(nc_path)  # .to_dataframe()#["index">dt]
    datalast = xr.open_dataset(nc_path_last)
    # print(data.to_dataframe().iloc[-1])
    data = xr.concat([datalast, data], dim="index").to_dataframe()
    start_index_forecast = data.index.get_loc(dtlast)
    start_index_visual = data.index.get_loc(dt)
    forecast_data = data[start_index_forecast:-1]
    visual_data = data[start_index_visual:-1]
    referencor=[]
    i=0
    for window, last_window in zip(range(window_size, len(forecast_data.index.tolist()), forecast_horizon),
                                          range(0, len(forecast_data.index.tolist()) - window_size, forecast_horizon)):
        refenrence = sarima(forecast_data[forecast_var][last_window:window])
        referencor.append((refenrence))
        print(i)
        i+=1

    np.save(file="reference_new_"+forecast_var,arr=np.array(referencor))
    return

#fullsarima("wind_dir_50")

