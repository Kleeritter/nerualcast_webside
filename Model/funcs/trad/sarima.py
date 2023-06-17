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

