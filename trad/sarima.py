def sarima(file,gesuchtes_datum,forecast_var):
    import pandas as pd
    import statsmodels.api as sm
    nc_path = file#'../Data/stunden/2022_resample_stunden.nc'
    import xarray as xr

    df = xr.open_dataset(nc_path).to_dataframe()
    # Laden Sie Ihre Temperaturdaten aus der NetCDF-Datei in ein Pandas DataFrame
    #df = pd.read_csv('temperaturdaten.csv', parse_dates=['datetime'], index_col='datetime')
    #gesuchtes_datum=pd.to_datetime("2022-07-16 00:00")
    # Bereiten Sie die Daten für das SARIMA-Modell vor
    # Nehmen wir an, Ihre Temperaturdaten sind in einer Spalte namens 'temperature'

    startindex= df[forecast_var].index.get_loc(gesuchtes_datum)-(4*7*24)
    endindex= df[forecast_var].index.get_loc(gesuchtes_datum)
    temperature = df[forecast_var][startindex:endindex]
    #print(temperature)
    # Trainingsdaten: Die letzten 4 Wochen
    #train_data = temperature[:-4*7*24]
    train_data= temperature
    # Erstellen und fitten Sie das SARIMA-Modell
    order = (0, 1, 1)  # Beispielwerte für AR, I, MA
    seasonal_order = (0, 1, 1, 24)  # Beispielwerte für saisonale AR, I, MA, Saisonlänge
    model = sm.tsa.statespace.SARIMAX(train_data, order=order, seasonal_order=seasonal_order)
    model_fit = model.fit()

    # Generieren Sie Vorhersagen für die nächsten 4 Wochen (24 Stunden pro Tag * 7 Tage pro Woche * 4 Wochen)
    forecast = model_fit.get_forecast(steps=24)#*7*4)

    # Extrahieren Sie die Vorhersageergebnisse
    forecast_values = forecast.predicted_mean
    confidence_interval = forecast.conf_int()

# Drucken Sie die Vorhersageergebnisse
    #print(forecast_values.values)
    return forecast_values.values
#print(confidence_interval)
