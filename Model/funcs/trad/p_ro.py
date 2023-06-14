def pp(file,gesuchtes_datum,forecast_var):
    import pandas as pd
    from prophet import Prophet
    import logging

    logging.getLogger("prophet").setLevel(logging.WARNING)
    logging.getLogger("cmdstanpy").disabled = True

    nc_path = file#'../Data/stunden/2022_resample_stunden.nc'
    import xarray as xr

    df = xr.open_dataset(nc_path).to_dataframe()
    # Laden Sie Ihre Temperaturdaten aus der NetCDF-Datei in ein Pandas DataFrame
    #df = pd.read_csv('temperaturdaten.csv', parse_dates=['datetime'], index_col='datetime')
    #gesuchtes_datum=pd.to_datetime("2022-07-16 00:00")
    # Bereiten Sie die Daten für Prophet vor
    # Nehmen wir an, Ihre Temperaturdaten sind in einer Spalte namens 'temperature'

    #print(df)
    # Trainingsdaten: Die letzten 24 Stunden
    startindex = df[forecast_var].index.get_loc(gesuchtes_datum) - (4 * 7 * 24)
    endindex = df[forecast_var].index.get_loc(gesuchtes_datum)
    temperature = df[forecast_var][startindex:endindex]

    df = temperature.reset_index().rename(columns={'index': 'ds', 'temp': 'y'})
    train_data = df
    # Erstellen und fitten Sie das Prophet-Modell
    model = Prophet()
    model.fit(train_data)

    # Generieren Sie Vorhersagen für die nächsten 24 Stunden
    future = model.make_future_dataframe(periods=24, freq='H')
    forecast = model.predict(future)

    # Extrahieren Sie die Vorhersageergebnisse
    forecast_values = forecast[['ds', 'yhat']].tail(24)
    forecast_values_array = forecast_values['yhat'].values
    confidence_interval = forecast[['ds', 'yhat_lower', 'yhat_upper']].tail(24)
    return forecast_values_array
    # Drucken Sie die Vorhersageergebnisse
    #print(forecast_values_array)
    #print(confidence_interval)
