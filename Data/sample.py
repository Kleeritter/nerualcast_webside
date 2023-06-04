import xarray as xr

# Öffnen der NetCDF-Datei und Auswählen der Variablen, die resampled werden sollen
ds = xr.open_dataset('Messwerte_2007.nc')
dsv = xr.open_dataset('Messwerte_2007.nc')


var = dsv['variablename']

# Resampling der Variablen auf stündliche Werte
hourly_var = var.resample(time='1H').mean()

# Speichern der resultierenden stündlichen Werte in einer neuen NetCDF-Datei
hourly_var.to_netcdf('merged_file_hourly.nc')


for var_name, var in ds.variables.items():
    # Resampling der Variablen auf stündliche Werte
    if 'time' in var.dims:
        ds['time'] = xr.decode_cf(ds['time'])
        hourly_var = var.resample(time='1H').mean()
        ds[var_name] = hourly_var