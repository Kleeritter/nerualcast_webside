import xarray as xr

# Liste der Pfade zu den NetCDF-Dateien
file_paths = ["Messwerte_2007.nc", "Messwerte_2008.nc", "Messwerte_2009.nc"]

# Laden und Zusammenführen der NetCDF-Dateien
ds = xr.open_mfdataset(file_paths, group="Stationsdaten")

# Ausgabe des kombinierten Datasets
print(ds)

# Schließen der NetCDF-Dateien
ds.close()
