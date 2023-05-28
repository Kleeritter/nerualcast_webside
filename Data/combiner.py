import xarray as xr

# Liste der Dateipfade
filepaths = ['Messwerte_2007.nc', 'Messwerte_2008.nc', 'Messwerte_2009.nc']

# Öffnen Sie jede Datei als Dataset
datasets = [xr.open_dataset(fp) for fp in filepaths]

# Fügen Sie die Variablen der einzelnen Datasets zu einer Liste zusammen
all_variables = []
for ds in datasets:
    all_variables += ds.variables
# Entfernen Sie doppelte Variablennamen
unique_variables = list(set(all_variables))

# Kombinieren Sie die Datasets entlang der Zeitdimension und allen Variablen
#merged_datasets = [ds.groupby('Stationsdaten') for ds in datasets]
merged_dataset =xr.merge(datasets, join='outer') #xr.merge(datasets, combine='nested',compat='override')

# Speichern Sie das resultierende Dataset als NetCDF-Datei
merged_dataset.to_netcdf('mergeder.nc')
merged_dataset.close()
