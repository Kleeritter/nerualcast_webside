import xarray as xr
import glob
import numpy as np
# Erstellen einer Liste der zu mergenden NetCDF-Dateien
years=np.arange(2007,2022,1)
filelist=[]
for year in years:
    filelist.append("Messwerte_"+str(year)+".nc")
files_to_merge = filelist
#files_to_merge= glob.glob('/home/alex/PycharmProjects/nerualcast/Data/*.nc')


# Öffnen der ersten Datei und Erstellen eines leeren xarray.Dataset-Objekts
merged_data = xr.open_dataset(files_to_merge[0])

# Schleife über die restlichen Dateien und Hinzufügen der Variablen zu merged_data
for file_name in files_to_merge[1:]:

    data = xr.open_dataset(file_name)
    #data['time'] = xr.decode_cf(data['time'], use_cftime=True)
    merged_data = xr.concat([merged_data, data], dim='time')

# Speichern des resultierenden xarray.Dataset-Objekts in einer neuen NetCDF-Datei
merged_data.to_netcdf('merged_file.nc')
