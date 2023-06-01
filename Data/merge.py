from netCDF4 import Dataset
import os
import xarray as xr
# Erstellen Sie eine neue NetCDF-Datei zum Zusammenfassen der Daten

# Erstellen Sie eine leere Dataset zum Zusammenführen der Daten
combined_dataset = xr.Dataset()

# Iterieren Sie über die Jahre 2010-2014
for year in range(2016, 2019):
    input_file = f"stunden/{year}_resample_stunden.nc"  # Annahme: Die Dateien haben das Format "datei_<jahr>.nc"

    # Überprüfen Sie, ob die Eingabedatei existiert
    if os.path.isfile(input_file):
        input_dataset = xr.open_dataset(input_file)  # Öffnen Sie die Eingabedatei

        # Führen Sie die Eingabedatei mit der kombinierten Dataset zusammen
        combined_dataset = xr.merge([combined_dataset, input_dataset])

        input_dataset.close()  # Schließen Sie die Eingabedatei

# Speichern Sie das kombinierte Dataset in eine neue Datei
output_file = "zusammengefasste_datei_2016-2019.nc"
combined_dataset.to_netcdf(output_file)

# Schließen Sie das kombinierte Dataset
combined_dataset.close()
