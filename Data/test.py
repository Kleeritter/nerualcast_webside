


import numpy as np
import math
def dew_point(T, tt):

    """Berechnet den Taupunkt in Grad Celsius mit der Goff-Gratch-Gleichung."""
    a = 7.5
    b = 237.3
    ef=6.112*math.exp((17.62*tt)/(243.12+tt))
    # Berechnung der relativen Luftfeuchtigkeit


    ee=ef-0.622*(T-tt)
    RH = (ee / ef)
    print(ef)
    print(RH*100)
    alpha = ((a * T) / (b + T)) + np.log10(RH)
    T_dp = (b * alpha) / (a - alpha)
    return T_dp

# Werte für die Trockentemperatur und die relative Luftfeuchtigkeit
dry_bulb_temp = 25
#relative_humidity = 65.2
tt=18

# Berechnung des Taupunkts
T_dp = dew_point(dry_bulb_temp, tt)
print("Der Taupunkt beträgt: {:.1f} °C".format(T_dp))

print(np.NaN +273.15)



import netCDF4 as nc

# Öffnen der NetCDF-Datei im Schreibmodus
dataset = nc.Dataset('example.nc', 'r+')

# Überprüfen der Variablen auf Leere
for varname, var in dataset.variables.items():
    if var[:].size == 0:
        print(f"Variable '{varname}' ist leer und wird aus der Datei entfernt.")
        dataset.variables.pop(varname)
        dataset.sync()

# Schließen der NetCDF-Datei
dataset.close()
