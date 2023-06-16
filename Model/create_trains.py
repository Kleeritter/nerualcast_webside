import torch
from sklearn.model_selection import train_test_split
from funcs.funcs_lstm_single  import TemperatureDataset
import pytorch_lightning as pl
import numpy as np
import random
import xarray as xr
pl.seed_everything(42)

# Setze den Random Seed für torch
torch.manual_seed(42)

# Setze den Random Seed für random
random.seed(42)

# Setze den Random Seed für numpy
np.random.seed(42)
forecast_vars=["wind_dir_50",'temp',"press_sl","humid","diffuscmp11","globalrcmp11","gust_10","gust_50", "rain", "wind_10", "wind_50"]
#forecast_var = 'temp'
window_sizes= [6,12,24,48,72,24*7,24*2*7,24*7*3,24*7*4]
#window_size= 24
lite = '../Data/stunden/2016_resample_stunden.nc'
full='../Data/zusammengefasste_datei_2016-2019.nc'
file_path =full # Replace with the actual path to your NetCDF file
#print(xr.open_dataset(file_path)["globalrcmp11"])
for forecast_var in forecast_vars:
    print(forecast_var)
    for window_size in window_sizes:
        training_data_path = 'opti/storage/training_data_lstm_single_train_'+forecast_var+"_"+str(window_size)+'.pt'
        val_data_path = 'opti/storage/training_data_lstm_single_val_'+forecast_var+"_"+str(window_size)+'.pt'


        dataset = TemperatureDataset(file_path,window_size=window_size,forecast_horizont=24,forecast_var=forecast_var)

        # Convert train_data to NumPy arrays
        train_data, val_data = train_test_split(dataset, test_size=0.3, random_state=42)
        #train_data_np = np.array(dataset)
        torch.save(train_data, training_data_path)
        torch.save(val_data, val_data_path)
        print("saved")


# Save the NumPy arrays
#np.savez(file_path, train_data=train_data_np)
# Load the training data from the file
#train_data = torch.load(training_data_path)
#val_data = torch.load(val_data_path)



