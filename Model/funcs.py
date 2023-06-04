
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn import preprocessing
import torch
import pytorch_lightning as pl
import xarray as xr
import numpy as np
from torch.utils.data import Dataset, DataLoader


class NetCDFDataset(Dataset):
    import pandas as pd
    import xarray as xr

    # from netCDF4 import Dataset
    import netCDF4

    def __init__(self, file_path, sliding_window_size):
        import xarray as xr
        self.dataset = xr.open_dataset(file_path)
        self.sliding_window_size = sliding_window_size

    def __len__(self):
        return len(self.dataset['temp'])

    def __getitem__(self, index):
        import numpy as np
        start_index = max(0, index - self.sliding_window_size + 1)
        end_index = index + 1

        temperature = self.dataset['temp'][start_index:end_index].values
        if temperature.shape[0] < self.sliding_window_size:
            temperature = np.pad(temperature, [(self.sliding_window_size - temperature.shape[0], 0)],
                                 mode='constant')

        #print(temperature)
        # Füge hier weitere Variablen hinzu, falls nötig

        return temperature

class TemperatureDataset(Dataset):
    def __init__(self, file_path):
        import xarray as xr
        self.data = xr.open_dataset(file_path)['temp']#.valuesmissing_values_mask = dataset['temp'].isnull()

# Zeige die Daten mit den fehlenden Werten an
        data_array = xr.open_dataset(file_path)['temp'].values

        # Überprüfe, ob es fehlende Werte gibt
        is_missing = np.isnan(data_array)

        # Zähle die Anzahl der fehlenden Werte
        num_missing = np.sum(is_missing)

        # Zeige die Anzahl der fehlenden Werte an
        print("Anzahl der NaN-Werte:", num_missing)

        # Finde die Positionen der fehlenden Werte
        missing_positions = np.argwhere(is_missing)
        print("Positionen der NaN-Werte:")
        for position in missing_positions:
            print(position)

        self.length = len(self.data) - 24

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        import torch
        import numpy as np
        window_size = 24  # Sliding window size (10 minutes * 144 = 24 hours)
        start_idx = idx
        end_idx = idx + window_size
        window_data = self.data[start_idx:end_idx].values
        print(window_data)

        target = self.data[end_idx:end_idx+24].values

        # Normalize window data and target
        window_data = (window_data - np.mean(window_data)) / np.std(window_data)#, ddof=1)
        std_target = np.std(target)#, ddof=1)
        if std_target != 0:
            target = (target - np.mean(target)) / std_target
        else:
            target = np.zeros_like(target)

        # Check if target has exactly 24 hours, otherwise adjust it
        if target.shape[0] < 24:
            target = np.pad(target, ((0, 24 - target.shape[0])), mode='constant')

        #if target.shape[0] < 24:
         #   target = np.pad(target, ((0, 24 - target.shape[0]), (0, 0), (0, 0)), mode='constant')
        #print(window_data)
        # Convert to torch tensors
        window_data = window_data.reshape((window_size, 1))
        target = target.reshape((24,))
        window_data = torch.from_numpy(window_data).float()#[:, np.newaxis]).float()
        #target = target.reshape((-1, 24))
        #target = target.squeeze()  # Entfernt die letzte Dimension mit Größe 1
        target = torch.from_numpy(target).float()
        #print(window_data)
        return window_data, target


class TemperatureModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size=1, hidden_size=32, num_layers=1, batch_first=True)
        self.linear = torch.nn.Linear(32, 24)

    def forward(self, x):
        #print(x)
        lstm_output, _ = self.lstm(x)
        output = self.linear(lstm_output[:, -1, :])
        #print(output)
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        #print(x)
        loss = torch.nn.MSELoss()(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.MSELoss()(y_hat, y)
        self.log('val_loss', loss)
        return loss
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.00005, weight_decay=0.0001)  # weight_decay-Wert anpassen
        return optimizer

def get_predictions(model, dataset):
    model.eval()
    predictions = []

    with torch.no_grad():
        for i in range(len(dataset)):
            window_data = dataset[i]
            window_data = window_data#.unsqueeze(0).permute(0, 2, 1)
            y_hat = model(window_data)
            predictions.append(y_hat.item())

    return predictions

class TemperatureDataset_multi(Dataset):
    def __init__(self, file_path):
        import xarray as xr
        self.data = xr.open_dataset(file_path)[['temp',"press_sl","humid","Geneigt CM-11","gust_10","gust_50", "rain", "wind_10", "wind_50","wind_dir_50"]]#.valuesmissing_values_mask = dataset['temp'].isnull()
        self.length = len(self.data["temp"]) - 24

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        import torch
        import numpy as np
        window_size = 24  # Sliding window size (10 minutes * 144 = 24 hours)
        start_idx = idx
        end_idx = idx + window_size
        window_data = self.data.isel(index=slice(start_idx, end_idx)).to_array().values#self.data[start_idx:end_idx].values
        target = self.data["temp"][end_idx:end_idx+24].values
        window_data_normalized = np.zeros((window_data.shape[0]+7,24))#np.zeros_like(window_data)
        #print(window_data_normalized)

        for i in range(window_data.shape[0]):
            if i != 9:
                variable = window_data[i, :]
                mean = np.mean(variable)
                std = np.std(variable)
                if std != 0:
                    variable_normalized = (variable - mean) / std
                else:
                    variable_normalized = np.zeros_like(variable)
                window_data_normalized[i, :] = variable_normalized
            else:
                variable = window_data[i, :]
                normalized_directions = variable % 360
                numeric_directions = (normalized_directions / 45).astype(int) % 8
                #print(numeric_directions)
                #windrichtungen = np.floor((variable % 360) / 45).astype(int)
                # Einteilung der Werte in Richtungen
                windrichtungen = ((variable + 22.5) // 45 % 8).astype(int)

                # One-Hot-Encoding
                encoder = preprocessing.OneHotEncoder(categories=[np.arange(8)], sparse_output=False)
                enc =np.transpose(encoder.fit_transform(windrichtungen.reshape(-1, 1)))
                #print(np.transpose(enc))
                #print(enc)
                for j in range(0,7):
                    window_data_normalized[i+j, :] = enc[j]#
        #print(window_data_normalized)#len(window_data_normalized))
        std_target = np.std(target)#, ddof=1)
        if std_target != 0:
            target = (target - np.mean(target)) / std_target
        else:
            target = np.zeros_like(target)

        # Check if target has exactly 24 hours, otherwise adjust it
        if target.shape[0] < 24:
            target = np.pad(target, ((0, 24 - target.shape[0])), mode='constant')
        # Convert to torch tensors
        window_data = window_data_normalized.transpose(1, 0)
        target = target.reshape((24,))
        window_data = torch.from_numpy(window_data).float()#[:, np.newaxis]).float()
        target = torch.from_numpy(target).float()
        return window_data, target


class TemperatureModel_multi_light(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size=3, hidden_size=32, num_layers=1, batch_first=True)
        self.linear = torch.nn.Linear(32, 24)

    def forward(self, x):
        lstm_output, _ = self.lstm(x)
        output = self.linear(lstm_output[:, -1, :])
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.MSELoss()(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.MSELoss()(y_hat, y)
        self.log('val_loss', loss)
        return loss
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=0.001)  # weight_decay-Wert anpassen
        return optimizer

class TemperatureModel_multi_full(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size=17, hidden_size=40, num_layers=1,batch_first=True)
        self.linear = torch.nn.Linear(40, 24)
        self.apply(self.initialize_weights)

    def initialize_weights(self, module):
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_uniform_(param, nonlinearity='relu')

    def forward(self, x):
        lstm_output, _ = self.lstm(x)
        output = self.linear(lstm_output[:, -1, :])
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.MSELoss()(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.MSELoss()(y_hat, y)
        self.log('val_loss', loss)
        return loss
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=0.001)  # weight_decay-Wert anpassen
        return optimizer


class TemperatureDataset_conv(Dataset):
    def __init__(self, file_path):
        self.data = xr.open_dataset(file_path)[
            ['temp', 'press_sl', 'humid', 'Geneigt CM-11', 'gust_10', 'gust_50', 'rain', 'wind_10', 'wind_50']]
        self.length = len(self.data["temp"]) - 24

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        window_size = 6  # Sliding window size in hours
        window_shift = 1  # Shift of the sliding window in hours
        start_idx = idx
        end_idx = idx + window_size
        window_data = self.data.isel(index=slice(start_idx, end_idx)).to_array().values
        target = self.data["temp"][end_idx:end_idx + 24].values

        window_data_normalized = (window_data - np.mean(window_data)) / np.std(window_data)
        target_normalized = (target - np.mean(target)) / np.std(target)

        # Check if target has exactly 24 hours, otherwise adjust it
        if target_normalized.shape[0] < 24:
            target_normalized = np.pad(target_normalized, ((0, 24 - target_normalized.shape[0])), mode='constant')

        window_data_normalized = np.expand_dims(window_data_normalized, axis=0)
        target_normalized = np.expand_dims(target_normalized, axis=0)

        window_data = torch.from_numpy(window_data_normalized).float()
        target = torch.from_numpy(target_normalized).float()

        return window_data, target



class TemperatureModel_conv(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.conv1= nn.Conv2d(9, 32, kernel_size=6, stride=1)
        self.fc = nn.Linear(32 * 19, 24)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001, weight_decay=0.001)
        return optimizer