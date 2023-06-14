from sklearn import preprocessing
import torch
import pytorch_lightning as pl
import xarray as xr
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

class TemperatureDataset_multi(Dataset):
    def __init__(self, file_path,forecast_horizont=24,window_size=24,forecast_var="temp"):
        self.data = xr.open_dataset(file_path)[['temp',"press_sl","humid","Geneigt CM-11","gust_10","gust_50", "rain", "wind_10", "wind_50","wind_dir_50"]]#.valuesmissing_values_mask = dataset['temp'].isnull()
        self.length = len(self.data[forecast_var]) - window_size
        self.forecast_horizont = forecast_horizont
        self.window_size = window_size
        self.forecast_var = forecast_var

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        start_idx = idx
        end_idx = idx + self.window_size
        window_data = self.data.isel(index=slice(start_idx, end_idx)).to_array().values#self.data[start_idx:end_idx].values
        target = self.data[self.forecast_var][end_idx:end_idx+self.forecast_horizont].values
        window_data_normalized = np.zeros((window_data.shape[0]+7,self.window_size))#np.zeros_like(window_data)
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
        if target.shape[0] < self.forecast_horizont:
            target = np.pad(target, ((0, self.forecast_horizont - target.shape[0])), mode='constant')
        # Convert to torch tensors
        window_data = window_data_normalized.transpose(1, 0)
        #print(window_data)
        target = target.reshape((self.forecast_horizont,))
        window_data = torch.from_numpy(window_data).float()#[:, np.newaxis]).float()
        target = torch.from_numpy(target).float()
        #print(window_data.shape)
        return window_data, target


class TemperatureModel_multi_light(pl.LightningModule):
    def __init__(self, window_size=24, forecast_horizont=24):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size=3, hidden_size=32, num_layers=1, batch_first=True)
        self.linear = torch.nn.Linear(32, forecast_horizont)

    def forward(self, x):
        #print(x.shape)
        lstm_output, _ = self.lstm(x)
        output = self.linear(lstm_output[:, -1, :])
        #print(output.shape)
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
    def __init__(self, window_size=24, forecast_horizont=24):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size=17, hidden_size=40, num_layers=1,batch_first=True)
        self.linear = torch.nn.Linear(40, forecast_horizont)
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
        self.log('val_loss', loss, prog_bar=False)
        return loss
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=0.001)  # weight_decay-Wert anpassen
        return optimizer
