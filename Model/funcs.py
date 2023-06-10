
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



class TemperatureDataset_conv(Dataset):
    def __init__(self, file_path, window_size=6, num_channels=17, batch_size=24,forecast_horizont=24,forecast_var="temp"):
        import xarray as xr
        self.data = xr.open_dataset(file_path)[
            ['temp', "press_sl", "humid", "Geneigt CM-11", "gust_10", "gust_50", "rain", "wind_10", "wind_50",
             "wind_dir_50"]]
        self.window_size = window_size
        self.num_channels = num_channels
        self.batch_size = batch_size
        self.length = len(self.data[forecast_var]) - window_size - (len(self.data[forecast_var]) - window_size) % batch_size
        self.forecast_horizont=forecast_horizont
        self.forecast_var=forecast_var
    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        window_size = self.window_size
        num_channels = self.batch_size

        start_idx = idx
        end_idx = idx + window_size
        window_data = self.data.isel(index=slice(start_idx, end_idx)).to_array().values
        target = self.data[self.forecast_var][end_idx:end_idx + self.forecast_horizont].values
        # Überprüfe, ob dies der letzte Batch ist

        window_data_normalized = np.zeros((window_size, num_channels))
        #print(window_data_normalized.shape)
        #print(window_data[0])
        # Order of channels: air temperature, relative humidity, air pressure, cloud coverage,
        # wind speed, wind direction (one-hot encoded), hourly precipitation, month, hour

        window_data_normalized[:, 0] = (window_data[ 0] - np.mean(window_data[ 0])) / np.std(
            window_data[ 0])  # air temperature
        window_data_normalized[:, 2] = (window_data[ 1] - np.mean(window_data[ 1])) / np.std(
            window_data[ 1])  # air pressure
        window_data_normalized[:, 1] = (window_data[ 2] - np.mean(window_data[ 2])) / np.std(
            window_data[ 2])  # humid
        window_data_normalized[:, 3] = (window_data[ 3] - np.mean(window_data[ 3])) / np.std(
            window_data[ 3])  # Globalstrahlung
        if np.std(window_data[ 4])!=0:
            window_data_normalized[:,4] = (window_data[ 4] - np.mean(window_data[ 4])) / np.std(
            window_data[ 4])  # gust_10
        else:
            window_data_normalized[:, 4] = np.zeros_like(window_data_normalized[:, 4])

        window_data_normalized[:,5] = (window_data[ 5] - np.mean(window_data[ 5])) / np.std(
            window_data[ 5])  # gust_50
        if np.std(window_data[ 6])!=0:
            window_data_normalized[:, -1] = (window_data[ 6] - np.mean(window_data[ 6])) / np.std(
            window_data[ 6])  # hourly precipitation
        else:
            window_data_normalized[:, -1] = np.zeros_like(window_data_normalized[:, 6])
        if np.std(window_data[7]) != 0:
            window_data_normalized[:, 6] = (window_data[ 7] - np.mean(window_data[ 7])) / np.std(
            window_data[ 7])  # wind_10
        else:
            window_data_normalized[:, 6] = np.zeros_like(window_data_normalized[:, 7])
        window_data_normalized[:, 7] = (window_data[ 8] - np.mean(window_data[ 8])) / np.std(
            window_data[8])  # wind_50
        #print(window_data_normalized.shape)
        # One-hot encode wind direction
        wind_direction = window_data[9]
        normalized_directions = wind_direction % 360
        numeric_directions = (normalized_directions / 45).astype(int) % 8
        encoder = preprocessing.OneHotEncoder(categories=[np.arange(8)], sparse_output=False)
        enc = encoder.fit_transform(numeric_directions.reshape(-1, 1))

        window_data_normalized[:, 8:-1] = enc

        std_target = np.std(target)
        if std_target != 0:
            target = (target - np.mean(target)) / std_target
        else:
            target = np.zeros_like(target)

        #window_data = window_data_normalized.transpose(1, 0)
        #window_data = window_data_normalized.transpose(1, 2, 0)
        #print(target)
        if target.shape[0] < self.forecast_horizont:
            target = np.pad(target, ((0, self.forecast_horizont - target.shape[0])), mode='constant')
        target = target.reshape((self.forecast_horizont,))
        #print(window_data_normalized.shape)
        #print(window_data)
        window_data = torch.from_numpy(window_data_normalized).float()
        target = torch.from_numpy(target).float()

        return window_data, target


class TemperatureModel_conv(pl.LightningModule):
    def __init__(self, window_size, num_channels, batch_size, forecast_horizont, forecast_var):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding='same')
        self.conv2 = nn.Conv2d(8, 8, kernel_size=2, stride=1, padding='same')
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(8, 16, kernel_size=2, stride=1, padding='same')
        self.conv4 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding='same')
        self.conv5 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding='same')
        self.conv6 = nn.Conv2d(32, 64, kernel_size=2, stride=1, padding='same')
        self.lstm = nn.LSTM(4, 24, num_layers=2, dropout=0.2)
        self.fc1 = nn.Linear(64*1*4, 512)#, dropout=0.2)
        self.fc2 = nn.Linear(512, 64)#, dropout=0.2)
        self.fc3 = nn.Linear(64, 24)#, dropout=0.2)
        self.flat= nn.Flatten()
        self.init_weights()
        self.window_size = window_size
    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                torch.nn.init.xavier_uniform_(module.weight)
    def forward(self, x):
        #x=x.permute(0,2,1)
        print(x.shape)
        x = x.view(x.size(0), 1, self.window_size, 17)
        #print(x.shape)
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = self.maxpool(x)
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = self.maxpool(x)
        x = F.leaky_relu(self.conv5(x))
        x = F.leaky_relu(self.conv6(x))
        #print(x.shape)
        #x = x.view(x.size(0), -1).permute(1, 0).unsqueeze(2)
        #print(x)
        #x = x.view(x.size(0) * x.size(1), -1)
        #x = x.view(x.size(0), -1, x.size(1))
        """
        x = x.squeeze(2)  # Shape: (24, 64, 4)
        x = x.permute(1, 0, 2)  # Shape: (64, 24, 4)
        x,_= self.lstm(x)
        """

        #print(x.shape)
        #x = x.squeeze(0)
        #print(x.shape)
        #x=x[-1:, :, :].squeeze(0)
        #x = x[:, -1, :]
        #x = x.view(x.size(0) * x.size(1), -1)
       # x = x.flatten(start_dim=1)  # Shape: (64, 576)
        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))  # Flatten des Tensors  # Flatten des Tensors
        #print(x.shape)
        #x=x[:, -1, :]
        #x = x.view(x.size(0) * x.size(1), -1)
        #print(x.shape)
        #x=x[-1]
        #x=
        #x = x[-24 :].view(24, 24)   # Extracting the last time step output
        #print(x.shape)
        x = F.leaky_relu(self.fc1(x))
        #print(x.shape)
        x = F.leaky_relu(self.fc2(x))
        #x=x.squeeze()
        #x=self.flat(x)
        #print(x.shape)
        x = self.fc3(x)#F.relu(self.fc3(x))
        #rint(x.shape())
        #x=x.squeeze(0)
        #x=x.transpose(0, 1).view(24, 24)
        #print(x.shape)
        #x = x.view(-1, x.size(1), self.fc3.out_features)
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
        optimizer = optim.Adam(self.parameters(), lr=0.00005, weight_decay=0.0001)
        return optimizer


import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

class CustomDataModule(LightningDataModule):
    def __init__(self, data, batch_size=1, num_workers=0, skip_last_batch=False, test_size=0.3, random_state=42):
        super().__init__()
        self.data = data
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.skip_last_batch = skip_last_batch
        self.test_size = test_size
        self.random_state = random_state

    def setup(self, stage=None):
        train_data, val_data = train_test_split(self.data, test_size=self.test_size, random_state=self.random_state)
        self.train_dataset = CustomDataset(train_data)
        self.val_dataset = CustomDataset(val_data)

    def train_dataloader(self):
        num_samples = len(self.train_dataset)
        if self.skip_last_batch and num_samples % self.batch_size != 0:
            num_samples -= num_samples % self.batch_size
            dataset = self.train_dataset[:num_samples]
        else:
            dataset = self.train_dataset

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        num_samples = len(self.val_dataset)
        if self.skip_last_batch and num_samples % self.batch_size != 0:
            num_samples -= num_samples % self.batch_size
            dataset = self.val_dataset[:num_samples]
        else:
            dataset = self.val_dataset

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )


