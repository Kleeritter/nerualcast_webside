import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset

class TemperatureDataset(Dataset):
    def __init__(self, file_path,forecast_horizont=24,window_size=24,forecast_var="temp"):
        import xarray as xr
        self.data = xr.open_dataset(file_path)[forecast_var]#.valuesmissing_values_mask = dataset['temp'].isnull()
        self.length = len(self.data) - window_size
        self.forecast_horizont = forecast_horizont
        self.window_size = window_size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        import torch
        import numpy as np
        window_size = self.window_size  # Sliding window size (10 minutes * 144 = 24 hours)
        forecast_horizon = self.forecast_horizont  # How many hours to predict (24 hours)
        start_idx = idx
        end_idx = idx + window_size
        window_data = self.data[start_idx:end_idx].values
        target = self.data[end_idx:end_idx+forecast_horizon].values

        # Normalize window data and target
        window_data = (window_data - np.mean(window_data)) / np.std(window_data)#, ddof=1)
        std_target = np.std(target)#, ddof=1)
        if std_target != 0:
            target = (target - np.mean(target)) / std_target
        else:
            target = np.zeros_like(target)

        # Check if target has exactly 24 hours, otherwise adjust it
        if target.shape[0] < forecast_horizon:
            target = np.pad(target, ((0, forecast_horizon - target.shape[0])), mode='constant')

        # Convert to torch tensors
        window_data = window_data.reshape((window_size, 1))
        target = target.reshape((24,))
        window_data = torch.from_numpy(window_data).float()
        target = torch.from_numpy(target).float()
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



