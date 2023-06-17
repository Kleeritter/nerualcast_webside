import torch
import pytorch_lightning as pl
import xarray as xr
import numpy as np
from torch.utils.data import Dataset, DataLoader


class TemperatureDataset(Dataset):
    def __init__(self, file_path):
        self.data = xr.open_dataset(file_path)['temp']
        self.length = len(self.data) - 144

    def __len__(self):
        return self.length#len(self.data)

    def __getitem__(self, idx):
        window_size = 144  # Sliding window size (10 minutes * 144 = 24 hours)
        start_idx = idx
        end_idx = idx + window_size
        window_data = self.data[start_idx:end_idx, :, :].values#self.data[start_idx:end_idx].values
        target = self.data[end_idx, :, :].values#self.data[end_idx].values

        # Normalize window data and target
        window_data = (window_data - np.mean(window_data)) / np.std(window_data)
        std_target = np.std(target)
        if std_target != 0:
            target = (target - np.mean(target)) / std_target
        else:
            # Handle the case when the standard deviation is zero
            target = np.zeros_like(target)  # Set the normalized target to zero or any other appropriate value

        # Reshape window data and target because of wrong directions
        window_data = window_data.reshape((window_size, 1))
        target = target.reshape((1,))

        # Convert to torch tensors
        window_data = torch.from_numpy(window_data).float()
        target = torch.from_numpy(target).float()
        #print(window_data)
        return window_data, target


class TemperatureModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size=1, hidden_size=64, num_layers=2, batch_first=True)
        self.linear = torch.nn.Linear(64, 1)

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

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


file_path = '../Data/zehner/2008_resample_zehner.nc'  # Replace with the actual path to your NetCDF file
dataset = TemperatureDataset(file_path)
train_loader = DataLoader(dataset, batch_size=12, shuffle=True)

model = TemperatureModel()
trainer = pl.Trainer(max_epochs=10, accelerator="auto",devices="auto")
trainer.fit(model, train_loader)
