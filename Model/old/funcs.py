import torch
import pytorch_lightning as pl
import xarray as xr
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
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

import torch
import pytorch_lightning as pl
import xarray as xr
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
class TemperatureDataset(Dataset):
    def __init__(self, file_path):
        import torch
        import pytorch_lightning as pl
        import xarray as xr
        import numpy as np
        from torch.utils.data import Dataset, DataLoader
        from sklearn.model_selection import train_test_split
        import seaborn as sns
        import matplotlib.pyplot as plt
        self.data = xr.open_dataset(file_path)['temp']
        #print(self.data.isnull().all())
        self.length = len(self.data) - 24

    def __len__(self):
        return self.length#len(self.data)

    def __getitem__(self, idx):
        import torch
        import pytorch_lightning as pl
        import xarray as xr
        import numpy as np
        from torch.utils.data import Dataset, DataLoader
        from sklearn.model_selection import train_test_split
        import seaborn as sns
        import matplotlib.pyplot as plt
        window_size = 24  # Sliding window size (10 minutes * 144 = 24 hours)
        start_idx = idx
        end_idx = idx + window_size
        window_data = self.data[start_idx:end_idx, :, :].values#self.data[start_idx:end_idx].values

        target = self.data[end_idx:end_idx+24, :, :].values#self.data[end_idx].values

        # Normalize window data and target
        window_data = (window_data - np.mean(window_data)) / np.std(window_data)
        std_target = np.std(target)
        if std_target != 0:
            target = (target - np.mean(target)) / std_target
        else:
            # Handle the case when the standard deviation is zero
            target = np.zeros_like(target)  # Set the normalized target to zero or any other appropriate value

        #target = (target - np.mean(target)) / np.std(target)

        # Reshape window data and target because of wrong directions
        # Check if target has exactly 24 hours, otherwise adjust it
        if target.shape[0] < 24:
            target = np.pad(target, ((0, 24 - target.shape[0]), (0, 0), (0, 0)), mode='constant')

        window_data = window_data.reshape((window_size, 1))
        target = target.reshape((24,))
        #print(window_data)
        # Convert to torch tensors
        window_data = torch.from_numpy(window_data).float()
        target = torch.from_numpy(target).float()
        #print(window_data)
        #print(target)
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
