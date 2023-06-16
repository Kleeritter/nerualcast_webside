import torch
from pytorch_lightning import LightningModule
from torch.nn import TransformerEncoder,TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
from sklearn import preprocessing
import torch
import xarray as xr
import numpy as np
from torch.utils.data import Dataset, DataLoader


class TFT_Dataset(Dataset):
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
        target = target.reshape((self.forecast_horizont,))
        window_data = torch.from_numpy(window_data).float()#[:, np.newaxis]).float()
        #print(window_data.shape)
        target = torch.from_numpy(target).float()
        #print(window_data.shape)
        return window_data, target
class TemperatureForecaster(LightningModule):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 d_model: int,
                 num_heads: int,
                 dropout: float):
        super().__init__()
        self.encoder = TransformerEncoder(TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dropout=dropout), num_layers=num_encoder_layers,
        )

        self.decoder = TransformerDecoder(TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dropout=dropout,), num_layers=num_decoder_layers,
        )

        self.linear = torch.nn.Linear(2176, 672)
        self.linear2= torch.nn.Linear(672, 24)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x,memory=self.encoder(x))
        x = x.transpose(1, 2)  # Vertauschen der Dimensionen für den Linearkombinationslayer
        x = self.linear(x.squeeze(dim=1))
        x=self.linear2(x)
        return x

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        x, y = batch
        y_hat = torch.squeeze(self(x))
        #print(len(y_hat))
        #print(len(y))
        loss = torch.nn.MSELoss()(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        x, y = batch
        y_hat = torch.squeeze(self(x))
        loss = torch.nn.MSELoss()(y_hat, y)
        self.log('val_loss', loss)

    def test_step(self, batch: torch.Tensor, batch_idx: int):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.MSELoss()(y_hat, y)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
