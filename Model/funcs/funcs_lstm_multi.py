from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
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


class TemperatureDataset_multi(Dataset):
    def __init__(self, file_path,forecast_horizont=24,window_size=24,forecast_var="temp"):
        self.data = xr.open_dataset(file_path)[["wind_dir_50","Geneigt CM-11",'temp',"press_sl","humid","diffuscmp11","globalrcmp11","gust_10","gust_50", "rain", "wind_10", "wind_50"]].to_dataframe()#.valuesmissing_values_mask = dataset['temp'].isnull()
        self.length = len(self.data[forecast_var]) - window_size
       # scaler = MinMaxScaler(feature_range=(0, 1))
       # self.data=scaler.fit_transform([[x] for x in self.data]).flatten()
        for column in self.data.columns:
            if column== "wind_dir_50":
                # Extrahiere die Windrichtungen
                wind_directions_deg = self.data[column].values

                # Konvertiere die Windrichtungen in Bogenmaß
                wind_directions_rad = np.deg2rad(wind_directions_deg)

                # Berechne die Sinus- und Kosinus-Werte der Windrichtungen
                sin_directions = np.sin(wind_directions_rad)
                cos_directions = np.cos(wind_directions_rad)

                # Kombiniere Sinus- und Kosinus-Werte zu einer einzigen Spalte
                combined_directions = np.arctan2(sin_directions, cos_directions)

                # Skaliere die kombinierten Werte auf den Bereich von 0 bis 1
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_directions = scaler.fit_transform(combined_directions.reshape(-1, 1)).flatten()

                self.data[column] = scaled_directions
            else:
                # Erstelle einen neuen Min-Max-Scaler für jede Spalte
                scaler = MinMaxScaler()

                # Extrahiere die Werte der aktuellen Spalte und forme sie in das richtige Format um
                values = self.data[column].values.reshape(-1, 1)

                # Skaliere die Werte der Spalte
                scaled_values = scaler.fit_transform(values)

                # Aktualisiere die Daten mit den skalierten Werten
                self.data[column] = scaled_values.flatten()

        #print(self.data)
        self.forecast_horizont = forecast_horizont
        self.window_size = window_size
        self.forecast_var = forecast_var

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        start_idx = idx
        end_idx = idx + self.window_size
        window_data = self.data.iloc[start_idx:end_idx]#isel(index=slice(start_idx, end_idx)).to_array()#.values#self.data[start_idx:end_idx].values
        target = self.data[self.forecast_var][end_idx:end_idx+self.forecast_horizont]#.values
       # window_data_normalized = np.zeros((window_data.shape[0],self.window_size))#np.zeros_like(window_data)
        #print(window_data_normalized)

        #for i in range(window_data.shape[0]):
         #   if i != 0:
          #      variable = window_data[i, :]
           #     mean = np.mean(variable)
            #    std = np.std(variable)
             #   if std != 0:
              #      variable_normalized = (variable - mean) / std
              #  else:
               #     variable_normalized = np.zeros_like(variable)
                #window_data_normalized[i, :] = variable_normalized
            #else:
             #   variable = window_data[i, :]
                #print(variable)
                #normalized_directions = variable % 360
                #numeric_directions = (normalized_directions / 45).astype(int) % 8
                #windrichtungen = ((variable + 22.5) // 45 % 8).astype(int)
                #encoder = preprocessing.OneHotEncoder(categories=[np.arange(8)], sparse_output=False)
                #enc =np.transpose(encoder.fit_transform(windrichtungen.reshape(-1, 1)))
                #for j in range(0,7):
                 #   window_data_normalized[i+j, :] = enc[j]#
              #  wind_directions_rad = np.deg2rad(variable)
                #print(wind_directions_rad)
                # Berechnen des Durchschnitts der Windrichtungen in Bogenmaß
               # mean_direction_rad = np.mean(wind_directions_rad)

                # Konvertieren des Durchschnitts zurück in Grad
                #mean_direction_deg = np.rad2deg(mean_direction_rad)

                # Subtrahieren des mittleren Winkels von allen Windrichtungen
                #normalized_directions_deg = variable- mean_direction_deg
                #print(normalized_directions_deg)
                # Anpassen der negativen Werte auf den positiven Bereich (0-360 Grad)
                #normalized_directions_deg = (normalized_directions_deg + 360) % 360
                #print(normalized_directions_deg)
                #window_data_normalized[i, :] = normalized_directions_deg

        #if self.forecast_var == "wind_dir_50":
         #   try:
          #      wind_directions_rad_tar = np.deg2rad(target)
#
                # Berechnen des Durchschnitts der Windrichtungen in Bogenmaß
 #               mean_direction_rad_tar = np.mean(wind_directions_rad_tar)

                # Konvertieren des Durchschnitts zurück in Grad
  #              mean_direction_deg_tar = np.rad2deg(mean_direction_rad_tar)

                # Subtrahieren des mittleren Winkels von allen Windrichtungen
   #             normalized_directions_deg_tar = target - mean_direction_deg_tar

                # Anpassen der negativen Werte auf den positiven Bereich (0-360 Grad)
    #            normalized_directions_deg_tar = (normalized_directions_deg_tar + 360) % 360

     #           target = normalized_directions_deg_tar
      #      except:
       #         target = np.zeros_like(target)

        #else:
         #   std_target = np.std(target)#, ddof=1)
          #  if std_target != 0:
             #   target = (target - np.mean(target)) / std_target
           # else:
            #    target = np.zeros_like(target)

        # Check if target has exactly 24 hours, otherwise adjust it
        if target.shape[0] < self.forecast_horizont:
            target = np.pad(target, ((0, self.forecast_horizont - target.shape[0])), mode='constant')
        # Convert to torch tensors
        #window_data = window_data_normalized.transpose(1, 0)
        #print(window_data)
        target = np.array(target).reshape((self.forecast_horizont,))
        window_data = torch.from_numpy(np.array(window_data)).float()#[:, np.newaxis]).float()
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
    def __init__(self, window_size=24, forecast_horizont=24,num_layers=1,hidden_size=40, learning_rate=0.001, weight_decay=0.001):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.lstm = torch.nn.LSTM(input_size=12, hidden_size=hidden_size, num_layers=num_layers,batch_first=True)
        self.linear = torch.nn.Linear(hidden_size, forecast_horizont)
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)  # weight_decay-Wert anpassen
        return optimizer
