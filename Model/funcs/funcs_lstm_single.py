import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

class TemperatureDataset(Dataset):
    def __init__(self, file_path,forecast_horizont=24,window_size=24,forecast_var="temp"):
        import xarray as xr
        self.data = xr.open_dataset(file_path)[forecast_var]#.valuesmissing_values_mask = dataset['temp'].isnull()
        print("max:", max(self.data.values))
        print("min:", min(self.data.values))
        scaler = MinMaxScaler(feature_range=(0, 1))
        self.data=scaler.fit_transform([[x] for x in self.data]).flatten()
        self.length = len(self.data) - window_size
        self.forecast_horizont = forecast_horizont
        self.window_size = window_size
        self.forecast_var=forecast_var


    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        import torch
        import numpy as np
        window_size = self.window_size  # Sliding window size (10 minutes * 144 = 24 hours)
        forecast_horizon = self.forecast_horizont  # How many hours to predict (24 hours)
        start_idx = idx
        end_idx = idx + window_size
        window_data = self.data[start_idx:end_idx]#.values
        target = self.data[end_idx:end_idx+forecast_horizon]#.values

        # Normalize window data and target
        if self.forecast_var == "wind_dir_50":
            # Konvertieren der Windrichtungen in Bogenmaß
            wind_directions_rad = np.deg2rad(window_data)

            # Berechnen des Durchschnitts der Windrichtungen in Bogenmaß
            mean_direction_rad = np.mean(wind_directions_rad)

            # Konvertieren des Durchschnitts zurück in Grad
            mean_direction_deg = np.rad2deg(mean_direction_rad)

            # Subtrahieren des mittleren Winkels von allen Windrichtungen
            normalized_directions_deg = window_data - mean_direction_deg

            # Anpassen der negativen Werte auf den positiven Bereich (0-360 Grad)
            normalized_directions_deg = (normalized_directions_deg + 360) % 360
            window_data= normalized_directions_deg

            # Konvertieren der Windrichtungen in Bogenmaß
            try:
                wind_directions_rad_tar = np.deg2rad(target)

                # Berechnen des Durchschnitts der Windrichtungen in Bogenmaß
                mean_direction_rad_tar = np.mean(wind_directions_rad_tar)

                # Konvertieren des Durchschnitts zurück in Grad
                mean_direction_deg_tar= np.rad2deg(mean_direction_rad_tar)

                # Subtrahieren des mittleren Winkels von allen Windrichtungen
                normalized_directions_deg_tar = target - mean_direction_deg_tar

                # Anpassen der negativen Werte auf den positiven Bereich (0-360 Grad)
                normalized_directions_deg_tar = (normalized_directions_deg_tar + 360) % 360
                target = normalized_directions_deg_tar
            except:
                target = np.zeros_like(target)
        #else:
            #window_data = (window_data - np.mean(window_data)) / np.std(window_data)#, ddof=1)
            #scaler = MinMaxScaler(feature_range=(0, 1))
            #<print(window_data)
            #window_data=scaler.fit_transform([[x] for x in window_data]).flatten()
            #print(window_data)
            # std_target = np.std(target)#, ddof=1)
            # if std_target != 0:
            #     target = (target - np.mean(target)) / std_target
            #else:
            #     target = np.zeros_like(target)

            #target=scaler.fit_transform([[x] for x in target]).flatten()


        # Check if target has exactly 24 hours, otherwise adjust it
        if target.shape[0] < forecast_horizon:
            target = np.pad(target, ((0, forecast_horizon - target.shape[0])), mode='constant')

        # Convert to torch tensors
        #print(len(window_data))
        window_data = window_data.reshape((window_size, 1))
        target = target.reshape((forecast_horizon,))
        window_data = torch.from_numpy(window_data).float()
        target = torch.from_numpy(target).float()
        return window_data, target


class TemperatureModel(pl.LightningModule):
    def __init__(self, hidden_size=32, learning_rate=0.00005, weight_decay=0.0001, optimizer="Adam",dropout=0, num_layers=1, weight_intiliazier="None"):
        super().__init__()
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer=optimizer
        self.dropout=dropout
        self.num_layers=num_layers
        self.weight_intiliazier= weight_intiliazier
        self.lstm = torch.nn.LSTM(input_size=1, hidden_size=self.hidden_size, num_layers=num_layers, batch_first=True,dropout=dropout)
        self.linear = torch.nn.Linear(self.hidden_size, 24)
        self.apply(self.initialize_weights)

    def initialize_weights(self, module):
        match self.weight_intiliazier:
            case "None":
                pass
            case "kaiming":
                for name, param in self.lstm.named_parameters():
                    if 'weight' in name:
                        torch.nn.init.kaiming_uniform_(param, nonlinearity='relu')
            case "normal":
                for name, param in self.lstm.named_parameters():
                    if 'weight' in name:
                        torch.nn.init.normal_(param)
            case "xavier":
                for name, param in self.lstm.named_parameters():
                    if 'weight' in name:
                        torch.nn.init.xavier_uniform_(param)

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
        #optimizer = torch.optim.Adam(self.parameters(), lr=0.00005, weight_decay=0.0001)  # weight_decay-Wert anpassen
        match self.optimizer:
            case "Adam":
                optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
                return optimizer
            case "AdamW":
                optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
                return optimizer





