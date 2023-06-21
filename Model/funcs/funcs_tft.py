from sklearn import preprocessing
import torch
import pytorch_lightning as pl
import xarray as xr
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn#
import math

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
        #print(len(target))
        #print(window_data.shape)
        return window_data, target




class TFT_Modell(pl.LightningModule):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, num_heads,dropout=0.1,forecast_horizont=24,window_size=24,forecast_var="temp"):
        super(TFT_Modell, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        # Eingabe-Embedding-Schicht
        self.embedding = nn.Linear(input_dim, hidden_dim)

        # Positionale Encoding-Schicht
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #print(self.device)
        self.devices=device
        self.positional_encoding = self.get_positional_encoding(hidden_dim).to(device)
        self.input_pos_embedding = torch.nn.Embedding(1024, embedding_dim=hidden_dim)
        self.target_pos_embedding = torch.nn.Embedding(1024, embedding_dim=hidden_dim)

        # Encoder-Schichten
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads,dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Decoder-Schichten
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=num_heads,dropout=dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Ausgabe-Linear-Schicht
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.do = nn.Dropout(p=self.dropout)
        self.ff=nn.Linear(window_size,forecast_horizont)
    def get_positional_encoding(self, d_model, max_len=1000):#"cuda:0"):
        # Positionale Encoding-Schicht erzeugen
        #print(self.devices)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        positional_encoding = pe.unsqueeze(0)
        #device = next(self.parameters()).device
        positional_encoding = positional_encoding.to(self.devices)
        return positional_encoding
    def forward(self, x):
        # Eingabe-Embedding
        #print(x.shape)
        x = self.embedding(x)

        # Positionale Codierung hinzufügen
        x = x + self.positional_encoding[:, :x.size(1), :]

        # Encoder-Schichten
        x = self.encoder(x)

        # Decoder-Schichten
        x = self.decoder(x, memory=self.encoder(x))

        # Ausgabe-Linear-Schicht
        x = self.fc(x)
        x=self.ff(torch.squeeze(x))
        return x

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

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = nn.MSELoss()(outputs, targets)
        self.log('test_loss', loss)
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=0.001)  # weight_decay-Wert anpassen
        return optimizer
