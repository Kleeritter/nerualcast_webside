import torch
import pytorch_lightning as pl
import xarray as xr
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from funcs import TemperatureDataset, TemperatureModel

file_path = '../Data/zusammengefasste_datei.nc'#'../Data/stunden/2020_resample_stunden.nc'  # Replace with the actual path to your NetCDF file
dataset = TemperatureDataset(file_path)
train_data, val_data = train_test_split(dataset, test_size=0.3, random_state=42)
#train_loader = DataLoader(dataset, batch_size=24, shuffle=False)
train_loader = DataLoader(train_data, batch_size=24, shuffle=False)
val_loader = DataLoader(val_data, batch_size=24, shuffle=False)

model = TemperatureModel()
trainer = pl.Trainer(max_epochs=200, accelerator="auto",devices="auto",val_check_interval=1.0) #log_every_n_steps=2,
trainer.fit(model, train_loader,val_loader)

  # Verwende hier den entsprechenden Dataloader (z.B. val_loader)
torch.save(model.state_dict(), 'lstm_model_old.pth')