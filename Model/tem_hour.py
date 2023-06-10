import torch
import pytorch_lightning as pl
import xarray as xr
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from pytorch_lightning import loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from funcs_lstm_single import TemperatureDataset, TemperatureModel
forecast_var = 'temp'

lite = '../Data/stunden/2016_resample_stunden.nc'
full='../Data/zusammengefasste_datei_2016-2019.nc'
file_path =full # Replace with the actual path to your NetCDF file

dataset = TemperatureDataset(file_path,window_size=4*7*24,forecast_horizont=24,forecast_var=forecast_var)
train_data, val_data = train_test_split(dataset, test_size=0.3, random_state=42)
train_loader = DataLoader(train_data, batch_size=24, shuffle=False,num_workers=8)
val_loader = DataLoader(val_data, batch_size=24, shuffle=False,num_workers=8)#, sampler=torch.utils.data.SubsetRandomSampler(range(10)))

model = TemperatureModel()
logger = loggers.TensorBoardLogger(save_dir='lightning_logs/lstm_uni/'+forecast_var, name='lstm_uni')
early_stopping = EarlyStopping('val_loss', patience=10,mode='min')
trainer = pl.Trainer(logger=logger,max_epochs=200, accelerator="auto",devices="auto",callbacks=[early_stopping],deterministic=True)#,val_check_interval=0.5) #log_every_n_steps=2,
trainer.fit(model, train_loader,val_loader)

  # Verwende hier den entsprechenden Dataloader (z.B. val_loader)
torch.save(model.state_dict(), 'output/lstm_uni/'+forecast_var+'.pth')