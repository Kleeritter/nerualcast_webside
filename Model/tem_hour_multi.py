import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning import loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.model_selection import train_test_split
from funcs.funcs_lstm_multi import TemperatureDataset_multi, TemperatureModel_multi_full

##  IO  ##
lite = '../Data/stunden/2016_resample_stunden.nc'
full='../Data/zusammengefasste_datei_2016-2019.nc'
file_path =full # Replace with the actual path to your NetCDF file
torch.set_float32_matmul_precision('medium') # Utilizing Tensorcores in CUDA Devices

##  HyperParameters   ##
window_size = 4*7*24
forecast_horizont = 24
num_layers = 1
hidden_size = 40
learning_rate = 0.001
weight_decay = 0.001
batch_size= 24
patience=10
max_epochs=200
forecast_var = 'temp'



dataset = TemperatureDataset_multi(file_path,forecast_horizont=forecast_horizont,window_size=window_size,forecast_var=forecast_var)
train_data, val_data = train_test_split(dataset, test_size=0.3, random_state=42)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False,num_workers=8)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False,num_workers=8)#, sampler=torch.utils.data.SubsetRandomSampler(range(10)))

model = TemperatureModel_multi_full(forecast_horizont=forecast_horizont,window_size=window_size,num_layers=num_layers,hidden_size=hidden_size,learning_rate=learning_rate,weight_decay=weight_decay)
logger = loggers.TensorBoardLogger(save_dir='lightning_logs/lstm_multi/'+forecast_var, name='multi_full')
early_stopping = EarlyStopping('val_loss', patience=patience,mode='min')
trainer = pl.Trainer(logger=logger,max_epochs=max_epochs, accelerator="auto",devices="auto",val_check_interval=1.0,deterministic=True,
                        callbacks=[early_stopping]) #log_every_n_steps=2,
trainer.fit(model, train_loader,val_loader)
torch.save(model.state_dict(), 'output/lstm_multi/'+forecast_var+'_unoptimiert.pth')