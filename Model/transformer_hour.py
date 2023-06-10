import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.model_selection import train_test_split
from funcs_tft import TFT_Dataset,TFT_Modell


forecast_var = 'humid'
lite = '../Data/stunden/2016_resample_stunden.nc'
full='../Data/zusammengefasste_datei_2016-2019.nc'
file_path =lite # Replace with the actual path to your NetCDF file

#Hyperparameter
window_size=24
input_dim = 17  # Anzahl der meteorologischen Parameter
output_dim = 1  # Vorhersage der Temperatur
hidden_dim = 64
num_layers = 4
num_heads = 4
batch_size = 32



dataset = TFT_Dataset(file_path,forecast_horizont=24,window_size=window_size,forecast_var=forecast_var)
train_data, val_data = train_test_split(dataset, test_size=0.3, random_state=42)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False,num_workers=8)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False,num_workers=8)#, sampler=torch.utils.data.SubsetRandomSampler(range(10)))

model = TFT_Modell(input_dim, output_dim, hidden_dim, num_layers, num_heads)
logger = loggers.TensorBoardLogger(save_dir='lightning_logs/tft/'+forecast_var, name='tft')
early_stopping = EarlyStopping('val_loss', patience=10,mode='min')
trainer = pl.Trainer(logger=logger,max_epochs=200, accelerator="auto",devices="auto",val_check_interval=1.0,deterministic=True,
                        callbacks=[early_stopping]) #log_every_n_steps=2,
trainer.fit(model, train_loader,val_loader)
torch.save(model.state_dict(), 'output/tft/'+forecast_var+'.pth')