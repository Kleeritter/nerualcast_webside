import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from funcs import TemperatureDataset_conv, TemperatureModel_conv,CustomDataModule
from pytorch_lightning import loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

lite = '../Data/stunden/2016_resample_stunden.nc'
full='../Data/zusammengefasste_datei_2016-2019.nc'
file_path =full # Replace with the actual path to your NetCDF file

forecast_var = 'temp'
window_size=24*4*7
dataset = TemperatureDataset_conv(file_path,window_size=window_size,forecast_horizont=24,forecast_var=forecast_var)
#train_loader = CustomDataModule(train_data, batch_size=24, shuffle=False,num_workers=8,skip_last_batch=True)
#val_loader = CustomDataModule(val_data, batch_size=24, shuffle=False,num_workers=8,skip_last_batch=True)#, sampler=torch.utils.data.SubsetRandomSampler(range(10)))
"""for batch_idx, (inputs, targets) in enumerate(train_loader):
    print(f"Batch {batch_idx+1}:")
    print("Inputs:", inputs)
    print("Targets:", targets)
    print()"""
data_module = CustomDataModule(dataset, batch_size=24, num_workers=4, skip_last_batch=True, test_size=0.3, random_state=42)
model = TemperatureModel_conv(window_size=window_size,forecast_horizont=24,forecast_var=forecast_var,num_channels=17,batch_size=24)
logger = loggers.TensorBoardLogger(save_dir='lightning_logs/conv/'+forecast_var, name='conv')
early_stopping = EarlyStopping('val_loss', patience=10,mode='min')
trainer = pl.Trainer(logger=logger,max_epochs=200, accelerator="auto",devices="auto",callbacks=[early_stopping],deterministic=True)#,val_check_interval=0.5) #log_every_n_steps=2,
trainer.fit(model, datamodule=data_module)

  # Verwende hier den entsprechenden Dataloader (z.B. val_loader)
torch.save(model.state_dict(), 'output/conv/'+forecast_var+'.pth')