import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from funcs import TemperatureDataset_conv, TemperatureModel_conv,CustomDataModule

file_path ='../Data/zusammengefasste_datei_2016-2019.nc'#'../Data/stunden/2016_resample_stunden.nc'#'../Data/zusammengefasste_datei_2016-2019.nc' #'#zusammengefasste_datei_2016-2019.nc'#'../Data/stunden/2020_resample_stunden.nc'  # Replace with the actual path to your NetCDF file
dataset = TemperatureDataset_conv(file_path)
#train_loader = CustomDataModule(train_data, batch_size=24, shuffle=False,num_workers=8,skip_last_batch=True)
#val_loader = CustomDataModule(val_data, batch_size=24, shuffle=False,num_workers=8,skip_last_batch=True)#, sampler=torch.utils.data.SubsetRandomSampler(range(10)))
"""for batch_idx, (inputs, targets) in enumerate(train_loader):
    print(f"Batch {batch_idx+1}:")
    print("Inputs:", inputs)
    print("Targets:", targets)
    print()"""
data_module = CustomDataModule(dataset, batch_size=24, num_workers=4, skip_last_batch=True, test_size=0.3, random_state=42)
model = TemperatureModel_conv()
trainer = pl.Trainer(max_epochs=200, accelerator="auto",devices="auto",val_check_interval=1.0) #log_every_n_steps=2,
trainer.fit(model, datamodule=data_module)
torch.save(model.state_dict(), 'output/lstm_model_conv.pth')