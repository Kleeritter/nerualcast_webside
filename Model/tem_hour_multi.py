import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from funcs import TemperatureDataset_multi, TemperatureModel_multi_full

file_path ='../Data/zusammengefasste_datei_2016-2019.nc' #'../Data/stunden/2016_resample_stunden.nc'#zusammengefasste_datei_2016-2019.nc'#'../Data/stunden/2020_resample_stunden.nc'  # Replace with the actual path to your NetCDF file
dataset = TemperatureDataset_multi(file_path)
train_data, val_data = train_test_split(dataset, test_size=0.3, random_state=42)
train_loader = DataLoader(train_data, batch_size=24, shuffle=False,num_workers=8)
val_loader = DataLoader(val_data, batch_size=24, shuffle=False,num_workers=8)#, sampler=torch.utils.data.SubsetRandomSampler(range(10)))

model = TemperatureModel_multi_full()
trainer = pl.Trainer(max_epochs=200, accelerator="auto",devices="auto",val_check_interval=1.0) #log_every_n_steps=2,
trainer.fit(model, train_loader,val_loader)
torch.save(model.state_dict(), 'output/lstm_model_multi_9var.pth')