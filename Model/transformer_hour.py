import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning import loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.model_selection import train_test_split
from funcs.funcs_tft import TFT_Dataset,TFT_Modell
from funcs.funcs_lstm_multi import TemperatureDataset_multi

torch.set_float32_matmul_precision('medium')

forecast_var = 'temp'
lite = '../Data/stunden/2016_resample_stunden.nc'
full='../Data/zusammengefasste_datei_2016-2019.nc'
file_path =full # Replace with the actual path to your NetCDF file

#Hyperparameter
window_size=4*7*24
forecast_horizont=24
input_dim = 12  # Anzahl der meteorologischen Parameter
output_dim = 1  # Vorhersage der Temperatur

num_layers = 2
num_heads = 4
batch_size = 24
hidden_dim = 12 #3*num_heads
num_encoder_layers = 3
num_decoder_layers = 3
d_model = 12
dropout = 0.1
patience=10
max_epochs=200


dataset = TemperatureDataset_multi(file_path,forecast_horizont=24,window_size=window_size,forecast_var=forecast_var)
train_data, val_data = train_test_split(dataset, test_size=0.3, random_state=42)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False,num_workers=8)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False,num_workers=8)#, sampler=torch.utils.data.SubsetRandomSampler(range(10)))

model = TFT_Modell(input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim, num_layers=num_layers, num_heads=num_heads,forecast_horizont=forecast_horizont,window_size=window_size)
#model =TemperatureForecaster(d_model=d_model,num_encoder_layers=num_encoder_layers,num_decoder_layers=num_decoder_layers,num_heads=num_heads, dropout=dropout)
logger = loggers.TensorBoardLogger(save_dir='lightning_logs/tft/'+forecast_var, name='tft')
early_stopping = EarlyStopping('val_loss', patience=patience,mode='min')
trainer = pl.Trainer(logger=logger,max_epochs=max_epochs, accelerator="auto",devices="auto",val_check_interval=1.0,deterministic=True,
                        callbacks=[early_stopping], accumulate_grad_batches=2) #log_every_n_steps=2,
trainer.fit(model, train_loader,val_loader)
torch.save(model.state_dict(), 'output/tft/'+forecast_var+'._unoptimiert.pth')