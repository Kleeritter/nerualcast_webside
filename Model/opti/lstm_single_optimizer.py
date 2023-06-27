import optuna_dashboard,optuna
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from pytorch_lightning import loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from funcs.funcs_lstm_single import TemperatureDataset, TemperatureModel
from optuna.integration import PyTorchLightningPruningCallback
import random
import numpy as np
import yaml
forecast_var = 'temp'
# Setzen Sie die Zufallssaat für die GPU
# Setze den Random Seed für PyTorch
pl.seed_everything(42)

# Setze den Random Seed für torch
torch.manual_seed(42)

# Setze den Random Seed für random
random.seed(42)

# Setze den Random Seed für numpy
np.random.seed(42)

torch.set_float32_matmul_precision('medium')
def objective(trial):
    # Define the hyperparameters to optimize
    learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-1,log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-1,log=True)
    hidden_size = trial.suggest_categorical('hidden_size', [1,2,4,8,16, 32, 64,128])
    #optimizer  = trial.suggest_categorical('optimizer', ["Adam","AdamW"])
    #dropout = trial.suggest_categorical('dropout', [0,0.2,0.5])
    num_layers = trial.suggest_categorical('num_layers', [1, 2, 3,4])
    batchsize = trial.suggest_categorical('batchsize', [6,12,24,8])
    weight_intiliazier = trial.suggest_categorical('weight_intiliazier', ["None", "xavier","kaiming","normal"])
    window_size= trial.suggest_categorical('window_size', [24,48,72,24*7,24*2*7,24*7*3,24*7*4])
    # Initialize the model with the suggested hyperparameters
    training_data_path = 'storage/training_data_lstm_single_train_' + forecast_var + "_" + str(window_size) + '.pt'
    val_data_path = 'storage/training_data_lstm_single_val_' + forecast_var + "_" + str(window_size) + '.pt'
    train_data = torch.load(training_data_path)
    val_data = torch.load(val_data_path)

    # Create data loaders for training and validation
    train_loader = DataLoader(train_data, batch_size=batchsize, shuffle=False, num_workers=8)
    val_loader = DataLoader(val_data, batch_size=batchsize, shuffle=False, num_workers=8)

    model = TemperatureModel(hidden_size=hidden_size, learning_rate=learning_rate, weight_decay=weight_decay,num_layers=num_layers,weight_intiliazier=weight_intiliazier)
    logger = loggers.TensorBoardLogger(save_dir='../lightning_logs/lstm_uni/' + forecast_var, name='lstm_optimierer')

    # Define the Lightning callbacks and trainer settings
    early_stopping = EarlyStopping('val_loss', patience=10, mode='min')
    pruning_callback = PyTorchLightningPruningCallback(trial, monitor='val_loss')
    trainer = pl.Trainer(logger=logger, max_epochs=30, accelerator="auto", devices="auto",
                        callbacks=[early_stopping,pruning_callback], deterministic=True,enable_progress_bar=False)

    # Train the model using the pre-loaded train and validation loaders
    trainer.fit(model, train_loader, val_loader)
    #trial.report(trainer.callback_metrics['val_loss'], epoch=trainer.current_epoch)

    # Handle pruning based on the reported value
    #if trial.should_prune():
     #   raise optuna.exceptions.TrialPruned()

    # Return the performance metric you want to optimize (e.g., validation loss)
    return trainer.callback_metrics['val_loss'].item()


study = optuna.create_study(direction='minimize', storage='sqlite:///storage/database.db',study_name="LSTM-Single_tester")
study.optimize(objective, n_trials=100)
best_params = study.best_trial.params
print(best_params)
# Erhalte die besten Parameter und speichere sie in einer Datei
with open('output/lstm_single/best_params_lstm_single'+forecast_var+'.yaml', 'w') as file:
    yaml.dump(best_params, file)