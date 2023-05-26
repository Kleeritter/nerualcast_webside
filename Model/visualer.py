import torch
import seaborn as sns
import xarray as xr
import matplotlib.pyplot as plt
from funcs import TemperatureModel
import numpy as np
# Passe den Dateipfad entsprechend an
checkpoint_path = 'lstm_model_old.pth'
checkpoint = torch.load(checkpoint_path)
print(checkpoint.keys())

# Passe die Architektur deines LSTM-Modells entsprechend an
model = TemperatureModel()  # Ersetze "YourLSTMModel" durch den tatsächlichen Namen deines Modells
model.load_state_dict(checkpoint)#['state_dict'])
model.eval()


# Passe den Dateipfad entsprechend an
nc_path = '../Data/stunden/2022_resample_stunden.nc'
data = xr.open_dataset(nc_path)
real_values = data['temp'][:24].values.reshape((24,))
real_valueser= data['temp'][:24].values#.reshape((24,1))# Passe den Namen der Wert-Variable entsprechend an
print(real_values)
# Erstelle das Sliding Window für die Vorhersage
sliding_window = []  # Liste für das Sliding Window

# Führe die Vorhersage für die ersten 24 Stunden durch
predicted_values = []



sliding_window= real_valueser
original_mean = np.mean(sliding_window)  # original_data sind die nicht normalisierten Daten
original_std = np.std(sliding_window)
sliding_window = (sliding_window - np.mean(sliding_window)) / np.std(sliding_window)
# Berechnung des Durchschnitts und der Standardabweichung des Originaldatensatzes

input_data = torch.Tensor(sliding_window)
predicted_value = model(input_data)
predicted_values.append(predicted_value.tolist())
flattened_predicted_values = [value for sublist in predicted_values[0] for value in sublist]
print(predicted_values[0][0])
denormalized_values = [(predicted_value * original_std )+ original_mean for predicted_value in predicted_values[0][0]]
print(denormalized_values)

##predicted_values.append(predicted_value.item())
# Passe die Achsenbezeichnungen und das Layout entsprechend an
plt.figure(figsize=(10, 6))
sns.lineplot(x=range(24), y=real_values, label='Real values')
sns.lineplot(x=range(24), y=denormalized_values, label='Predicted values')
plt.xlabel('Zeit')
plt.ylabel('Werte')
plt.title('Vorhersagequalität des LSTM-Modells')
plt.legend()
plt.show()
