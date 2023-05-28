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
nc_path = '../Data/stunden/2007_resample_stunden.nc'
data = xr.open_dataset(nc_path)
time= data["time"][4580:4580+24]
print(len(data["time"]))
print(time)
real_values = data['temp'][4580:4580+24].values.reshape((24,))
real_valueser= data['temp'][4580-24:4580 ].values#.reshape((24,1))# Passe den Namen der Wert-Variable entsprechend an
print(real_valueser)
# Erstelle das Sliding Window für die Vorhersage
sliding_window = []  # Liste für das Sliding Window

# Führe die Vorhersage für die ersten 24 Stunden durch
predicted_values = []



sliding_window= real_valueser
original_mean = np.mean(sliding_window)  # original_data sind die nicht normalisierten Daten
original_std = np.std(sliding_window)
sliding_window = (sliding_window - np.mean(sliding_window)) / np.std(sliding_window)
# Berechnung des Durchschnitts und der Standardabweichung des Originaldatensatzes

sliding_window=np.expand_dims(sliding_window, axis=0)
sliding_window=np.expand_dims(sliding_window, axis=2)
#input_data = torch.Tensor(sliding_window)
input_data = torch.from_numpy(sliding_window).float()
with torch.no_grad():
    predicted_value = model(input_data)
predicted_values.append(predicted_value.tolist())
predictions=predicted_value.squeeze().tolist()
print(predictions)
flattened_predicted_values = [value for sublist in predicted_values[0] for value in sublist]
print(predicted_values[0][0])
denormalized_values = [(predicted_value * original_std )+ original_mean for predicted_value in predicted_values[0][0]]
denormalized_values = [(predicted_value * original_std )+ original_mean for predicted_value in predictions]

print(denormalized_values)

##predicted_values.append(predicted_value.item())
# Passe die Achsenbezeichnungen und das Layout entsprechend an
sns.set_theme(style="darkgrid")
plt.figure(figsize=(10, 6))
plt.ylim(273, 300)
sns.lineplot(x=time, y=real_values, label='Real values')
sns.lineplot(x=time, y=denormalized_values, label='Predicted values')
plt.xlabel('Zeit')
plt.ylabel('Temperatur in K')
plt.title('Vorhersagequalität des LSTM-Modells')
plt.legend()
plt.show()
