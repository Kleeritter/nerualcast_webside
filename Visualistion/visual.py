from Model.old.funcs import  TemperatureModel, get_predictions
import torch
import xarray as xr
import matplotlib.pyplot as plt

model = TemperatureModel()  # Erstelle ein neues Modellobjekt
model.load_state_dict(torch.load('../Model/output/old/lstm_model_old.pth'))
model.eval()  # Setze das Modell in den Evaluationsmodus

selected_day_data = xr.open_dataset("../Data/stunden/2022_resample_stunden.nc")['temp'][0:24].values.reshape((24, 1))
#print(selected_day_data)
#selected_day_data = selected_day_data[:, np.newaxis]  # Shape anpassen (24, num_features) -> (24, 1, num_features)
#selected_day_data = selected_day_data.unsqueeze(0)  # Shape anpassen (24, num_features) -> (1, 24, num_features)
selected_day_data = selected_day_data.np.permute(0, 2, 1)
selected_day_data = torch.from_numpy(selected_day_data).float()  # In einen Tensor konvertieren

predictions = get_predictions(model, selected_day_data)



plt.figure(figsize=(12, 6))
plt.plot(selected_day_data, label="Gemessene Werte")
plt.plot(predictions, label="Vorhersagen")
plt.xlabel("Stunden")
plt.ylabel("Temperatur")
plt.legend()
plt.show()
