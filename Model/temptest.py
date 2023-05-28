import netCDF4
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from netCDF4 import Dataset
# Pfad zur NetCDF-Datei

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
file_path = '../Data/zehner/normal/2008_normal_zehner.nc'

# NetCDF-Datei öffnen
dataset = netCDF4.Dataset(file_path)

# Variablen laden
temperature = dataset.variables['temp'][:].squeeze()
time = dataset.variables['time'][:]

# NetCDF-Datei schließen
dataset.close()

import torch
import torch.nn as nn

# Definieren der Hyperparameter
window_size = 10  # Fenstergröße für das Sliding Window
input_size = 2  # Anzahl der Eingangsmerkmale (Temperatur und Zeit)
hidden_size = 64  # Größe der versteckten Schicht im LSTM
output_size = 1  # Anzahl der Ausgänge (vorhergesagte Temperatur)

# Erstellen der Eingangs- und Ausgangsdaten für das LSTM
inputs = []
outputs = []
for i in range(len(temperature) - window_size):
    input_data = np.column_stack((temperature[i:i+window_size], time[i:i+window_size]))
    output_data = temperature[i+window_size]
    inputs.append(input_data)
    outputs.append(output_data)

inputs = np.array(inputs, dtype=np.float32)  # Konvertieren in NumPy-Array
outputs = np.array(outputs, dtype=np.float32)  # Konvertieren in NumPy-Array

# Konvertieren in Tensoren
inputs = torch.from_numpy(inputs).to(device)
outputs = torch.from_numpy(outputs).to(device)

# Aufteilen der Daten in Trainings- und Testsets
train_size = int(0.8 * len(inputs))
train_inputs, test_inputs = inputs[:train_size], inputs[train_size:]
train_outputs, test_outputs = outputs[:train_size], outputs[train_size:]

# Definieren des LSTM-Modells
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        _, (hidden, _) = self.lstm(input)
        output = self.fc(hidden[-1])
        return output

model = LSTM(input_size, hidden_size, output_size).to(device)
#model.to(device)
# Definieren des Verlustfunktion und Optimierers
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training des LSTM-Modells
num_epochs = 100
batch_size = 16

for epoch in range(num_epochs):
    permutation = torch.randperm(train_inputs.size(0))
    for i in range(0, train_inputs.size(0), batch_size):
        indices = permutation[i:i+batch_size]
        batch_inputs, batch_outputs = train_inputs[indices], train_outputs[indices]
        #batch_inputs, batch_outputs= batch_inputs.to(device), batch_outputs.to(device)
        optimizer.zero_grad()
        output = model(batch_inputs.permute(1, 0, 2))
        loss = criterion(output.squeeze(), batch_outputs)
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch: {epoch+1}/{num_epochs}, Loss: {loss.item()}')

# Auswerten des Modells auf dem Testset
with torch.no_grad():
    test_output = model(test_inputs.unsqueeze(1))
    test_loss = criterion(test_output.squeeze(), test_outputs)
    print(f'Test Loss: {test_loss.item()}')


