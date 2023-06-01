#from netCDF4 import Dataset
import torch
from torch.autograd import Variable

from torch.utils.data import DataLoader

from Model.old.funcs import NetCDFDataset

file_path = '../Data/zehner/normal/2008_normal_zehner.nc'
sliding_window_size = 5  # Größe des Sliding-Window

dataset = NetCDFDataset(file_path, sliding_window_size)
print(dataset)
dataloader = DataLoader(dataset, batch_size=10, shuffle=False)
print(dataloader)
class LSTMModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):

        print(x.size())
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size)).to(device).squeeze()

        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size)).to(device).squeeze()

        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))

        h_out = h_out.view(-1, self.hidden_size)

        out = self.fc(h_out)


        return out


input_size = 1  # Je nach Anzahl der Variablen anpassen
hidden_size = 64
num_layers = 2
output_size = 1

model = LSTMModel(input_size, hidden_size, num_layers, output_size)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Verwende die GPU, falls verfügbar
model.to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10

for epoch in range(num_epochs):
    for batch in dataloader:
        batch = batch.to(device)
        print(batch)
        targets = torch.zeros(batch.shape[0], output_size).to(device)
        # Vorwärtsdurchlauf
        outputs = model(batch)
        loss = criterion(outputs, targets)  # Definiere deine Zielfunktion entsprechend

        # Rückwärtsdurchlauf und Optimierung
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
