def start_index_real(nc_path,gesuchtes_datum,window_size=24,forecast_horizon=24):
    import xarray as xr
    data = xr.open_dataset(nc_path)
    dataf = data.to_dataframe()
    start_index_real = dataf.index.get_loc(gesuchtes_datum)-window_size
    return start_index_real
def end_index_real(nc_path,gesuchtes_datum,window_size=24,forecast_horizon=24):
    import xarray as xr
    data = xr.open_dataset(nc_path)
    dataf = data.to_dataframe()
    end_index_real = dataf.index.get_loc(gesuchtes_datum) +forecast_horizon
    return end_index_real
def start_index_test(nc_path,gesuchtes_datum,window_size=24,forecast_horizon=24):
    import xarray as xr
    data = xr.open_dataset(nc_path)
    dataf = data.to_dataframe()
    start_index_test= dataf.index.get_loc(gesuchtes_datum) -window_size
    return start_index_test
def end_index_test(nc_path,gesuchtes_datum,window_size=24,forecast_horizon=24):
    import xarray as xr
    data = xr.open_dataset(nc_path)
    dataf = data.to_dataframe()
    end_index_test= dataf.index.get_loc(gesuchtes_datum)
    return end_index_test
import numpy as np
from sklearn.metrics import mean_squared_error

# Setze den Random Seed für numpy
np.random.seed(42)
def skill_score(actual_values, prediction, reference_values):
    forecast = np.sqrt(mean_squared_error(actual_values, prediction))
    reference = np.sqrt(mean_squared_error(actual_values, reference_values))
    perfect_forecast =1# np.sqrt(mean_squared_error(actual_values, actual_values))

    sc= (forecast- reference) / (perfect_forecast - reference)
    return sc


def lstm_uni(modell,real_valueser,start_index, end_index,forecast_horizon=24,window_size=24):
    from funcs.funcs_lstm_single import TemperatureModel
    import numpy as np
    import torch
    checkpoint_path = modell
    checkpoint = torch.load(checkpoint_path)
    #print(end_index-start_index)
    # Passe die Architektur deines LSTM-Modells entsprechend an
    model =TemperatureModel()# model = TemperatureModel(hidden_size=hidden_size, learning_rate=learning_rate, weight_decay=weight_decay,optimizer=optimizer,num_layers=num_layers,dropout=dropout)
  # Ersetze "YourLSTMModel" durch den tatsächlichen Namen deines Modells
    model.load_state_dict(checkpoint)  # ['state_dict'])
    model.eval()
    sliding_window = []  # Liste für das Sliding Window

    # Führe die Vorhersage für die ersten 24 Stunden durch
    predicted_values = []

    sliding_window= real_valueser[start_index:end_index].values
    #print(len(sliding_window))
    original_mean = np.mean(sliding_window)  # original_data sind die nicht normalisierten Daten
    original_std = np.std(sliding_window)
    sliding_window = (sliding_window - np.mean(sliding_window)) / np.std(sliding_window)
    # Berechnung des Durchschnitts und der Standardabweichung des Originaldatensatzes

    sliding_window=np.expand_dims(sliding_window, axis=0)
    sliding_window=np.expand_dims(sliding_window, axis=2)
    input_data = torch.from_numpy(sliding_window).float()
    with torch.no_grad():
        predicted_value = model(input_data)
    predicted_values.append(predicted_value.tolist())
    predictions=predicted_value.squeeze().tolist()
    denormalized_values = [(predicted_value * original_std )+ original_mean for predicted_value in predictions]
    return denormalized_values


def multilstm_full(modell,data,start_idx,end_idx,forecast_horizon=24,window_size=24):
    from funcs.funcs_lstm_multi import TemperatureModel_multi_full
    import numpy as np
    import torch
    from sklearn import preprocessing
    checkpoint_path = modell
    checkpoint = torch.load(checkpoint_path)
    print(end_idx-start_idx)

    # Passe die Architektur deines LSTM-Modells entsprechend an
    model = TemperatureModel_multi_full()  # Ersetze "YourLSTMModel" durch den tatsächlichen Namen deines Modells
    model.load_state_dict(checkpoint)  # ['state_dict'])
    model.eval()
    sliding_window = []  # Liste für das Sliding Window

    # Führe die Vorhersage für die ersten 24 Stunden durch
    predicted_values = []
    window_data = data.isel(index=slice(start_idx, end_idx)).to_array().values
    window_data_normalized = np.zeros((window_data.shape[0] + 7, window_size))  # np.zeros_like(window_data)
    means=[]
    stds=[]
    for i in range(window_data.shape[0]):
        if i != 9:
            variable = window_data[i, :]
            mean = np.mean(variable)
            std = np.std(variable)
            if std != 0:
                variable_normalized = (variable - mean) / std
            else:
                variable_normalized = np.zeros_like(variable)
            window_data_normalized[i, :] = variable_normalized
        else:
            variable = window_data[i, :]
            normalized_directions = variable % 360
            numeric_directions = (normalized_directions / 45).astype(int) % 8
            # print(numeric_directions)
            # windrichtungen = np.floor((variable % 360) / 45).astype(int)
            # Einteilung der Werte in Richtungen
            windrichtungen = ((variable + 22.5) // 45 % 8).astype(int)

            # One-Hot-Encoding
            encoder = preprocessing.OneHotEncoder(categories=[np.arange(8)], sparse_output=False)
            enc = np.transpose(encoder.fit_transform(windrichtungen.reshape(-1, 1)))
            # print(np.transpose(enc))
            # print(enc)
            for j in range(0, 7):
                window_data_normalized[i + j, :] = enc[j]  #
    sliding_window= window_data_normalized
    stds.append(np.std(window_data[0]))
    means.append(np.mean(window_data[0]))
    #original_mean = np.mean(sliding_window)  # original_data sind die nicht normalisierten Daten
    #original_std = np.std(sliding_window)
    #sliding_window = (sliding_window - np.mean(sliding_window)) / np.std(sliding_window)
    # Berechnung des Durchschnitts und der Standardabweichung des Originaldatensatzes

    #sliding_window=np.expand_dims(window_data_normalized, axis=0)
    sliding_window = sliding_window.transpose(1, 0)
    #print(sliding_window)
    sliding_window=np.expand_dims(sliding_window, axis=0)

    #sliding_window=np.expand_dims(sliding_window, axis=2)
    #input_data = torch.Tensor(sliding_window)
    input_data = torch.from_numpy(sliding_window).float()
    #print(input_data.shape)
    with torch.no_grad():
        predicted_value = model(input_data)
    predicted_values.append(predicted_value.tolist())
    predictions=predicted_value.squeeze().tolist()
    #print(predicted_value.shape)
    denormalized_values = [(predicted_value * stds[0] )+ means[0] for predicted_value in predictions]
    #denormalized_values=np.arange(0,len(denormalized_values))
    #print(denormalized_values)
    return denormalized_values

def tft(modell,data,start_idx,end_idx,forecast_horizon=24,window_size=24):
    from Model.funcs_tft import TFT_Modell
    import numpy as np
    import torch
    from sklearn import preprocessing
    checkpoint_path = modell
    checkpoint = torch.load(checkpoint_path)
    print(end_idx-start_idx)
    input_dim = 17  # Anzahl der meteorologischen Parameter
    output_dim = 1  # Vorhersage der Temperatur

    num_layers = 2
    num_heads = 17
    batch_size = 24
    hidden_dim = 3 * num_heads
    num_encoder_layers = 3
    num_decoder_layers = 3
    d_model = 17
    dropout = 0.1

    # Passe die Architektur deines LSTM-Modells entsprechend an
    model = TFT_Modell(input_dim, output_dim, hidden_dim, num_layers, num_heads, forecast_horizont=forecast_horizon,
                       window_size=window_size)
    model.load_state_dict(checkpoint)  # ['state_dict'])
    model.eval()
    sliding_window = []  # Liste für das Sliding Window

    # Führe die Vorhersage für die ersten 24 Stunden durch
    predicted_values = []
    window_data = data.isel(index=slice(start_idx, end_idx)).to_array().values
    window_data_normalized = np.zeros((window_data.shape[0] + 7, window_size))  # np.zeros_like(window_data)
    means=[]
    stds=[]
    for i in range(window_data.shape[0]):
        if i != 9:
            variable = window_data[i, :]
            mean = np.mean(variable)
            std = np.std(variable)
            if std != 0:
                variable_normalized = (variable - mean) / std
            else:
                variable_normalized = np.zeros_like(variable)
            window_data_normalized[i, :] = variable_normalized
        else:
            variable = window_data[i, :]
            normalized_directions = variable % 360
            numeric_directions = (normalized_directions / 45).astype(int) % 8
            # print(numeric_directions)
            # windrichtungen = np.floor((variable % 360) / 45).astype(int)
            # Einteilung der Werte in Richtungen
            windrichtungen = ((variable + 22.5) // 45 % 8).astype(int)

            # One-Hot-Encoding
            encoder = preprocessing.OneHotEncoder(categories=[np.arange(8)], sparse_output=False)
            enc = np.transpose(encoder.fit_transform(windrichtungen.reshape(-1, 1)))
            # print(np.transpose(enc))
            # print(enc)
            for j in range(0, 7):
                window_data_normalized[i + j, :] = enc[j]  #
    sliding_window= window_data_normalized
    stds.append(np.std(window_data[0]))
    means.append(np.mean(window_data[0]))
    #original_mean = np.mean(sliding_window)  # original_data sind die nicht normalisierten Daten
    #original_std = np.std(sliding_window)
    #sliding_window = (sliding_window - np.mean(sliding_window)) / np.std(sliding_window)
    # Berechnung des Durchschnitts und der Standardabweichung des Originaldatensatzes

    #sliding_window=np.expand_dims(window_data_normalized, axis=0)
    sliding_window = sliding_window.transpose(1, 0)
    #print(sliding_window)
    sliding_window=np.expand_dims(sliding_window, axis=0)

    #sliding_window=np.expand_dims(sliding_window, axis=2)
    #input_data = torch.Tensor(sliding_window)
    input_data = torch.from_numpy(sliding_window).float()
    #print(input_data.shape)
    with torch.no_grad():
        predicted_value = model(input_data)
    predicted_values.append(predicted_value.tolist())
    predictions=predicted_value.squeeze().tolist()
    #print(predicted_value.shape)
    denormalized_values = [(predicted_value * stds[0] )+ means[0] for predicted_value in predictions]
    #denormalized_values=np.arange(0,len(denormalized_values))
    #print(denormalized_values)
    return denormalized_values

def multilstm_light(modell,data,start_idx,end_idx):
    from funcs import TemperatureModel_multi_light
    import numpy as np
    import torch
    checkpoint_path = modell
    checkpoint = torch.load(checkpoint_path)
    #print(checkpoint.keys())

    # Passe die Architektur deines LSTM-Modells entsprechend an
    model = TemperatureModel_multi_light()  # Ersetze "YourLSTMModel" durch den tatsächlichen Namen deines Modells
    model.load_state_dict(checkpoint)  # ['state_dict'])
    model.eval()
    sliding_window = []  # Liste für das Sliding Window

    # Führe die Vorhersage für die ersten 24 Stunden durch
    predicted_values = []
    window_data = data.isel(index=slice(start_idx, end_idx)).to_array().values
    window_data_normalized = np.zeros_like(window_data)
    means=[]
    stds=[]
    for i in range(window_data.shape[0]):
        variable = window_data[i, :]
        mean = np.mean(variable)
        means.append(mean)
        std = np.std(variable)
        stds.append(std)
        if std != 0:
            variable_normalized = (variable - mean) / std
        else:
            variable_normalized = np.zeros_like(variable)
        window_data_normalized[i, :] = variable_normalized
    sliding_window= window_data_normalized
    #original_mean = np.mean(sliding_window)  # original_data sind die nicht normalisierten Daten
    #original_std = np.std(sliding_window)
    #sliding_window = (sliding_window - np.mean(sliding_window)) / np.std(sliding_window)
    # Berechnung des Durchschnitts und der Standardabweichung des Originaldatensatzes

    #sliding_window=np.expand_dims(window_data_normalized, axis=0)
    sliding_window = sliding_window.transpose(1, 0)
    #print(sliding_window)
    sliding_window=np.expand_dims(sliding_window, axis=0)

    #sliding_window=np.expand_dims(sliding_window, axis=2)
    #input_data = torch.Tensor(sliding_window)
    input_data = torch.from_numpy(sliding_window).float()
    with torch.no_grad():
        predicted_value = model(input_data)
    predicted_values.append(predicted_value.tolist())
    predictions=predicted_value.squeeze().tolist()
    #print(predictions)
    denormalized_values = [(predicted_value * stds[0] )+ means[0] for predicted_value in predictions]
    #denormalized_values=np.arange(0,len(denormalized_values))
    #print(denormalized_values)
    return denormalized_values


def conv(modell,data,start_idx,end_idx):
    from funcs import TemperatureModel_conv
    import numpy as np
    import torch
    from sklearn import preprocessing
    checkpoint_path = modell
    checkpoint = torch.load(checkpoint_path)
    #print(checkpoint.keys())

    # Passe die Architektur deines LSTM-Modells entsprechend an
    model = TemperatureModel_conv()  # Ersetze "YourLSTMModel" durch den tatsächlichen Namen deines Modells
    model.load_state_dict(checkpoint)  # ['state_dict'])
    model.eval()
    sliding_window = []  # Liste für das Sliding Window
    window_size = 6
    num_channels = 17
    # Führe die Vorhersage für die ersten 24 Stunden durch
    inputs=[]
    predicted_values = []
    for i in range(0,24):
        start_id =start_idx+i
        end_id= start_id+6
        window_data = data.isel(index=slice(start_id, end_id)).to_array().values
        window_data_normalized = np.zeros((window_size, num_channels))
        window_data_normalized[:, 0] = (window_data[0] - np.mean(window_data[0])) / np.std(
            window_data[0])  # air temperature
        window_data_normalized[:, 2] = (window_data[1] - np.mean(window_data[1])) / np.std(
            window_data[1])  # air pressure
        window_data_normalized[:, 1] = (window_data[2] - np.mean(window_data[2])) / np.std(
            window_data[2])  # humid
        window_data_normalized[:, 3] = (window_data[3] - np.mean(window_data[3])) / np.std(
            window_data[3])  # Globalstrahlung
        if np.std(window_data[4]) != 0:
            window_data_normalized[:, 4] = (window_data[4] - np.mean(window_data[4])) / np.std(
                window_data[4])  # gust_10
        else:
            window_data_normalized[:, 4] = np.zeros_like(window_data_normalized[:, 4])

        window_data_normalized[:, 5] = (window_data[5] - np.mean(window_data[5])) / np.std(
            window_data[5])  # gust_50
        if np.std(window_data[6]) != 0:
            window_data_normalized[:, -1] = (window_data[6] - np.mean(window_data[6])) / np.std(
                window_data[6])  # hourly precipitation
        else:
            window_data_normalized[:, -1] = np.zeros_like(window_data_normalized[:, 6])
        if np.std(window_data[7]) != 0:
            window_data_normalized[:, 6] = (window_data[7] - np.mean(window_data[7])) / np.std(
                window_data[7])  # wind_10
        else:
            window_data_normalized[:, 6] = np.zeros_like(window_data_normalized[:, 7])
        window_data_normalized[:, 7] = (window_data[8] - np.mean(window_data[8])) / np.std(
            window_data[8])  # wind_50
        # print(window_data_normalized.shape)
        # One-hot encode wind direction
        wind_direction = window_data[9]
        normalized_directions = wind_direction % 360
        numeric_directions = (normalized_directions / 45).astype(int) % 8
        encoder = preprocessing.OneHotEncoder(categories=[np.arange(8)], sparse_output=False)
        enc = encoder.fit_transform(numeric_directions.reshape(-1, 1))

        window_data_normalized[:, 8:-1] = enc
        #print(window_data_normalized)
        inputs.append(window_data_normalized)
    #print(np.transpose(inputs)[0][0])
    window_data = data.isel(index=slice(start_idx, end_idx)).to_array().values
    stbw=np.std(window_data[0])#np.transpose(inputs)[0][0])
    mean=np.mean(window_data[0])#np.transpose(inputs)[0][0])
    #print(stbw)
    #print(mean)
    #window_data = torch.from_numpy(window_data_normalized).float()

    #sliding_window=np.expand_dims(sliding_window, axis=2)
    #input_data = torch.Tensor(sliding_window)
    tensor=torch.empty((24,6,17))
    for i in range(0,24):
        tensor[i]=torch.tensor(inputs[i])
    input_data = tensor#torch.from_numpy(tensor).float()
    with torch.no_grad():
        predicted_value = model(input_data)
    predicted_values.append(predicted_value.tolist())
    predictions=predicted_value.squeeze().tolist()[-1]
    #print(predictions)
    denormalized_values = [(predicted_value *stbw )+ mean for predicted_value in predictions]
    #denormalized_values=np.arange(0,len(denormalized_values))
    #print(denormalized_values)
    return denormalized_values