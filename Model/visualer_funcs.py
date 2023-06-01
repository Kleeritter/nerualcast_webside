def start_index_real(nc_path,gesuchtes_datum):
    import xarray as xr
    data = xr.open_dataset(nc_path)
    dataf = data.to_dataframe()
    start_index_real = dataf.index.get_loc(gesuchtes_datum)-1
    return start_index_real
def end_index_real(nc_path,gesuchtes_datum):
    import xarray as xr
    data = xr.open_dataset(nc_path)
    dataf = data.to_dataframe()
    end_index_real = dataf.index.get_loc(gesuchtes_datum) +24
    return end_index_real
def start_index_test(nc_path,gesuchtes_datum):
    import xarray as xr
    data = xr.open_dataset(nc_path)
    dataf = data.to_dataframe()
    start_index_test= dataf.index.get_loc(gesuchtes_datum) -24
    return start_index_test
def end_index_test(nc_path,gesuchtes_datum):
    import xarray as xr
    data = xr.open_dataset(nc_path)
    dataf = data.to_dataframe()
    end_index_test= dataf.index.get_loc(gesuchtes_datum)
    return end_index_test


def lstm_uni(modell,real_valueser,start_index, end_index):
    from funcs import TemperatureModel
    import numpy as np
    import torch
    checkpoint_path = modell
    checkpoint = torch.load(checkpoint_path)
    #print(checkpoint.keys())

    # Passe die Architektur deines LSTM-Modells entsprechend an
    model = TemperatureModel()  # Ersetze "YourLSTMModel" durch den tatsächlichen Namen deines Modells
    model.load_state_dict(checkpoint)  # ['state_dict'])
    model.eval()
    sliding_window = []  # Liste für das Sliding Window

    # Führe die Vorhersage für die ersten 24 Stunden durch
    predicted_values = []

    sliding_window= real_valueser[start_index:end_index].values
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
    #print(predictions)
    flattened_predicted_values = [value for sublist in predicted_values[0] for value in sublist]
    denormalized_values = [(predicted_value * original_std )+ original_mean for predicted_value in predicted_values[0][0]]
    denormalized_values = [(predicted_value * original_std )+ original_mean for predicted_value in predictions]

    #print(denormalized_values)
    return denormalized_values


def multilstm_full(modell,data,start_idx,end_idx):
    from funcs import TemperatureModel_multi_full, TemperatureModel_multi_light
    import numpy as np
    import torch
    checkpoint_path = modell
    checkpoint = torch.load(checkpoint_path)
    #print(checkpoint.keys())

    # Passe die Architektur deines LSTM-Modells entsprechend an
    model = TemperatureModel_multi_full()  # Ersetze "YourLSTMModel" durch den tatsächlichen Namen deines Modells
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

def multilstm_light(modell,data,start_idx,end_idx):
    from funcs import TemperatureModel_multi_full, TemperatureModel_multi_light
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