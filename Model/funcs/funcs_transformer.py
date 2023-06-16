
import pytorch_lightning as pl

from pytorch_forecasting import TimeSeriesDataSet
from sklearn import preprocessing
import torch
import xarray as xr
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

def test(file_path):
    data = xr.open_dataset(file_path)[
        ['temp', "press_sl", "humid", "Geneigt CM-11", "gust_10", "gust_50", "rain", "wind_10", "wind_50",
         "wind_dir_50"]].to_dataframe()  # .valuesmissing_values_mask = dataset['temp'].isnull()
    print(data)
    dataset = TimeSeriesDataSet(
        data,
        #group_ids=["group"],
        target="temp",
        time_idx="index",
        max_encoder_length=24*4*7,
        max_prediction_length=24,
        #time_varying_unknown_reals=["target"],
        #static_categoricals=["holidays"],
        target_normalizer=None
    )
    # pass the dataset to a dataloader
    dataloader = dataset.to_dataloader(batch_size=1)



def tft():
    tft = TemporalFusionTransformer.from_dataset(
        data,
        hidden_size=16,
        attention_head_size=2,
        dropout=0.1,
        hidden_continuous_size=8,
        output_size=7,  # Anzahl der Vorhersageparameter
        loss=nn.MSELoss(),
    )
    return tft