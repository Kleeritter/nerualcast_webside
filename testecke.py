import numpy as np

data = [[373.0311], [373.0311], [373.52124], [373.5911], [374.14996], [372.641]]

# Umwandeln der Daten in ein numpy-Array
data_array = np.squeeze(data)

print(data_array[0])


import numpy as np
from sklearn.preprocessing import normalize

data = [[373.0311], [373.0311], [373.52124], [373.5911], [374.14996], [372.641]]

# Umwandeln der Daten in ein 2D-NumPy-Array
data_array = np.array(data)

# Normalisieren der Daten
normalized_data = normalize(data_array)

print(normalized_data)
