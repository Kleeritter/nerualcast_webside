from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Annahme: Normalisierung wurde mit einem Mindestwert von 0 und einem Maximalwert von 1 durchgeführt
min_value = 261
max_value = 310


# Werte aus der anderen Datei, die normalisiert werden sollen

train_values=[261,310]
values_to_normalize = [270, 300, 305,400]

# MinMaxScaler-Objekt erstellen und mit den bekannten Maximal- und Minimalwerten anpassen

...
scaler = MinMaxScaler()




# Normalisierung durchführen
X_train_minmax = scaler.fit_transform(np.array(train_values).reshape(-1, 1))

# Normalisierte Werte ausgeben
print(X_train_minmax.flatten())

# MinMaxScaler-Objekt erstellen und mit den ursprünglichen Daten anpassen
xtest=scaler.transform(np.array(values_to_normalize).reshape(-1,1)).flatten()

print(xtest)
# Denormalisierung durchführen
denormalized_values = scaler.inverse_transform([[x] for x in xtest])

# Denormalisierte Werte ausgeben
print(denormalized_values.flatten())
