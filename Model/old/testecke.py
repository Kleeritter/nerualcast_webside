import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
print(pd.date_range(start='2023-01-01', periods=5, freq='D'))
# Beispiel-Daten erstellen
data = pd.DataFrame({
    'Datum': pd.date_range(start='2023-01-01', periods=5, freq='D'),
    'Serie1a': [0, 1, 2, 3, 4] ,
    'Serie1b': [4, 3, 2, 1, 0] ,
    'Serie2a': [1, 2, 3, 2, 1] ,
    'Serie2b': [3, 1, 2, 4, 3]
})

# Subplots erstellen
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# Daten für jeden Subplot auswählen und zeichnen
sns.lineplot(x='Datum', y='value', hue='variable', data=pd.melt(data, ['Datum', 'Serie1a']), ax=axs[0, 0])
sns.lineplot(x='Datum', y='value', hue='variable', data=pd.melt(data, ['Datum', 'Serie1b']), ax=axs[0, 1])
sns.lineplot(x='Datum', y='value', hue='variable', data=pd.melt(data, ['Datum', 'Serie2a']), ax=axs[1, 0])
sns.lineplot(x='Datum', y='value', hue='variable', data=pd.melt(data, ['Datum', 'Serie2b']), ax=axs[1, 1])

# Diagramm anzeigen
plt.tight_layout()
plt.show()