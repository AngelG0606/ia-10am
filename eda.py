import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. Generación/Carga de Datos
np.random.seed(42)
tiempo = pd.date_range(start='2026-04-15', periods=500, freq='min')

valores = np.linspace(10, 50, 500) + np.sin(np.linspace(0, 20, 500))*5 + np.random.normal(0, 2, 500)

df = pd.DataFrame({'Timestamp': tiempo, 'Lectura': valores})
df.set_index('Timestamp', inplace=True)

# 2. Inspección Temporal
plt.figure(figsize=(10, 4))
plt.plot(df.index, df['Lectura'], label='Señal Cruda del Sensor')
plt.title("Inspección Visual de la Serie de Tiempo")
plt.xlabel("Tiempo")
plt.ylabel("Amplitud")
plt.grid(True)
plt.show()

# 3. Histograma
plt.figure(figsize=(6, 4))
sns.histplot(df['Lectura'], kde=True, bins=30)
plt.title("Histograma y Densidad de las Lecturas")
plt.show()

# 4. Estadística
print("=== Perfil Estadístico del Dataset ===")
print(df.describe())