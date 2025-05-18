import pandas as pd
import numpy as np
import tensorflow as tf
import os
import glob
import matplotlib.pyplot as plt

def load_data(directory_path, file_type):
    all_data = []
    if file_type == 'stat':
        print(f"Wczytywanie danych statycznych z: {directory_path}...")
        file_pattern = os.path.join(directory_path, '*_stat_*.xlsx')
        files = glob.glob(file_pattern)
    elif file_type == 'dynamic':
        print(f"Wczytywanie danych dynamicznych z: {directory_path}...")
        all_xlsx_files = glob.glob(os.path.join(directory_path, '*.xlsx'))
        files = [f for f in all_xlsx_files if '_stat_' not in os.path.basename(f) and '_random_' not in os.path.basename(f)]
    else:
        print(f"Bledny typ plikow: {file_type}")
        return None

    if not files:
        print(f"Nieznaleziono plików: '{file_type}' w {directory_path}")
        return None

    for file in files:
        try:
            df = pd.read_excel(file)
            all_data.append(df)
        except Exception as e:
            print(f"Blad przy odczycie pliku: {file}: {e}")

    return pd.concat(all_data, ignore_index=True) if all_data else None

static_data_f8 = load_data('F8', 'stat')
static_data_f10 = load_data('F10', 'stat')

if static_data_f8 is not None and static_data_f10 is not None:
    training_data = pd.concat([static_data_f8, static_data_f10], ignore_index=True)
elif static_data_f8 is not None:
    training_data = static_data_f8
elif static_data_f10 is not None:
    training_data = static_data_f10
else:
    training_data = None
    print("Blad: Brak statycznych danych treningowych")

dynamic_data_f8 = load_data('F8', 'dynamic')
dynamic_data_f10 = load_data('F10', 'dynamic')

if dynamic_data_f8 is not None and dynamic_data_f10 is not None:
     verification_data = pd.concat([dynamic_data_f8, dynamic_data_f10], ignore_index=True)
elif dynamic_data_f8 is not None:
    verification_data = dynamic_data_f8
elif dynamic_data_f10 is not None:
    verification_data = dynamic_data_f10
else:
    verification_data = None
    print("Nie znaleziono plikow do weryfikacji.")

X_train = training_data[['data__coordinates__x', 'data__coordinates__y']].values
y_train = training_data[['reference__x', 'reference__y']].values

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,)), # Warstwa wejsciowa (x,y)
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(2) #Warstwa danych wyjsciowych (x, y)
])

model.compile(optimizer='adam', loss='mse')
print("Rozpoczynanie treningu sieci neuronowej...")
history = model.fit(X_train, y_train, epochs=100, validation_split=0.2, verbose=0)
print("Trening zakonczony.")

print("Rozpoczynanie weryfikacji...")
X_verify = verification_data[['data__coordinates__x', 'data__coordinates__y']].values
y_verify = verification_data[['reference__x', 'reference__y']].values

raw_errors = np.sqrt(np.sum((X_verify - y_verify)**2, axis=1))
corrected_predictions = model.predict(X_verify)
corrected_errors = np.sqrt(np.sum((corrected_predictions - y_verify)**2, axis=1))
print("Weryfikacja zakończona.")

def plot_cdf(errors, label):
    sorted_errors = np.sort(errors)
    cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    plt.plot(sorted_errors, cdf, label=label)

plt.figure(figsize=(10, 6))
plot_cdf(raw_errors, 'CDF Błędu Surowych Danych')
plot_cdf(corrected_errors, 'CDF Błędu Skorygowanych Danych przez Sieć Neuronową')
plt.xlabel('Błąd Lokalizacji')
plt.ylabel('CDF')
plt.title('Porównanie CDF Błędów Lokalizacji')
plt.legend()
plt.grid(True)
plt.show()