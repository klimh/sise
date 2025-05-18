import pandas as pd
import numpy as np
import tensorflow as tf
import os
import glob
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def detect_outliers(data, reference, threshold_multiplier=5):
    """
    Detect outliers using distance-based method specific for UWB data
    Args:
        data: Input coordinates from UWB
        reference: Reference coordinates
        threshold_multiplier: Multiplier for IQR method (default=5)
    Returns:
        Boolean mask where True indicates non-outlier data
    """
    # Check for NaN values
    valid_mask = ~np.isnan(data).any(axis=1) & ~np.isnan(reference).any(axis=1)
    if not np.any(valid_mask):
        raise ValueError("Wszystkie dane zawierają wartości NaN!")

    print(f"\nSprawdzanie danych:")
    print(f"Znaleziono {np.sum(~valid_mask)} wierszy z wartościami NaN")
    print(f"Liczba prawidłowych wierszy: {np.sum(valid_mask)} z {len(valid_mask)}")

    # Use only valid data for calculations
    valid_data = data[valid_mask]
    valid_reference = reference[valid_mask]

    # Calculate Euclidean distance between measured and reference points
    distances = np.sqrt(np.sum((valid_data - valid_reference) ** 2, axis=1))

    # Use IQR method with more permissive threshold
    Q1 = np.percentile(distances, 25)
    Q3 = np.percentile(distances, 75)
    IQR = Q3 - Q1

    # Define bounds - using more permissive threshold
    lower_bound = Q1 - threshold_multiplier * IQR
    upper_bound = Q3 + threshold_multiplier * IQR

    # Create mask for valid data
    distance_mask = (distances >= lower_bound) & (distances <= upper_bound)

    # Combine masks
    final_mask = np.zeros(len(data), dtype=bool)
    final_mask[valid_mask] = distance_mask

    print(f"\nStatystyki filtracji outlierów:")
    print(f"Średnia odległość błędu: {np.mean(distances):.2f} m")
    print(f"Mediana odległości błędu: {np.median(distances):.2f} m")
    print(f"Dolny próg: {lower_bound:.2f} m")
    print(f"Górny próg: {upper_bound:.2f} m")
    print(f"Liczba zachowanych próbek: {np.sum(final_mask)} z {len(final_mask)}")
    print(f"Procent zachowanych próbek: {(np.sum(final_mask) / len(final_mask)) * 100:.1f}%")

    return final_mask


def load_data(directory_path, file_type):
    all_data = []
    if file_type == 'stat':
        print(f"Wczytywanie danych statycznych z: {directory_path}...")
        file_pattern = os.path.join(directory_path, '*_stat_*.xlsx')
        files = glob.glob(file_pattern)
    elif file_type == 'dynamic':
        print(f"Wczytywanie danych dynamicznych z: {directory_path}...")
        all_xlsx_files = glob.glob(os.path.join(directory_path, '*.xlsx'))
        files = [f for f in all_xlsx_files if
                 '_stat_' not in os.path.basename(f) and '_random_' not in os.path.basename(f)]
    else:
        print(f"Bledny typ plikow: {file_type}")
        return None

    if not files:
        print(f"Nieznaleziono plików: '{file_type}' w {directory_path}")
        return None

    for file in files:
        try:
            print(f"Wczytywanie pliku: {file}")
            df = pd.read_excel(file)
            # Sprawdź i usuń duplikaty kolumn
            df = df.loc[:, ~df.columns.duplicated()]
            print(f"Kształt danych z pliku {file}: {df.shape}")
            all_data.append(df)
        except Exception as e:
            print(f"Blad przy odczycie pliku: {file}: {e}")

    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)
        print(f"Całkowity kształt danych {file_type}: {combined_data.shape}")
        return combined_data
    return None


def create_sequences(data, sequence_length):
    """
    Create sequences from data for temporal processing
    """
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length)])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)


# Load and preprocess data
print("Wczytywanie danych...")
static_data_f8 = load_data('F8', 'stat')
static_data_f10 = load_data('F10', 'stat')

if static_data_f8 is not None and static_data_f10 is not None:
    training_data = pd.concat([static_data_f8, static_data_f10], ignore_index=True)
    print("Połączono dane F8 i F10")
elif static_data_f8 is not None:
    training_data = static_data_f8
    print("Używam tylko danych F8")
elif static_data_f10 is not None:
    training_data = static_data_f10
    print("Używam tylko danych F10")
else:
    training_data = None
    print("Blad: Brak statycznych danych treningowych")
    raise ValueError("Nie znaleziono żadnych danych treningowych!")

print(f"Kształt danych treningowych: {training_data.shape}")
print("Dostępne kolumny:", training_data.columns.tolist())

# Data preprocessing
print("\nPrzetwarzanie danych...")
try:
    # Sprawdź duplikaty kolumn w danych treningowych
    training_data = training_data.loc[:, ~training_data.columns.duplicated()]

    # Sprawdź wartości NaN w kolumnach
    nan_counts = training_data[['data__coordinates__x', 'data__coordinates__y',
                                'reference__x', 'reference__y']].isna().sum()
    print("\nLiczba wartości NaN w kolumnach:")
    print(nan_counts)

    X_train = training_data[['data__coordinates__x', 'data__coordinates__y']].values
    y_train = training_data[['reference__x', 'reference__y']].values

    print(f"\nKształt X_train przed filtracją outlierów: {X_train.shape}")
    print(f"Kształt y_train przed filtracją outlierów: {y_train.shape}")

    if X_train.shape[0] == 0:
        raise ValueError("Dane wejściowe są puste!")

    # Detect and remove outliers with new method
    outlier_mask = detect_outliers(X_train, y_train, threshold_multiplier=5)

    X_train = X_train[outlier_mask]
    y_train = y_train[outlier_mask]

    print(f"\nKształt X_train po filtracji outlierów: {X_train.shape}")
    print(f"Kształt y_train po filtracji outlierów: {y_train.shape}")

    if X_train.shape[0] == 0:
        raise ValueError("Po usunięciu outlierów nie pozostały żadne dane!")

except KeyError as e:
    print(f"Błąd: Nie znaleziono wymaganych kolumn w danych. {e}")
    print("Dostępne kolumny:", training_data.columns.tolist())
    raise
except Exception as e:
    print(f"Nieoczekiwany błąd podczas przetwarzania danych: {e}")
    raise

# Standardize the data
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train)

# Create sequences for temporal processing
sequence_length = 5
X_seq, y_seq = create_sequences(X_train_scaled, sequence_length)

# Split the data
X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_seq, y_seq, test_size=0.2, random_state=42
)

# Create enhanced neural network model
print("Tworzenie modelu sieci neuronowej...")
model = tf.keras.Sequential([
    # Input layer
    tf.keras.layers.Input(shape=(sequence_length, 2)),

    # LSTM layers for temporal processing
    tf.keras.layers.LSTM(128, return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(64),

    # Dense layers for feature extraction
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),

    # Output layer
    tf.keras.layers.Dense(2)
])

# Compile model with fixed learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

# Train the model with callbacks
print("Rozpoczynanie treningu sieci neuronowej...")
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        patience=10,
        restore_best_weights=True,
        monitor='val_loss'
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        patience=5,
        factor=0.5,
        min_lr=1e-6,
        monitor='val_loss'
    )
]

history = model.fit(
    X_train_final, y_train_final,
    epochs=50,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    verbose=1
)

print("Trening zakończony.")

# Load verification data
print("\nWczytywanie danych weryfikacyjnych...")
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
    raise ValueError("Nie znaleziono plików do weryfikacji.")

# Evaluate on verification data
print("Rozpoczynanie weryfikacji...")
X_verify = verification_data[['data__coordinates__x', 'data__coordinates__y']].values
y_verify = verification_data[['reference__x', 'reference__y']].values

# Preprocess verification data
X_verify_scaled = scaler_X.transform(X_verify)
X_verify_seq, y_verify_seq = create_sequences(X_verify_scaled, sequence_length)

# Calculate errors
raw_errors = np.sqrt(np.sum((X_verify - y_verify) ** 2, axis=1))
predictions_scaled = model.predict(X_verify_seq)
predictions = scaler_y.inverse_transform(predictions_scaled)
corrected_errors = np.sqrt(np.sum((predictions - y_verify[sequence_length::]) ** 2, axis=1))

print("Weryfikacja zakończona.")


# Plot results
def plot_cdf(errors, label):
    sorted_errors = np.sort(errors)
    cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    plt.plot(sorted_errors, cdf, label=label)
    return sorted_errors, cdf


plt.figure(figsize=(10, 6))
raw_sorted, raw_cdf = plot_cdf(raw_errors, 'CDF Błędu Surowych Danych')
corr_sorted, corr_cdf = plot_cdf(corrected_errors, 'CDF Błędu Skorygowanych Danych')
plt.xlabel('Błąd Lokalizacji [m]')
plt.ylabel('CDF')
plt.title('Porównanie CDF Błędów Lokalizacji')
plt.legend()
plt.grid(True)
plt.savefig('cdf_comparison.png')
plt.show()

# Save CDF values to Excel
cdf_df = pd.DataFrame({
    'Błąd [m]': corr_sorted,
    'CDF': corr_cdf
})
cdf_df.to_excel('cdf_values.xlsx', index=False)

# Print model summary for report
model.summary()

# Save training history
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.title('Training History')
plt.legend()
plt.grid(True)
plt.savefig('training_history.png')
plt.show()

# Calculate and print improvement statistics
mean_raw_error = np.mean(raw_errors)
mean_corrected_error = np.mean(corrected_errors)
improvement_percentage = ((mean_raw_error - mean_corrected_error) / mean_raw_error) * 100

print(f"\nStatystyki poprawy:")
print(f"Średni błąd surowych danych: {mean_raw_error:.2f} m")
print(f"Średni błąd po korekcji: {mean_corrected_error:.2f} m")
print(f"Procentowa poprawa: {improvement_percentage:.2f}%")