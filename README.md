# Poprawa lokalizacji UWB przy pomocy sieci neuronowych

## Autorzy
| Imię i Nazwisko (inicjały) | Nick                                    |
| -------------------------- | --------------------------------------- |
| Klim Hudzenko (KH)        | klimh                                   |
| Remigiusz Tomecki (RT)    | Remko23                                 |

## Opis projektu
Projekt ma na celu poprawę dokładności lokalizacji UWB (Ultra-Wideband) przy użyciu zaawansowanej sieci neuronowej. System wykorzystuje dane statyczne do treningu oraz dane dynamiczne do weryfikacji skuteczności rozwiązania.

## Architektura sieci neuronowej

### Struktura warstw
1. **Warstwa wejściowa**:
   - Kształt: (5, 2) - sekwencje 5 próbek z 2 wymiarami (x, y)

2. **Warstwy LSTM**:
   - LSTM(128) z return_sequences=True
   - Dropout(0.2)
   - LSTM(64)

3. **Warstwy gęste**:
   - Dense(128) z aktywacją ReLU
   - BatchNormalization
   - Dropout(0.3)
   - Dense(64) z aktywacją ReLU
   - BatchNormalization
   - Dropout(0.2)

4. **Warstwa wyjściowa**:
   - Dense(2) - przewidywane koordynaty (x, y)

### Funkcje aktywacji
- ReLU (Rectified Linear Unit) w warstwach ukrytych
- Liniowa w warstwie wyjściowej

### Mechanizm eliminacji błędnych pomiarów
1. **Preprocesssing danych**:
   - Wykrywanie wartości odstających (outliers) metodą z-score
   - Standaryzacja danych wejściowych i wyjściowych
   - Tworzenie sekwencji czasowych dla lepszego przetwarzania temporalnego

2. **Regularyzacja w sieci**:
   - Dropout dla zapobiegania przeuczeniu
   - Normalizacja wsadowa (Batch Normalization)
   - Wczesne zatrzymywanie (Early Stopping)

## Algorytm uczenia

### Optymalizator
- Adam z harmonogramem zmiennej wartości współczynnika uczenia
- Początkowy learning rate: 0.001
- Decay steps: 1000
- Decay rate: 0.9

### Funkcja straty
- Mean Squared Error (MSE)

### Metryki
- Mean Absolute Error (MAE)

### Parametry treningu
- Epochs: 200 (z Early Stopping)
- Batch size: 32
- Validation split: 0.2

## Wyniki
Wyniki porównania dystrybuant błędu znajdują się w pliku `cdf_values.xlsx` oraz są wizualizowane w pliku `cdf_comparison.png`.

## Struktura projektu
```
.
├── main.py              # Główny skrypt
├── README.md           # Dokumentacja
├── cdf_comparison.png  # Wykres porównawczy CDF
├── training_history.png # Historia treningu
├── cdf_values.xlsx     # Wartości dystrybuanty błędu
├── F8/                 # Katalog z danymi F8
└── F10/               # Katalog z danymi F10
```

## Wymagania
```
tensorflow
numpy
pandas
matplotlib
scikit-learn
```

## Jak uruchomić
1. Upewnij się, że masz zainstalowane wszystkie wymagane biblioteki
2. Umieść dane w odpowiednich katalogach (F8 i F10)
3. Uruchom skrypt:
```bash
python main.py
```