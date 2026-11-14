# Architektury modeli

Opis struktur modeli używanych w `src/main.ipynb`, ich liczby parametrów i miejsc zapisu wag. Każdy model ma dwie wersje w notebooku: `Model_CNN` i `ModelMLP`.

## Wspólne ustawienia treningu

- **Dane:** Fashion-MNIST, train 55 000, dev 5 000, test 10 000. Dane ładowane przez `load_data(..., dtype='float64')`.
- **Wejście:** `(28, 28, 1)`, normalizacja skalarną średnią i odchyleniem liczonymi na zbiorze train.
- **Optymalizator:** SGD (`SGD`), bez momentum i bez decay.
- **Strata:** softmax + cross-entropy (`SoftmaxCrossEntropy`).
- **Batch size:** 64.
- **Learning rate:** 3e-3.
- **Liczba epok:** 20.
- **Generator losowy:** `rng = np.random.default_rng(seed=123)`, przekazywany do obu modeli.

## CNN (`Model_CNN`)

| Warstwa | Kształt wyjścia | Parametry |
|---|---|---|
| Wejście | (28, 28, 1) | 0 |
| `Conv2d(8, f=3, s=1, p='same')` | (28, 28, 8) | 3·3·1·8 + 8 = 80 |
| `ReLU` | (28, 28, 8) | 0 |
| `MaxPool2d()` (f=2, s=2) | (14, 14, 8) | 0 |
| `Conv2d(16, f=3, s=1, p='same')` | (14, 14, 16) | 3·3·8·16 + 16 = 1 168 |
| `ReLU` | (14, 14, 16) | 0 |
| `MaxPool2d()` (f=2, s=2) | (7, 7, 16) | 0 |
| `Flatten` | (784,) | 0 |
| `Dense(64)` | (64,) | 784·64 + 64 = 50 240 |
| `ReLU` | (64,) | 0 |
| `Dense(10)` | (10,) | 64·10 + 10 = 650 |

**Razem:** 52 138 parametrów.

**Wagi:** `models/my_CNN-1.npz`.

## MLP (`ModelMLP`)

| Warstwa | Kształt wyjścia | Parametry |
|---|---|---|
| Wejście | (28, 28, 1) | 0 |
| `Flatten` | (784,) | 0 |
| `Dense(64)` | (64,) | 784·64 + 64 = 50 240 |
| `ReLU` | (64,) | 0 |
| `Dense(10)` | (10,) | 64·10 + 10 = 650 |

**Razem:** 50 890 parametrów (zbliżona liczba do CNN, co ułatwia porównanie).

**Wagi:** `models/my_MLP-1.npz`.

## Uwagi

- Oba modele trenowane z tym samym `lr` i liczbą epok. Dla uczciwego porównania każdy model powinien mieć własne dobrane `lr`.
- Aby wczytać wagi, najpierw zbuduj model (`build()`) z tym samym `input_shape`, a dopiero potem wywołaj `load_weights`.
- Ustawienia w tabelach opisują notebook. Wagi w `models/` mogły powstać przy innym lr lub liczbie epok. Przed porównaniem sprawdź konfigurację, z którą je wytrenowano.
