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

## Siedem modeli z tuningu (`trained_models/`)

Modele z sekcji „Siedem modeli” w `src/main.ipynb`. Wszystkie: wejście (28, 28, 1), Adam, stały lr, batch 64, 8 epok na pełnym train, float32, softmax + cross-entropy. Szablony warstw (`ARCHITECTURES`) są w notebooku, a hiperparametry konkretnego modelu w `trained_models/<model>.json`. Przyjęta konwencja: Conv/Dense bez biasu → BatchNorm → ReLU, Dropout po ReLU albo poolingu, ReLU po sumie w resblocku (poza blokiem). Tabele pochodzą z `Sequential.summary()`.

**Przywracanie modelu:** uruchom komórkę z `ARCHITECTURES` w notebooku, potem:

```python
model = tuning.load_trained(ARCHITECTURES['conv_deep'], 'conv_deep')   # buduje model z JSON-a i wczytuje wagi
```

Wartości w kodzie poniżej (np. `Dropout(0.911...)`) to dokładnie to, z czym model trenowano. Kod można wkleić zamiast szablonu.

### `softmax_regression`: Regresja softmax (jedna warstwa Dense)

| Warstwa | Wyjście | Parametry |
|---|---|---|
| Wejście | (28, 28, 1) | 0 |
| Flatten 1 | (784,) | 0 |
| Dense 1 | (10,) | 7 850 |

**Razem:** 7 850 parametrów.

**Wagi:** `trained_models/softmax_regression.npz`. **Konfiguracja i historia:** `trained_models/softmax_regression.json`. **Wyniki:** dev 0,8396, czas 0,0 min (patrz `experiments.md`).

Kod warstw:

```python
layers = [Flatten(),
 Dense(10, bias=True, init=xavier)]
```

### `mlp`: MLP, jedna warstwa ukryta (256)

| Warstwa | Wyjście | Parametry |
|---|---|---|
| Wejście | (28, 28, 1) | 0 |
| Flatten 1 | (784,) | 0 |
| Dense 1 | (256,) | 200 704 |
| BatchNorm2d 1 | (256,) | 512 |
| ReLU 1 | (256,) | 0 |
| Dropout 1 | (256,) | 0 |
| Dense 2 | (10,) | 2 570 |

**Razem:** 203 786 parametrów.

**Wagi:** `trained_models/mlp.npz`. **Konfiguracja i historia:** `trained_models/mlp.json`. **Wyniki:** dev 0,8926, czas 0,1 min (patrz `experiments.md`).

Kod warstw:

```python
layers = [Flatten(),
 Dense(256, bias=False, init=he),
 BatchNorm2d(axes=(0,), momentum=0.1),
 ReLU(),
 Dropout(0.7073279246778355),
 Dense(10, bias=True, init=xavier)]
```

### `mlp_deep`: MLP głęboki, 4 warstwy ukryte (512-256-128-64)

| Warstwa | Wyjście | Parametry |
|---|---|---|
| Wejście | (28, 28, 1) | 0 |
| Flatten 1 | (784,) | 0 |
| Dense 1 | (512,) | 401 408 |
| BatchNorm2d 1 | (512,) | 1 024 |
| ReLU 1 | (512,) | 0 |
| Dropout 1 | (512,) | 0 |
| Dense 2 | (256,) | 131 072 |
| BatchNorm2d 2 | (256,) | 512 |
| ReLU 2 | (256,) | 0 |
| Dropout 2 | (256,) | 0 |
| Dense 3 | (128,) | 32 768 |
| BatchNorm2d 3 | (128,) | 256 |
| ReLU 3 | (128,) | 0 |
| Dropout 3 | (128,) | 0 |
| Dense 4 | (64,) | 8 192 |
| BatchNorm2d 4 | (64,) | 128 |
| ReLU 4 | (64,) | 0 |
| Dropout 4 | (64,) | 0 |
| Dense 5 | (10,) | 650 |

**Razem:** 576 010 parametrów.

**Wagi:** `trained_models/mlp_deep.npz`. **Konfiguracja i historia:** `trained_models/mlp_deep.json`. **Wyniki:** dev 0,9020, czas 0,3 min (patrz `experiments.md`).

Kod warstw:

```python
layers = [Flatten(),
 Dense(512, bias=False, init=he),
 BatchNorm2d(axes=(0,), momentum=0.1),
 ReLU(),
 Dropout(0.9163220738266988),
 Dense(256, bias=False, init=he),
 BatchNorm2d(axes=(0,), momentum=0.1),
 ReLU(),
 Dropout(0.9163220738266988),
 Dense(128, bias=False, init=he),
 BatchNorm2d(axes=(0,), momentum=0.1),
 ReLU(),
 Dropout(0.9163220738266988),
 Dense(64, bias=False, init=he),
 BatchNorm2d(axes=(0,), momentum=0.1),
 ReLU(),
 Dropout(0.9163220738266988),
 Dense(10, bias=True, init=xavier)]
```

### `mlp_deep_res`: MLP głęboki z blokami residualnymi (3 bloki po 256)

| Warstwa | Wyjście | Parametry |
|---|---|---|
| Wejście | (28, 28, 1) | 0 |
| Flatten 1 | (784,) | 0 |
| Dense 1 | (256,) | 200 704 |
| BatchNorm2d 1 | (256,) | 512 |
| ReLU 1 | (256,) | 0 |
| ResidualBlock 1 | (256,) | 132 096 |
| ReLU 2 | (256,) | 0 |
| Dropout 1 | (256,) | 0 |
| ResidualBlock 2 | (256,) | 132 096 |
| ReLU 3 | (256,) | 0 |
| Dropout 2 | (256,) | 0 |
| ResidualBlock 3 | (256,) | 132 096 |
| ReLU 4 | (256,) | 0 |
| Dropout 3 | (256,) | 0 |
| Dense 2 | (10,) | 2 570 |

**Razem:** 600 074 parametrów.

**Wagi:** `trained_models/mlp_deep_res.npz`. **Konfiguracja i historia:** `trained_models/mlp_deep_res.json`. **Wyniki:** dev 0,9004, czas 0,4 min (patrz `experiments.md`).

Kod warstw:

```python
layers = [Flatten(),
 Dense(256, bias=False, init=he),
 BatchNorm2d(axes=(0,), momentum=0.1),
 ReLU(),
 ResidualBlock([
     Dense(256, bias=False, init=he),
     BatchNorm2d(axes=(0,), momentum=0.1),
     ReLU(),
     Dense(256, bias=False, init=he),
     BatchNorm2d(axes=(0,), momentum=0.1)]),
 ReLU(),
 Dropout(0.9066351196001362),
 ResidualBlock([
     Dense(256, bias=False, init=he),
     BatchNorm2d(axes=(0,), momentum=0.1),
     ReLU(),
     Dense(256, bias=False, init=he),
     BatchNorm2d(axes=(0,), momentum=0.1)]),
 ReLU(),
 Dropout(0.9066351196001362),
 ResidualBlock([
     Dense(256, bias=False, init=he),
     BatchNorm2d(axes=(0,), momentum=0.1),
     ReLU(),
     Dense(256, bias=False, init=he),
     BatchNorm2d(axes=(0,), momentum=0.1)]),
 ReLU(),
 Dropout(0.9066351196001362),
 Dense(10, bias=True, init=xavier)]
```

### `conv`: CNN, 2 bloki Conv-BN-ReLU-MaxPool (32, 64)

| Warstwa | Wyjście | Parametry |
|---|---|---|
| Wejście | (28, 28, 1) | 0 |
| Conv2d 1 | (28, 28, 32) | 288 |
| BatchNorm2d 1 | (28, 28, 32) | 64 |
| ReLU 1 | (28, 28, 32) | 0 |
| MaxPool2d 1 | (14, 14, 32) | 0 |
| Dropout 1 | (14, 14, 32) | 0 |
| Conv2d 2 | (14, 14, 64) | 18 432 |
| BatchNorm2d 2 | (14, 14, 64) | 128 |
| ReLU 2 | (14, 14, 64) | 0 |
| MaxPool2d 2 | (7, 7, 64) | 0 |
| Dropout 2 | (7, 7, 64) | 0 |
| Flatten 1 | (3136,) | 0 |
| Dense 1 | (10,) | 31 370 |

**Razem:** 50 282 parametrów.

**Wagi:** `trained_models/conv.npz`. **Konfiguracja i historia:** `trained_models/conv.json`. **Wyniki:** dev 0,9118, czas 4,7 min (patrz `experiments.md`).

Kod warstw:

```python
layers = [Conv2d(32, f=3, s=1, p='same', bias=False, init=he),
 BatchNorm2d(axes=(0, 1, 2), momentum=0.1),
 ReLU(),
 MaxPool2d(f=2, s=2),
 Dropout(0.9905975200331721),
 Conv2d(64, f=3, s=1, p='same', bias=False, init=he),
 BatchNorm2d(axes=(0, 1, 2), momentum=0.1),
 ReLU(),
 MaxPool2d(f=2, s=2),
 Dropout(0.9905975200331721),
 Flatten(),
 Dense(10, bias=True, init=xavier)]
```

### `conv_deep`: CNN głęboki, 3 etapy po 2 konwolucje (32, 64, 128)

| Warstwa | Wyjście | Parametry |
|---|---|---|
| Wejście | (28, 28, 1) | 0 |
| Conv2d 1 | (28, 28, 32) | 288 |
| BatchNorm2d 1 | (28, 28, 32) | 64 |
| ReLU 1 | (28, 28, 32) | 0 |
| Conv2d 2 | (28, 28, 32) | 9 216 |
| BatchNorm2d 2 | (28, 28, 32) | 64 |
| ReLU 2 | (28, 28, 32) | 0 |
| MaxPool2d 1 | (14, 14, 32) | 0 |
| Dropout 1 | (14, 14, 32) | 0 |
| Conv2d 3 | (14, 14, 64) | 18 432 |
| BatchNorm2d 3 | (14, 14, 64) | 128 |
| ReLU 3 | (14, 14, 64) | 0 |
| Conv2d 4 | (14, 14, 64) | 36 864 |
| BatchNorm2d 4 | (14, 14, 64) | 128 |
| ReLU 4 | (14, 14, 64) | 0 |
| MaxPool2d 2 | (7, 7, 64) | 0 |
| Dropout 2 | (7, 7, 64) | 0 |
| Conv2d 5 | (7, 7, 128) | 73 728 |
| BatchNorm2d 5 | (7, 7, 128) | 256 |
| ReLU 5 | (7, 7, 128) | 0 |
| Conv2d 6 | (7, 7, 128) | 147 456 |
| BatchNorm2d 6 | (7, 7, 128) | 256 |
| ReLU 6 | (7, 7, 128) | 0 |
| MaxPool2d 3 | (3, 3, 128) | 0 |
| Dropout 3 | (3, 3, 128) | 0 |
| Flatten 1 | (1152,) | 0 |
| Dense 1 | (10,) | 11 530 |

**Razem:** 298 410 parametrów.

**Wagi:** `trained_models/conv_deep.npz`. **Konfiguracja i historia:** `trained_models/conv_deep.json`. **Wyniki:** dev 0,9262, test 0,9271, czas 13,6 min (patrz `experiments.md`).

Kod warstw:

```python
layers = [Conv2d(32, f=3, s=1, p='same', bias=False, init=he),
 BatchNorm2d(axes=(0, 1, 2), momentum=0.1),
 ReLU(),
 Conv2d(32, f=3, s=1, p='same', bias=False, init=he),
 BatchNorm2d(axes=(0, 1, 2), momentum=0.1),
 ReLU(),
 MaxPool2d(f=2, s=2),
 Dropout(0.9111869137715352),
 Conv2d(64, f=3, s=1, p='same', bias=False, init=he),
 BatchNorm2d(axes=(0, 1, 2), momentum=0.1),
 ReLU(),
 Conv2d(64, f=3, s=1, p='same', bias=False, init=he),
 BatchNorm2d(axes=(0, 1, 2), momentum=0.1),
 ReLU(),
 MaxPool2d(f=2, s=2),
 Dropout(0.9111869137715352),
 Conv2d(128, f=3, s=1, p='same', bias=False, init=he),
 BatchNorm2d(axes=(0, 1, 2), momentum=0.1),
 ReLU(),
 Conv2d(128, f=3, s=1, p='same', bias=False, init=he),
 BatchNorm2d(axes=(0, 1, 2), momentum=0.1),
 ReLU(),
 MaxPool2d(f=2, s=2),
 Dropout(0.9111869137715352),
 Flatten(),
 Dense(10, bias=True, init=xavier)]
```

### `conv_deep_res`: CNN głęboki z blokami residualnymi (32, 64, 128)

| Warstwa | Wyjście | Parametry |
|---|---|---|
| Wejście | (28, 28, 1) | 0 |
| Conv2d 1 | (28, 28, 32) | 288 |
| BatchNorm2d 1 | (28, 28, 32) | 64 |
| ReLU 1 | (28, 28, 32) | 0 |
| ResidualBlock 1 | (28, 28, 32) | 18 560 |
| ReLU 2 | (28, 28, 32) | 0 |
| MaxPool2d 1 | (14, 14, 32) | 0 |
| Dropout 1 | (14, 14, 32) | 0 |
| ResidualBlock 2 | (14, 14, 64) | 57 664 |
| ReLU 3 | (14, 14, 64) | 0 |
| MaxPool2d 2 | (7, 7, 64) | 0 |
| Dropout 2 | (7, 7, 64) | 0 |
| ResidualBlock 3 | (7, 7, 128) | 230 016 |
| ReLU 4 | (7, 7, 128) | 0 |
| MaxPool2d 3 | (3, 3, 128) | 0 |
| Dropout 3 | (3, 3, 128) | 0 |
| Flatten 1 | (1152,) | 0 |
| Dense 1 | (10,) | 11 530 |

**Razem:** 318 122 parametrów.

**Wagi:** `trained_models/conv_deep_res.npz`. **Konfiguracja i historia:** `trained_models/conv_deep_res.json`. **Wyniki:** dev 0,9198, czas 18,6 min (patrz `experiments.md`).

Kod warstw:

```python
layers = [Conv2d(32, f=3, s=1, p='same', bias=False, init=he),
 BatchNorm2d(axes=(0, 1, 2), momentum=0.1),
 ReLU(),
 ResidualBlock([
     Conv2d(32, f=3, s=1, p='same', bias=False, init=he),
     BatchNorm2d(axes=(0, 1, 2), momentum=0.1),
     ReLU(),
     Conv2d(32, f=3, s=1, p='same', bias=False, init=he),
     BatchNorm2d(axes=(0, 1, 2), momentum=0.1)]),
 ReLU(),
 MaxPool2d(f=2, s=2),
 Dropout(0.9675362118938842),
 ResidualBlock([
     Conv2d(64, f=3, s=1, p='same', bias=False, init=he),
     BatchNorm2d(axes=(0, 1, 2), momentum=0.1),
     ReLU(),
     Conv2d(64, f=3, s=1, p='same', bias=False, init=he),
     BatchNorm2d(axes=(0, 1, 2), momentum=0.1)]),
 ReLU(),
 MaxPool2d(f=2, s=2),
 Dropout(0.9675362118938842),
 ResidualBlock([
     Conv2d(128, f=3, s=1, p='same', bias=False, init=he),
     BatchNorm2d(axes=(0, 1, 2), momentum=0.1),
     ReLU(),
     Conv2d(128, f=3, s=1, p='same', bias=False, init=he),
     BatchNorm2d(axes=(0, 1, 2), momentum=0.1)]),
 ReLU(),
 MaxPool2d(f=2, s=2),
 Dropout(0.9675362118938842),
 Flatten(),
 Dense(10, bias=True, init=xavier)]
```
