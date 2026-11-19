# outputs/ – wyniki tuningu hiperparametrów

Każdy plik `tuning_<model>.csv` to wynik random searchu jednego z siedmiu modeli (`softmax_regression`, `mlp`, `mlp_deep`, `mlp_deep_res`, `conv`, `conv_deep`, `conv_deep_res`). Powstają w `src/main.ipynb` (sekcja „2. Tuning”) przez `tuning.main`.

**Protokół:** 20 losowych konfiguracji na model, każda trenowana 3 epoki na pierwszych 10 000 obrazach zbioru train, ocena na pełnym dev (5 000). Batch size 64, optymalizator Adam, float32. Zbiór testowy nie jest używany.

**Kolumny:**

| Kolumna | Znaczenie |
|---|---|
| `run` | numer przebiegu (0–19) |
| `seed` | seed wag i kolejności danych w tym przebiegu |
| `lr`, `lambda`, `Beta1`, `keepprob`, `augment` | losowane hiperparametry (`keepprob` jest wspólne dla wszystkich warstw Dropout; modele bez Dropout nie mają tej kolumny jako przeszukiwanej) |
| `batch_size`, `Beta2`, `layer<i>.n_C` itd. | wartości stałe, zapisane dla pełnej informacji (`layer3.0.n_C` to pierwsza warstwa bloku na pozycji 3) |
| `n_params` | liczba parametrów modelu |
| `best_dev_acc`, `best_dev_epoch` | najlepsza dokładność dev w tym przebiegu i epoka, w której wystąpiła |
| `final_train_acc`, `final_dev_acc` | dokładność po ostatniej epoce |
| `minutes` | czas przebiegu |

**Inne pliki:**

- `tuning01.csv` – starszy tuning (inny format kolumn, sieć z dwoma blokami Conv-BN-ReLU), z którego pochodzi konfiguracja modelu 92% w `experiments.md`.
- `tuning.csv` – wynik ręcznego uruchomienia `tuning.main` z notebooka (domyślna nazwa pliku).

Najlepszą konfigurację z pliku odczytuje `tuning.best_config('outputs/tuning_<model>.csv')`. Wytrenowane na pełnym zbiorze modele i ich wagi są w `../trained_models/`.
