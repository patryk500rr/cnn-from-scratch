# CNN from scratch

Sieć konwolucyjna napisana od zera w numpy (Conv2d, BatchNorm, ReLU, MaxPool, Dropout,
ResidualBlock, Dense, softmax + cross-entropy, SGD i Adam) i trenowana na Fashion-MNIST.
Plan projektu: `plan.md`, log eksperymentów: `experiments.md`, lista poprawek: `poprawki.md`.

## Instalacja

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dane

Dane **nie są w repozytorium** (folder `data/` jest w `.gitignore`). Pobierz Fashion-MNIST w
wersji CSV (np. z Kaggle: "Fashion MNIST" od zalando-research) i umieść dwa pliki dokładnie
pod tymi nazwami:

```
data/fashion-mnist_train.csv   # 60 000 wierszy: kolumna `label` + 784 kolumny pikseli
data/fashion-mnist_test.csv    # 10 000 wierszy, ten sam format
```

`load_data` z `src/data.py` tasuje train (seed 123), odcina 5 000 przykładów na dev
i normalizuje dane średnią i odchyleniem policzonymi na train.

## Uruchomienie

Notebook z eksperymentami: `src/main.ipynb` (uruchamiaj Jupyter z folderu `src/`, bo moduły
są importowane bez prefiksu pakietu).

```bash
cd src
jupyter notebook main.ipynb
```

Strojenie hiperparametrów (random search), wyniki dopisywane do `outputs/tuning.csv`. Podajesz
listę warstw w kolejności, klasę optymalizatora i opcjonalnie harmonogram lr. `tuning.py` sam
dobiera hiperparametry, które mają sens dla tej sieci (lr, λ, batch size, augmentacja, bety
optymalizatora, liczba filtrów i rozmiar filtra w każdym Conv2d, neurony w ukrytych Dense,
keep_prob w każdym Dropout):

```python
from tuning import list_hyperparameters, main
list_hyperparameters(layers, Adam)           # wypisuje przeszukiwane hiperparametry i zakresy
main(layers, Adam, n_runs=20, data=data)     # space={...} zmienia zakres albo ustala wartość
```

Z terminala (domyślne warstwy z `default_layers()`):

```bash
python3 src/tuning.py
```

Argument `data=` to szóstka z `load_data`, a `out_path=` to własna ścieżka wyniku. Plik CSV ma
kolumny zależne od przestrzeni, więc inna sieć wymaga innego `out_path`.

## Struktura

- `src/layers.py` – warstwy i `ResidualBlock`
- `src/model.py` – `Sequential` (fit, evaluate, summary, save/load, L2)
- `src/optimizers.py`, `src/init.py` – optymalizatory (SGD, Adam), inicjalizacje
- `src/data.py` – wczytywanie, mini-batche, augmentacja
- `src/evaluation.py`, `src/grad_check.py`, `src/utils.py` – ewaluacja, gradient check, pomocnicze
- `src/tuning.py` – random search
