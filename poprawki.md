# Poprawki w projekcie CNN od zera w numpy

Lista zmian do wprowadzenia albo sprawdzenia przy budowie pierwszej wersji modelu (krok 4 planu). Każdy punkt ma status: **zrobione**, **do zrobienia** albo **do decyzji**.

## Zrobione

- **Data loader zwraca tablice, nie generatory.** `Data.load_data` zwraca sześć tablic `(X, Y)` dla train, dev i test. Minibatche tworzy osobno `Data.get_minibatch`, który tasuje tylko, gdy `do_shuffle=True`.
- **Ścieżki tylko w `Data`.** Ścieżki są przekazywane do konstruktora `Data`, a nie do każdej funkcji. `SGD` nie dostaje już ścieżek.
- **Dzielnik w `compute_miu_std`.** Odchylenie liczone przez `1/X.size`, a nie `1/(m·X.size)`. Po tej poprawce `check_norm` daje średnią ≈ 0 i odchylenie 1.
- **`SoftmaxCrossEntropy.forward`.** Logarytm softmaxu liczony z logitów (bez `eps`), koszt `J` zgodny z referencją, `dZ = (Ŷ − Y)/m`. Sprawdzone dla dużych logitów.
- **`SoftmaxCrossEntropy.backward` bez argumentów.** `Y` jest w cache, więc wywołanie to `loss.backward()`.
- **`grad_check` z argumentem `rng`.** Generator nie jest już zmienną globalną. Wywołanie `loss.backward()` jest poprawne.
- **`SGD` robi jeden krok.** Forward, koszt, backward i aktualizacja wag na jednym batchu. Zwraca koszt batcha.
- **`tests.py` przemianowany na `utils.py`.** Zawiera funkcje pomocnicze: `check_norm`, `plot_distribution`, `check_minibatch_len`.
- **Docstringi.** Zaktualizowane w `data.py`, `utils.py`, `grad_check.py`, `evaluation.py` i w nagłówku `layers.py`.

## Do zrobienia

- **`evaluation.accuracy`: błędna logika.** Obecnie `np.sum(np.where(Y_hat == Y))` zwraca sumę indeksów, a nie dokładność. Powinno być `np.mean(Y_hat == Y)`. Dokładność liczona na indeksach klas, a nie na one-hot. Docstring już opisuje poprawne zachowanie.
- **`optimizers.SGD`: plik jest zepsuty.** Sygnatura to `SGD(parameters, loss, X, Y, lr)`, ale w ciele używane są `model` i `J`, których nie ma w zakresie. Wymaga przywrócenia wersji z forward, kosztem, backward i aktualizacją wag.
- **Podział `SGD` i `fit`.** `SGD` ma być tylko krokiem aktualizacji `step(params)`. Pętlę po batchach i epokach, z historią kosztu i ewaluacją na dev, napisz w `train.py` jako `fit`.
- **Nowy `fit` i `evaluate`.** `evaluate(model, loss, X, Y, batch_size)` bez `backward` (plan, krok 4). `fit` oddaje wyniki każdej epoki przez `yield`.
- **`Data` jako zbiór funkcji.** Klasa trzyma tylko ścieżki i `rng`. Funkcje z argumentami byłyby prostsze. Do rozważenia.
- **`main.ipynb` używa starego API.** Komórki zakładają `load_data` zwracające generatory. Trzeba je zaktualizować.
- **Testy w `tests/`.** `check_minibatch_len` i podobne sprawdzenia przenieść do osobnego folderu testów (np. `pytest`).
- **`layers.Flatten.backward`.** Klucz `'dA_flat'` w `grads` przechowuje już odspłaszczony gradient. Zmienić nazwę albo nie zapisywać go tam wcale.
- **`data.load_data`: wcięcie ciała funkcji.** Ciało jest wcięte o 12 spacji zamiast 8. Składnia jest poprawna, ale niespójna z resztą pliku.

## Do decyzji

- **Flaga `training` w `Sequential` i warstwach.** Potrzebna przy dropout i batch norm (kroki 13 i 14). Na razie nie jest potrzebna.
- **Gdzie trzymać `J` z epoki.** `fit` może zwracać historię jako słownik albo `yield` na każdą epokę. Ustalić przy pisaniu `fit`.
