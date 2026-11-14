# Poprawki w projekcie CNN od zera w numpy

## Na teraz: naprawa `Conv2d.backward` (pętla po pozycjach w `dX`)

Cel: skrócić czas treningu CNN. Profilowanie pokazało, że około 85% czasu zjada liczenie `dX` przez pętlę po pozycjach wyjścia.

- [ ] Zmierzyć czas `forward` i `backward` dla każdej warstwy (`perf_counter`) i zapisać wynik jako punkt odniesienia.
- [ ] Przepisać `dX` tak, jak `MaxPool2d.backward`: pętla po przesunięciach `f × f`, a w każdym przesunięciu operacja na całym batchu (bez pętli po pozycjach).
- [ ] Obsłużyć padding `'same'` (`self.cache['p']`) i zachować `float64`.
- [ ] Sprawdzić wynik względem `conv_naive` (forward i backward) i gradientem numerycznym (`grad_check`). Błąd względny ma być rzędu `10⁻⁷` lub mniejszy.
- [ ] Zmierzyć czas po zmianie i porównać z punktem odniesienia.
- [ ] Wpisać wynik do `experiments.md` razem z łączną liczbą parametrów i czasem treningu.

Lista zmian do wprowadzenia albo sprawdzenia przy budowie pierwszej wersji modelu (krok 4 planu). Każdy punkt ma status: **zrobione**, **do zrobienia** albo **do decyzji**.

## Zrobione

- **Data loader zwraca tablice, nie generatory.** `load_data` zwraca sześć tablic `(X, Y)` dla train, dev i test. Minibatche tworzy osobno `get_minibatch`, który tasuje tylko, gdy `rng` jest podany.
- **Ścieżki tylko w `load_data`.** Funkcje modelu, warstw, kosztu, optymalizatora i pętli treningowej nie dostają ścieżek.
- **Dzielnik w `compute_miu_std`.** Odchylenie liczone przez `1/X.size`. Po tej poprawce `check_norm` daje średnią ≈ 0 i odchylenie 1.
- **`SoftmaxCrossEntropy.forward`.** Logarytm softmaxu liczony z logitów (bez `eps`), koszt `J` zgodny z referencją, `dZ = (Ŷ − Y)/m`.
- **`SoftmaxCrossEntropy.backward` bez argumentów.** `Y` jest w cache, więc wywołanie to `loss.backward()`.
- **`grad_check` z argumentem `rng`.** Generator nie jest już zmienną globalną.
- **`tests.py` przemianowany na `utils.py`.** Zawiera funkcje pomocnicze: `check_norm`, `plot_distribution`, `check_minibatch_len`.
- **Docstringi.** Uzupełnione i zaktualizowane we wszystkich plikach, łącznie z `build`, `summary`, `predict`, `fit` i klasą `Conv2d`.
- **`accuracy` liczy dokładność.** Używa `np.mean(Y_hat == Y)`, a `evaluate` porównuje indeksy klas (`Y.argmax(axis=1)`).
- **`SGD` robi jeden krok.** Przyjmuje `(parameters, lr)` i aktualizuje wagi w miejscu. Zastąpił zepsuty `SGD` z pętlą po epokach. Nazwa funkcji w `optimizers.py` to teraz `SGD`.
- **`fit` jest metodą `Sequential`.** Pętla po epokach i batchach, ewaluacja na train i dev oraz wybór zwracanych wyników (`return_`) są w `model.py`. Plik `train.py` został usunięty.
- **`evaluate` bez `backward`.** Liczy koszt i dokładność na zbiorze, bez zmiany wag.
- **`Data` jako zbiór funkcji.** `load_data`, `get_minibatch`, `shuffle` i `compute_miu_std` są funkcjami modułowymi, a `seed` i `dtype` są argumentami.
- **Wcięcie w `data.py`.** Ciała funkcji są wcięte standardowo.
- **`main.ipynb` i API danych.** Wywołanie `load_data` w notebooku zgadza się z obecnym API.
- **`build` we wszystkich warstwach.** Każda warstwa z wagami tworzy je na podstawie `input_shape` i zwraca kształt wyjścia. Bezparametrowe zwracają kształt, `Flatten` ma `build`.
- **`Sequential.build`, `summary` i `predict`.** `build` tworzy wagi warstwa po warstwie, `summary` zwraca liczbę parametrów i kształty, `predict` zwraca przewidywane klasy.
- **`Conv2d` z `build` i `'same'`.** Konstruktor przyjmuje `(n_C, f, s, p)`, a `backward` używa `self.cache['p']`, więc padding `'same'` działa.
- **Pooling wektorowo.** `MaxPool2d` i `AvgPool2d` liczą forward przez `sliding_window_view`, a backward przez pętlę po `f × f` przesunięciach zamiast po pozycjach. `GlobalAvgPool` jest wektorowy w forward i backward. Sprawdzone względem wersji naiwnych i gradientem numerycznym (błędy rzędu `10⁻¹⁰`).

## Do zrobienia

- **`main.py` usunięty.** W historii są `run_MLP` i `run_conv_naive`, a na dysku ich nie ma. Zdecydować, czy przywrócić je (np. w `main.py` albo w notebooku), czy przenieść do notebooka.
- **`Sequential.summary()` buduje wagi ponownie.** `summary` wywołuje `build`, który losuje nowe wagi i przesuwa stan `rng`. Po treningu nadpisze wytrenowany model. Zmienić tak, żeby `summary` tylko liczył kształty i parametry bez losowania.
- **`layers.Flatten.backward`.** Klucz `'dA_flat'` w `grads` przechowuje już odspłaszczony gradient. Zmienić nazwę albo nie zapisywać go tam wcale.
- **Testy w `tests/`.** `check_minibatch_len` i podobne sprawdzenia przenieść do osobnego folderu testów (np. `pytest`).
- **Konwolucja `Conv2d.backward` nadal w pętlach.** Forward jest już wektorowy, a `dX` wciąż liczy się przez pętlę po pozycjach wyjścia. Zrobić tym samym sposobem co pooling (pętla po `f × f` przesunięciach), po pomiarze czasu.
- **`Sequential.parameters()`: zmiana formatu zwracanych wartości.** Obecnie zwraca płaską listę par `(parametr, gradient)` w kolejności warstw, a warstwy rozpoznaje po atrybucie `W`. Problemy: pary nie niosą nazwy, więc `grad_check` musi zgadywać etykietę po liczbie wymiarów; optymalizator z pamięcią (momentum, Adam) nie ma stabilnego klucza do swojego stanu; a gradienty trzeba mieć z poprzedniego `backward`. Propozycja: każda warstwa zwraca swoje parametry i gradienty jako słownik, a `Sequential` składa z tego słownik z nazwami w stylu `'layer2.W'`. Dotyczy: `model.py` (`parameters`, `hasattr`, `fit`), `optimizers.SGD` (pętla po parach), `grad_check` (pętla po parach, etykiety z `ndim` i indeksu), a także docstringi tych miejsc.

## Do decyzji

- **Flaga `training` w `Sequential` i warstwach.** Potrzebna przy dropout i batch norm (kroki 13 i 14). Na razie nie jest potrzebna.
- **Gdzie trzymać `J` z epoki.** `fit` zwraca historię jako słowniki wybrane przez `return_` po zakończeniu treningu. Ewentualnie `yield` po każdej epoce, jeśli chcesz podgląd w trakcie.
