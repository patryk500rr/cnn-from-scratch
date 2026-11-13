# Poprawki w projekcie CNN od zera w numpy

Lista zmian do wprowadzenia albo sprawdzenia przy budowie pierwszej wersji modelu (krok 4 planu). Każdy punkt ma status: **zrobione**, **do zrobienia** albo **do decyzji**.

## Zrobione

- **Data loader zwraca tablice, nie generatory.** `load_data` zwraca sześć tablic `(X, Y)` dla train, dev i test. Minibatche tworzy osobno `get_minibatch`, który tasuje tylko, gdy `rng` jest podany.
- **Ścieżki tylko w `load_data`.** Funkcje modelu, warstw, kosztu, optymalizatora i pętli treningowej nie dostają ścieżek.
- **Dzielnik w `compute_miu_std`.** Odchylenie liczone przez `1/X.size`. Po tej poprawce `check_norm` daje średnią ≈ 0 i odchylenie 1.
- **`SoftmaxCrossEntropy.forward`.** Logarytm softmaxu liczony z logitów (bez `eps`), koszt `J` zgodny z referencją, `dZ = (Ŷ − Y)/m`.
- **`SoftmaxCrossEntropy.backward` bez argumentów.** `Y` jest w cache, więc wywołanie to `loss.backward()`.
- **`grad_check` z argumentem `rng`.** Generator nie jest już zmienną globalną.
- **`tests.py` przemianowany na `utils.py`.** Zawiera funkcje pomocnicze: `check_norm`, `plot_distribution`, `check_minibatch_len`.
- **Docstringi.** Uzupełnione i zaktualizowane we wszystkich plikach, łącznie z `build` i klasą `Conv2d`.
- **`accuracy` liczy dokładność.** Używa `np.mean(Y_hat == Y)`, a `evaluate` porównuje indeksy klas (`Y.argmax(axis=1)`).
- **`step_SGD` robi jeden krok.** Przyjmuje `(parameters, lr)` i aktualizuje wagi w miejscu. Zastąpił zepsuty `SGD`.
- **Podział na krok i pętlę.** `step_SGD` aktualizuje wagi, a pętla po batchach i epokach jest w `train.fit`, który zwraca wyniki każdej epoki przez `yield`.
- **`evaluate` bez `backward`.** Liczy koszt i dokładność na zbiorze, bez zmiany wag.
- **`Data` jako zbiór funkcji.** `load_data`, `get_minibatch`, `shuffle` i `compute_miu_std` są funkcjami modułowymi, a `seed` i `dtype` są argumentami.
- **Wcięcie w `data.py`.** Ciała funkcji są wcięte standardowo.
- **`main.ipynb` i API danych.** Wywołanie `load_data` w notebooku zgadza się z obecnym API.
- **`build` we wszystkich warstwach.** Każda warstwa z wagami tworzy je na podstawie `input_shape` i zwraca kształt wyjścia. Bezparametrowe zwracają kształt, `Flatten` ma `build`.
- **`Conv2d` z `build` i `'same'`.** Konstruktor przyjmuje `(n_C, f, s, p)`, a `backward` używa `self.cache['p']`, więc padding `'same'` działa.

## Do zrobienia

- **`layers.Flatten.backward`.** Klucz `'dA_flat'` w `grads` przechowuje już odspłaszczony gradient. Zmienić nazwę albo nie zapisywać go tam wcale.
- **Testy w `tests/`.** `check_minibatch_len` i podobne sprawdzenia przenieść do osobnego folderu testów (np. `pytest`).
- **Pooling wektorowo.** `MaxPool2d` i `AvgPool2d` liczą okna w pętli po przykładach, pozycjach i kanałach, czyli około 200 tys. wywołań `np.max` na batch 64 dla mapy 28×28×16. Wersja wektorowa: dla `s = f` zmiana kształtu na `(m, H/f, f, W/f, f, C)` i `max` albo `mean` po osiach `f`; dla nakładających się okien `sliding_window_view`. Backward wektorowo przez maskę argmax z tych samych okien. Naiwne wersje (`*_naive`) zostają jako wzorzec do porównań. Zrobić po zmierzeniu, że pooling (a nie konwolucja) jest wąskim gardłem.
- **Konwolucja `Conv2d.backward` nadal w pętlach.** Forward jest już wektorowy, a `dX` wciąż liczy się przez pętlę po pozycjach. Do zrobienia po pierwszym działającym modelu, przy sprawdzaniu czasu.
- **`Sequential.parameters()`: zmiana formatu zwracanych wartości.** Obecnie zwraca płaską listę par `(parametr, gradient)` w kolejności warstw, a warstwy rozpoznaje po atrybucie `W`. Problemy: pary nie niosą nazwy, więc `grad_check` musi zgadywać etykietę po liczbie wymiarów; optymalizator z pamięcią (momentum, Adam) nie ma stabilnego klucza do swojego stanu; a gradienty trzeba mieć z poprzedniego `backward`. Propozycja: każda warstwa zwraca swoje parametry i gradienty jako słownik, a `Sequential` składa z tego słownik z nazwami w stylu `'layer2.W'`. Dotyczy: `model.py` (`parameters`, `hasattr`), `optimizers.step_SGD` (pętla po parach), `train.fit` (wywołanie optymalizatora), `grad_check` (pętla po parach, etykiety z `ndim` i indeksu), a także docstringi tych miejsc.
- **`fit` wewnątrz klasy `Sequential`.** Pętla treningowa (batche, epoki, ewaluacja na dev) ma być metodą `Sequential.fit`, a nie osobną funkcją w `train.py`. Wtedy `fit` ma dostęp do warstw i `parameters()`. Dotyczy: `model.py` (nowa metoda), `train.py` (usunąć albo zostawić jako cienki wrapper), `main.py` (wywołanie), notebooki (wywołanie). Ustalić przy zmianie formatu `parameters()`, bo `fit` go używa.

## Do decyzji

- **Flaga `training` w `Sequential` i warstwach.** Potrzebna przy dropout i batch norm (kroki 13 i 14). Na razie nie jest potrzebna.
- **Gdzie trzymać `J` z epoki.** `fit` może zwracać historię jako słownik albo `yield` na każdą epokę. Obecnie `yield` na każdą epokę jest wybrany.
