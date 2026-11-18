# Poprawki w projekcie CNN od zera w numpy

Lista zadań i decyzji w projekcie. Każdy punkt ma status: **do zrobienia** albo **do decyzji**.

## Do zrobienia

- **Pomiar czasu per warstwa.** Dotąd mierzono tylko czas całej epoki. Dodać `perf_counter` dla `forward` i `backward` każdej warstwy, żeby mieć punkt odniesienia dla kolejnych zmian.
- **Testy w `tests/`.** `check_minibatch_len`, gradient check i porównania z wersjami naiwnymi przenieść do osobnego folderu testów (np. `pytest`).
- ~~**`ExponentialDecay` ma brać `alpha0` z modelu.**~~ Zrobione: `alpha0` jest argumentem `get_alpha(epoch, alpha0)`, a `fit` przekazuje `lr` (`lr_scheduler.get_alpha(i, lr)`).
- ~~**Nie liczyć okien dwa razy w `Conv2d.backward`.**~~ Zrobione: `forward` zapisuje macierz okien (`cols`, im2col) w cache, a `backward` jej używa.
- **Uproszczony backward batch norm.** Zamiast wielu pośrednich tablic liczyć gradient wzorem z jedną sumą po batchu: `dZ = γ / (N·√(σ²+ε)) · (N·dZ_out − Σ dZ_out − Z_norm · Σ(dZ_out · Z_norm))`. Sprawdzić gradient checkiem.
- **Metryki train na podzbiorze.** `fit` przepuszcza 55 tys. przykładów forward po każdej epoce. Liczyć train na podzbiorze (np. 5 tys.) albo co kilka epok. Dev zostawić w całości.
- **Float32 w treningu.** Trening w `float32`, gradient check zostawić w `float64`. Najpierw porównać wyniki z obecnymi.
- **Większy batch.** Batch 128 zmniejsza liczbę kroków Pythona o połowę, ale zmienia dynamikę uczenia. Porównać wyniki przed zmianą.
- ~~**Kopie w `tensordot` w `Conv2d`.**~~ Zrobione: `Conv2d` nie używa już `tensordot`. Okna są kopiowane raz w `forward` (`cols @ W_mat`).
- **L2 dla `ResidualBlock`: zostało `l2_penalty`.** Gradient L2 w `Sequential.backward` obejmuje już warstwy bloku, ale `l2_penalty` dalej sumuje tylko `self.layers`, więc kara i gradient opisują różne funkcje. Poprawić `l2_penalty` (ta sama lista wag co w `backward`, najlepiej wspólna metoda pomocnicza). W `backward` do poprawy jest też projekcja: `layer.projectile.grads['dW'] += ... * layer_residual.W` powinno używać `layer.projectile.W` (dla bloku z projekcją rzuca `AttributeError`). Potem sprawdzić grad check z `l2_lambda`, w tym dla wag bloku.
- **Wybór inicjalizacji projekcji w `ResidualBlock`.** `projectile` jest na sztywno budowany z `init=he`. Dodać argument konstruktora (np. `projection_init=he`, opcjonalnie `projection_scale`), żeby dało się wybrać inicjalizację tak jak w `Conv2d` i `Dense`, i przekazać go do `Conv2d` w `build()`.
- **`save` i `load_weights` dla `ResidualBlock`.** Obie metody iterują tylko po `self.layers` i szukają `W`, `b`, `gamma`, `beta` bezpośrednio na warstwie, więc wagi wewnątrz bloku (konwolucje, BN z `miu_run` i `std2_run`, projekcja 1×1) nie są zapisywane ani wczytywane. Model z blokiem po `load_weights` ma losowe wagi w bloku. Dodać zapis i odczyt z kluczami zawierającymi pozycję bloku i warstwy (np. `Block{i}.W{j}`, `Block{i}.projection.W`), z tym samym sprawdzaniem kształtów, i przetestować: zapis, nowy model, wczytanie, identyczne logity.
- **Rozwinąć `evaluate` o inne statystyki.** Dziś zwraca tylko stratę i accuracy. Dodać m.in. macierz pomyłek, precision, recall i F1 dla każdej klasy, accuracy per klasa oraz top-k accuracy, żeby było widać, które klasy (np. koszulka, bluza, płaszcz) model myli. Zwracać je w słowniku albo osobną funkcją w `evaluation.py`, a podstawowy zwrot `(J, acc)` zostawić, żeby nie zepsuć `fit`.

## Do decyzji

- ~~**Flaga `training` w `Sequential` i warstwach.**~~ Zrobione: `Sequential.forward(X, training=False)` przekazuje flagę do warstw, które ją przyjmują (BatchNorm2d, Dropout).
