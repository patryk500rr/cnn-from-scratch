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
- ~~**L2 dla `ResidualBlock`.**~~ Zrobione: `backward` i `l2_penalty` liczą te same wagi (warstwy bloku i projekcja). Grad check z `l2_lambda` daje błędy rzędu 1e-9 dla bloku z projekcją i bez.
- **Wybór inicjalizacji projekcji w `ResidualBlock`.** `projectile` jest na sztywno budowany z `init=he`. Dodać argument konstruktora (np. `projection_init=he`, opcjonalnie `projection_scale`), żeby dało się wybrać inicjalizację tak jak w `Conv2d` i `Dense`, i przekazać go do `Conv2d` w `build()`.
- ~~**`save` i `load_weights` dla `ResidualBlock`.**~~ Zrobione: klucze `Block<k>-W<j>` itd. wraz ze statystykami BN (`miu_run`, `std2_run`) i projekcją. Test: zapis, nowy model, wczytanie daje identyczne wyjście.
- **Rozwinąć `evaluate` o inne statystyki.** Dziś zwraca tylko stratę i accuracy. Dodać m.in. macierz pomyłek, precision, recall i F1 dla każdej klasy, accuracy per klasa oraz top-k accuracy, żeby było widać, które klasy (np. koszulka, bluza, płaszcz) model myli. Zwracać je w słowniku albo osobną funkcją w `evaluation.py`, a podstawowy zwrot `(J, acc)` zostawić, żeby nie zepsuć `fit`.
- **Zmiana wysokości i szerokości wewnątrz `ResidualBlock`.** Blok zakłada dziś, że gałąź główna zachowuje H i W, a projekcja jest konwolucją 1×1 ze stride 1 (zmienia tylko liczbę kanałów). `build()` porównuje `input_shape` z `output_shape` i dla zmiany H lub W zbuduje projekcję, która nie zmniejszy mapy, więc suma `F(x) + P(x)` rzuci błąd kształtów. Dodać obsługę stride > 1 w gałęzi głównej: projekcja Conv 1×1 z tym samym stride i docelową liczbą kanałów (opcjonalnie z BN), warunek projekcji `input_shape != output_shape` obejmujący H, W i C, oraz asercję z czytelnym komunikatem, gdy kształty F(a) i skrótu się nie zgadzają. Sprawdzić gradient checkiem na wejściu 8×8 ze stride 2.

## Do decyzji

- ~~**Flaga `training` w `Sequential` i warstwach.**~~ Zrobione: `Sequential.forward(X, training=False)` przekazuje flagę do warstw, które ją przyjmują (BatchNorm2d, Dropout).
