# Poprawki w projekcie CNN od zera w numpy

Lista zadań i decyzji w projekcie. Tu zostają tylko rzeczy **do zrobienia**; zrobione punkty są usuwane.

## Do zrobienia

- **Pomiar czasu per warstwa.** Dotąd mierzono tylko czas całej epoki. Dodać `perf_counter` dla `forward` i `backward` każdej warstwy, żeby mieć punkt odniesienia dla kolejnych zmian.
- **Testy w `tests/`.** `check_minibatch_len`, gradient check i porównania z wersjami naiwnymi przenieść do osobnego folderu testów (np. `pytest`).
- **Uproszczony backward batch norm.** Zamiast wielu pośrednich tablic liczyć gradient wzorem z jedną sumą po batchu: `dZ = γ / (N·√(σ²+ε)) · (N·dZ_out − Σ dZ_out − Z_norm · Σ(dZ_out · Z_norm))`. Sprawdzić gradient checkiem.
- **Metryki train na podzbiorze.** `fit` przepuszcza 55 tys. przykładów forward po każdej epoce. Liczyć train na podzbiorze (np. 5 tys.) albo co kilka epok. Dev zostawić w całości.
- **Dokończyć float32.** `tuning.py` trenuje już w float32 (`Sequential.astype`), ale nie porównano jeszcze wyników z float64. Pooling i `conv_naive` mają float64 wpisane na sztywno (`dtype='float64'` w `backward`), więc model z poolingiem nie będzie w pełni float32. Gradient check zostaje w float64.
- **Większy batch.** Batch 128 zmniejsza liczbę kroków Pythona o połowę, ale zmienia dynamikę uczenia. Porównać wyniki przed zmianą.
- **Wybór inicjalizacji projekcji w `ResidualBlock`.** `projectile` jest na sztywno budowany z `init=he`. Dodać argument konstruktora (np. `projection_init=he`, opcjonalnie `projection_scale`), żeby dało się wybrać inicjalizację tak jak w `Conv2d` i `Dense`, i przekazać go do `Conv2d` w `build()`.
- **Rozwinąć `evaluate` o inne statystyki.** Dziś zwraca tylko stratę i accuracy. Dodać m.in. macierz pomyłek, precision, recall i F1 dla każdej klasy, accuracy per klasa oraz top-k accuracy, żeby było widać, które klasy (np. koszulka, bluza, płaszcz) model myli. Zwracać je w słowniku albo osobną funkcją w `evaluation.py`, a podstawowy zwrot `(J, acc)` zostawić, żeby nie zepsuć `fit`.
- **Zmiana wysokości i szerokości wewnątrz `ResidualBlock`.** Blok zakłada dziś, że gałąź główna zachowuje H i W, a projekcja jest konwolucją 1×1 ze stride 1 (zmienia tylko liczbę kanałów). Dziś `build()` rzuca `ValueError`, gdy zmienia się H, W albo liczba osi (zamiast psuć się w treningu). Dodać obsługę stride > 1 w gałęzi głównej: projekcja Conv 1×1 z tym samym stride i docelową liczbą kanałów (opcjonalnie z BN), warunek projekcji `input_shape != output_shape` obejmujący H, W i C. Sprawdzić gradient checkiem na wejściu 8×8 ze stride 2.

