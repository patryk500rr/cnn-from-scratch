# Poprawki w projekcie CNN od zera w numpy

Lista zadań i decyzji w projekcie. Każdy punkt ma status: **do zrobienia** albo **do decyzji**.

## Do zrobienia

- **Pomiar czasu per warstwa.** Dotąd mierzono tylko czas całej epoki. Dodać `perf_counter` dla `forward` i `backward` każdej warstwy, żeby mieć punkt odniesienia dla kolejnych zmian.
- **Testy w `tests/`.** `check_minibatch_len`, gradient check i porównania z wersjami naiwnymi przenieść do osobnego folderu testów (np. `pytest`).
- ~~**`ExponentialDecay` ma brać `alpha0` z modelu.**~~ Zrobione: `alpha0` jest argumentem `get_alpha(epoch, alpha0)`, a `fit` ma przekazywać `lr`. Do poprawy zostało wywołanie `get_alpha(i)` w `model.py`, które musi dostać `lr`.
- **Nie liczyć okien dwa razy w `Conv2d.backward`.** `backward` ponownie robi `np.pad` i `sliding_window_view` dla tego samego wejścia co `forward`. Zapisać przekształcone okna w `forward` i użyć ich w `backward`.
- **Uproszczony backward batch norm.** Zamiast wielu pośrednich tablic liczyć gradient wzorem z jedną sumą po batchu: `dZ = γ / (N·√(σ²+ε)) · (N·dZ_out − Σ dZ_out − Z_norm · Σ(dZ_out · Z_norm))`. Sprawdzić gradient checkiem.
- **Metryki train na podzbiorze.** `fit` przepuszcza 55 tys. przykładów forward po każdej epoce. Liczyć train na podzbiorze (np. 5 tys.) albo co kilka epok. Dev zostawić w całości.
- **Float32 w treningu.** Trening w `float32`, gradient check zostawić w `float64`. Najpierw porównać wyniki z obecnymi.
- **Większy batch.** Batch 128 zmniejsza liczbę kroków Pythona o połowę, ale zmienia dynamikę uczenia. Porównać wyniki przed zmianą.
- **Kopie w `tensordot` w `Conv2d`.** Sprawdzić, czy `tensordot` alokuje kopię okien z `sliding_window_view` (przez `transpose` i `reshape`), za pomocą `np.may_share_memory` i `tracemalloc`. Jeśli tak, zastanowić się nad układem danych, który nie wymusza kopii, albo nad `einsum`.

## Do decyzji

- **Flaga `training` w `Sequential` i warstwach.** Potrzebna przy dropout i batch norm (kroki 13 i 14). Na razie nie jest potrzebna.
