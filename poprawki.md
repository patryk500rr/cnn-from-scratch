# Poprawki w projekcie CNN od zera w numpy

Lista zadań i decyzji w projekcie. Każdy punkt ma status: **do zrobienia** albo **do decyzji**.

## Do zrobienia

- **Pomiar czasu per warstwa.** Dotąd mierzono tylko czas całej epoki. Dodać `perf_counter` dla `forward` i `backward` każdej warstwy, żeby mieć punkt odniesienia dla kolejnych zmian.
- **Testy w `tests/`.** `check_minibatch_len`, gradient check i porównania z wersjami naiwnymi przenieść do osobnego folderu testów (np. `pytest`).
- **`ExponentialDecay` ma brać `alpha0` z modelu.** Obecnie `alpha0` podajesz ręcznie w konstruktorze harmonogramu, więc może się rozjechać z `lr` przekazanym do `fit`. Wartość startowa powinna pochodzić z jednego miejsca, np. z argumentu `lr` w `fit` albo z atrybutu modelu/optymalizatora, żeby nie trzeba było podawać jej dwa razy.

## Do decyzji

- **Flaga `training` w `Sequential` i warstwach.** Potrzebna przy dropout i batch norm (kroki 13 i 14). Na razie nie jest potrzebna.
