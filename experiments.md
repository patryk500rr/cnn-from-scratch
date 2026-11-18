# Log eksperymentów

Każdy wpis opisuje jedną konfigurację i wynik. Dev służy do wyboru modelu, test zostaje nietknięty do samego końca projektu.

## Baseline: MLP (Flatten → Dense(128) → ReLU → Dense(10))

**Dane**
- Train: 55 000 przykładów (60 000 z CSV minus 5 000 na dev), tasowane co epokę.
- Dev: 5 000 przykładów wydzielonych z zbioru treningowego.
- Normalizacja: średnia i odchylenie liczone na train (skalary).
- Test: nie używany.

**Model**
- Flatten: (m, 28, 28, 1) → (m, 784).
- Dense(784 → 128), inicjalizacja He.
- ReLU.
- Dense(128 → 10), inicjalizacja He, logity.
- Strata: softmax + cross-entropy.

**Trening**
- Optymalizator: SGD (`step_SGD`), bez momentum i bez decay.
- Epoki: 50.
- Batch size: 64.
- Learning rate: 1e-2.
- Seed (tasowanie): `np.random.default_rng(123)`.

**Wyniki**
- Dokładność train: ok. 93%.
- Dokładność dev: ok. 90%.
- Luka train–dev: ok. 3 pp, bez wyraźnego przeuczenia.

**Uwagi**
- Dokładne wartości do uzupełnienia z historii `fit` (dla wszystkich epok).
- Przed tym wynikiem mogły być inne konfiguracje, które sprawdzałeś na dev. Wpisz je tutaj, żeby było widać, ile prób dotyczyło dev.

## Model_CNN: czas treningu przed naprawą `Conv2d.backward`

**Model**
- Conv2d(8, f=3, s=1, p='same') → ReLU → MaxPool2d → Conv2d(16, f=3, s=1, p='same') → ReLU → MaxPool2d → Flatten → Dense(64) → ReLU → Dense(10).
- Łączna liczba parametrów: **52 138**.

**Trening**
- Optymalizator: SGD, bez momentum i bez decay.
- Epoki: 20.
- Batch size: 64.
- Learning rate: 3e-3.
- Seed: `np.random.default_rng(seed=123)`.

**Czas**
- Średni czas na epokę: **1.15 min**.
- Wersja `Conv2d.backward`: stara, z pętlą po pozycjach wyjścia przy liczeniu `dA`.

**Wyniki**
- Dokładność train i dev: do uzupełnienia z historii `fit`.

**Uwagi**
- Czas dotyczy wersji sprzed wektoryzacji `dA`. Porównanie z wersją po naprawie: wpis po pomiarze.

## Model_CNN: czas treningu po naprawie `Conv2d.backward`

**Model**
- Ta sama architektura co wyżej: Conv2d(8) → ReLU → MaxPool2d → Conv2d(16) → ReLU → MaxPool2d → Flatten → Dense(64) → ReLU → Dense(10).
- Łączna liczba parametrów: **52 138**.

**Trening**
- Optymalizator: SGD, bez momentum i bez decay.
- Epoki: 20.
- Batch size: 64.
- Learning rate: 3e-3.
- Seed: `np.random.default_rng(seed=123)`.

**Czas**
- Średni czas na epokę: **0.204 min**.
- Wersja `Conv2d.backward`: nowa, z pętlą po przesunięciach `f × f` przy liczeniu `dA`.
- Przyspieszenie względem wersji sprzed naprawy (1.15 min): około 5.6×.

**Wyniki**
- Dokładność train: ok. 86%.
- Dokładność dev: ok. 85%.
- Luka train–dev: ok. 1 pp, bez wyraźnego przeuczenia.

**Uwagi**
- Nowy `backward` przeszedł porównanie z `conv_naive` (błąd około `1e-15`) i gradient check (błąd względny około `1e-10`).

## CNN z batch norm: 2 bloki Conv → BN → ReLU, bez poolingu

**Model**
- Conv2d(16, f=3, s=1, p='same', bias=False, init=he) → BatchNorm2d(axes=(0, 1, 2)) → ReLU → Conv2d(32, f=3, s=1, p='same', bias=False, init=he) → BatchNorm2d(axes=(0, 1, 2)) → ReLU → Flatten → Dense(10, bias=True, init=he).
- Brak poolingu: mapa 28×28×32 trafia do Flatten, więc Dense ma 25 088 wejść.
- Łączna liczba parametrów (z `summary()`): **255 738**. Z tego Dense 250 890, Conv 144 + 4 608, BN 32 + 64.

**Trening**
- Optymalizator: Adam, domyślne Beta1=0,9, Beta2=0,999, eps=1e-8.
- Harmonogram: `ExponentialDecay(alpha0=1e-2)`, domyślne gamma=0,95. Argument `lr=1e-2` w `fit` jest w tym przypadku ignorowany, bo harmonogram ma pierwszeństwo.
- Epoki: 10. Batch size: 64.
- Inicjalizacja: He.
- Seed: `rng` z notebooka, bez resetu przed budową modelu.

**Czas**
- Całkowity czas: 9,2 min, czyli około 0,92 min na epokę. Przed poprawką `Conv2d.backward` i bez BN porównanie nie jest bezpośrednie.

**Wyniki** (po epoce)
- Najlepsza dokładność dev: **0,905** w epoce 4.
- Dokładność dev w epoce 9: 0,891.
- Dokładność train w epoce 9: 0,980, loss train w epoce 9: 0,0559.
- Luka train–dev w epoce 9: ok. 9 pp, a dev spada po epoce 4 mimo dalszego spadku lossu train.

**Uwagi**
- Model wyraźnie się przeucza: dev przestaje rosnąć po około 4 epokach, a train dochodzi do 98%.
- Główne źródło parametrów to Dense po Flatten. Dodanie poolingu przed Flatten zmniejszy liczbę parametrów około cztery razy.
- Dla porównania z modelem z poolingiem (52 138 parametrów, dev ok. 85%) ten model ma lepszy dev, ale o dużo więcej parametrów.
- Loss dev nie został wypisany w tym uruchomieniu, więc nie ma go w tym wpisie.

## CNN z dropoutem i BN, 2 bloki po 16 filtrów, dev 92%

**Model**
- Conv2d(16, f=3, s=1, p='same', bias=False, init=he) → BatchNorm2d(axes=(0, 1, 2)) → ReLU → Dropout(keepprob=0,825) → Conv2d(16, f=3, s=1, p='same', bias=False, init=he) → BatchNorm2d(axes=(0, 1, 2)) → ReLU → Dropout(keepprob=0,825) → Flatten → Dense(10, init=scaled_normal, scale=0,01).
- L2: λ = 0,0098 (`l2_lambda`).
- Brak poolingu, więc Dense ma 16·28·28 = 12 544 wejść.
- Łączna liczba parametrów: **127 962** (policzone ręcznie: Conv 144 + 2 304, BN 64, Dense 125 440 + 10 bias; zgadza się z wierszem `run 4` w `outputs/tuning01.csv`).

**Trening**
- Optymalizator: Adam, Beta1 = 0,908447 (domyślne Beta2 = 0,999, eps = 1e-8).
- Learning rate: stały, lr = 0,001416 (bez harmonogramu).
- Epoki: 13. Batch size: 64.
- Dane: pełny zbiór train (55 000), bez augmentacji.
- Seed: `np.random.default_rng(123)`.
- Czas całkowity: do uzupełnienia.

**Wyniki**
- Dokładność train: **0,9652**.
- Dokładność dev: **0,9174**
- Dokładność test: **0,9171**. Test został użyty po raz pierwszy właśnie tutaj, na tej jednej konfiguracji. Jeśli dalej będziesz strojić na dev, ten wynik nie będzie już czystym szacunkiem.

**Uwagi**
- Hiperparametry (λ, keep_prob, Beta1, lr, liczba filtrów) pochodzą z wiersza `run 4` strojenia (`outputs/tuning01.csv`: 3 epoki na 10 000 przykładów, dev 0,871). Różnice względem tamtego runu: pełny zbiór train, 10 epok, inicjalizacja Dense `scaled_normal` (w `tuning.py` było `xavier`).
- Jeśli wynik 0,92 jest z tej konfiguracji, większość zysku względem 0,871 pochodzi prawdopodobnie z większej liczby danych i epok, ale to nie jest jeszcze sprawdzone.
- Dev i test dają prawie ten sam wynik (0,917 i 0,917), a train jest na poziomie 0,965. Luka train–dev to ok. 5 pp.
