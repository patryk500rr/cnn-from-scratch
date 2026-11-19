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


## Siedem modeli: tuning (20 runów na model) i trening na pełnym zbiorze

**Protokół**
- Dane: Fashion-MNIST, train 55 000, dev 5 000, float32. Zbiór testowy nietknięty, poza jednym pomiarem dla modelu wybranego na dev (patrz niżej).
- Tuning: random search, 20 konfiguracji na model, 3 epoki na pierwszych 10 000 obrazach train, ocena na pełnym dev. Strojone: `lr` (log 2e-4 do 1e-2), `lambda` (L2, log 1e-5 do 1e-1), `Beta1` (1 − log 1e-2 do 2e-1), `keepprob` (jedno wspólne dla wszystkich Dropout, 0,5 do 1) i `augment` (przesunięcia, tak/nie). Stałe: Adam, Beta2 = 0,999, batch 64, szerokości warstw.
- Wszystkie modele użyły tego samego seeda generatora konfiguracji (0), więc modele z Dropout dostały te same 20 losowań. Regresja softmax nie ma `keepprob`, więc ma o jedną wylosowaną wartość mniej na run, i jej losowania różnią się od runu 1.
- Trening końcowy: najlepsza konfiguracja z tuningu danego modelu, pełny train, 8 epok, stały `lr`, bez harmonogramu.
- Batch size 64 wybrany pomiarem na M4 Pro (24 GB): w modelach konwolucyjnych czas na obraz jest taki sam dla batcha 32, 64 i 128, a szczyt pamięci przy 64 to ok. 0,7 GB.

**Wyniki** (dokładność po ostatniej epoce, numeracja epok od 0)

| Model | Parametry | Czas [min] | Train | Dev | Najlepszy dev (epoka) | lr | λ | Beta1 | keepprob | augment |
|---|---|---|---|---|---|---|---|---|---|---|
| `softmax_regression` | 7 850 | 0,0 | 0,8563 | 0,8396 | 0,8494 (3) | 0,00255 | 0,00567 | 0,9679 | brak | nie |
| `mlp` | 203 786 | 0,1 | 0,9167 | 0,8926 | 0,8942 (6) | 0,00140 | 0,00131 | 0,8947 | 0,707 | nie |
| `mlp_deep` | 576 010 | 0,3 | 0,9293 | 0,9020 | 0,9020 (7) | 0,00049 | 0,00311 | 0,9871 | 0,916 | nie |
| `mlp_deep_res` | 600 074 | 0,4 | 0,9328 | 0,9004 | 0,9004 (7) | 0,00057 | 0,00001 | 0,9895 | 0,907 | nie |
| `conv` | 50 282 | 4,7 | 0,9410 | 0,9118 | 0,9178 (5) | 0,00882 | 0,00001 | 0,8671 | 0,991 | nie |
| `conv_deep` | 298 410 | 13,6 | 0,9522 | 0,9262 | 0,9262 (7) | 0,00036 | 0,07772 | 0,8562 | 0,911 | nie |
| `conv_deep_res` | 318 122 | 18,6 | 0,9499 | 0,9198 | 0,9244 (6) | 0,00215 | 0,00828 | 0,9490 | 0,968 | nie |

- Najlepszy na dev: **`conv_deep`**, dev 0,9262 (92,62%), test **0,9271** (92,71%, strata 0,2045). Test policzony raz, tylko dla tego modelu (`trained_models/test_result.json`).
- **Cel 96% na dev nie został osiągnięty.** Brakuje 3,4 pp.

**Uwagi**
- Kolejność po dev: regresja softmax < MLP < MLP głęboki ≈ MLP głęboki z resblockami < CNN < CNN głęboki z resblockami < CNN głęboki. Konwolucje dają 1-2,5 pp więcej niż MLP (`conv_deep` 0,9262 wobec 0,9020 dla `mlp_deep`).
- Bloki residualne nie pomogły przy tej głębokości (3 bloki): MLP −0,16 pp, CNN −0,64 pp względem wersji bez nich. Dev ma 5 000 przykładów (błąd standardowy ok. 0,4 pp), a każdy model to jeden przebieg i jeden seed, więc te różnice nie są rozróżnialne od szumu.
- Augmentacja przesunięciami: w tuningu (3 epoki) średni dev z augmentacją był niższy niż bez niej w każdym z 7 modeli (o 3-8 pp), więc żaden zwycięzca jej nie używa. Krótki tuning faworyzuje brak augmentacji, bo augmentacja spowalnia zbieżność. Nie wiadomo, czy przy dłuższym treningu byłaby korzystna. To najbardziej obiecujący niesprawdzony kierunek.
- Model nie jest zbieżny po 8 epokach: najlepszy dev przypada na ostatnią epokę (7) dla `mlp_deep`, `mlp_deep_res` i `conv_deep`. Dłuższy trening z malejącym lr prawdopodobnie da trochę więcej.
- `conv_deep`: wybrane λ = 0,0777 leży blisko górnej granicy przeszukiwanego zakresu (0,1), więc zakres warto poszerzyć. Luka train-dev wynosi 2,6 pp.
- 96% na Fashion-MNIST jest poza zasięgiem takich małych sieci (ogólna wiedza spoza projektu, tu niesprawdzona): proste CNN dają zwykle 93-94%, a wyniki ok. 96% wymagają dużo większych sieci, silnej augmentacji i długiego treningu.
- Pliki `outputs/tuning_*.csv` i `trained_models/*` zapisywały częściowo dwa równoległe przebiegi (skrypt w tle i kernel notebooka), bo zapisują pod tymi samymi nazwami. Dla spójności `conv` został wytrenowany ponownie z najlepszą konfiguracją z aktualnego CSV. Dla wszystkich siedmiu modeli konfiguracja w JSON-ie jest teraz najlepszą z odpowiedniego CSV.
- Opis plików: `outputs/README.md`, `trained_models/README.md`. Architektury z kodem do odtworzenia: `architektury.md`.

## Regresja softmax (jedna warstwa Dense) (`softmax_regression`): dev 0,8396

**Model**
- Flatten → Dense(10).
- Jedna warstwa Dense(10) z biasem, inicjalizacja Xavier, logity, softmax + cross-entropy.
- Łączna liczba parametrów: **7 850**.
- Regularyzacja: L2 λ = 0,00567, bez Dropout, augmentacja: nie.

**Trening**
- Optymalizator: Adam, Beta1 = 0,9679, Beta2 = 0,999, eps = 1e-8.
- Learning rate: stały, lr = 0,00255.
- Epoki: 8. Batch size: 64. Dane: pełny train (55 000), float32.
- Seed: 379523 (wagi i kolejność danych).
- Czas całkowity: **0,0 min** (całe `fit`, z liczeniem train i dev po każdej epoce).

**Wyniki**
- Dokładność train (ostatnia epoka): 0,8563. Dokładność dev (ostatnia epoka): **0,8396**. Najlepszy dev 0,8494 w epoce 3.
- Luka train-dev: 1,7 pp.
- Tuning: run 7 z 20, dev 0,8346 po 3 epokach na 10 000 obrazach.

**Pliki**
- Wagi: `trained_models/softmax_regression.npz`. Konfiguracja, architektura jako kod i historia z `fit`: `trained_models/softmax_regression.json`. Wyniki tuningu: `outputs/tuning_softmax_regression.csv`.

## MLP, jedna warstwa ukryta (256) (`mlp`): dev 0,8926

**Model**
- Flatten → Dense(256) → BN → ReLU → Dropout(0,707) → Dense(10).
- Warstwy Conv i Dense (poza ostatnią) bez biasu, po nich BatchNorm i ReLU; ostatni Dense z biasem i inicjalizacją Xavier, logity.
- Łączna liczba parametrów: **203 786**.
- Regularyzacja: L2 λ = 0,00131, Dropout keepprob = 0,707 (to samo dla wszystkich warstw Dropout), BatchNorm, augmentacja: nie.

**Trening**
- Optymalizator: Adam, Beta1 = 0,8947, Beta2 = 0,999, eps = 1e-8.
- Learning rate: stały, lr = 0,00140.
- Epoki: 8. Batch size: 64. Dane: pełny train (55 000), float32.
- Seed: 834954 (wagi i kolejność danych).
- Czas całkowity: **0,1 min** (całe `fit`, z liczeniem train i dev po każdej epoce).

**Wyniki**
- Dokładność train (ostatnia epoka): 0,9167. Dokładność dev (ostatnia epoka): **0,8926**. Najlepszy dev 0,8942 w epoce 6.
- Luka train-dev: 2,4 pp.
- Tuning: run 16 z 20, dev 0,8566 po 3 epokach na 10 000 obrazach.

**Pliki**
- Wagi: `trained_models/mlp.npz`. Konfiguracja, architektura jako kod i historia z `fit`: `trained_models/mlp.json`. Wyniki tuningu: `outputs/tuning_mlp.csv`.

## MLP głęboki, 4 warstwy ukryte (512-256-128-64) (`mlp_deep`): dev 0,9020

**Model**
- Flatten → Dense(512) → BN → ReLU → Dropout(0,916) → Dense(256) → BN → ReLU → Dropout(0,916) → Dense(128) → BN → ReLU → Dropout(0,916) → Dense(64) → BN → ReLU → Dropout(0,916) → Dense(10).
- Warstwy Conv i Dense (poza ostatnią) bez biasu, po nich BatchNorm i ReLU; ostatni Dense z biasem i inicjalizacją Xavier, logity.
- Łączna liczba parametrów: **576 010**.
- Regularyzacja: L2 λ = 0,00311, Dropout keepprob = 0,916 (to samo dla wszystkich warstw Dropout), BatchNorm, augmentacja: nie.

**Trening**
- Optymalizator: Adam, Beta1 = 0,9871, Beta2 = 0,999, eps = 1e-8.
- Learning rate: stały, lr = 0,00049.
- Epoki: 8. Batch size: 64. Dane: pełny train (55 000), float32.
- Seed: 328175 (wagi i kolejność danych).
- Czas całkowity: **0,3 min** (całe `fit`, z liczeniem train i dev po każdej epoce).

**Wyniki**
- Dokładność train (ostatnia epoka): 0,9293. Dokładność dev (ostatnia epoka): **0,9020**. Najlepszy dev 0,9020 w epoce 7.
- Luka train-dev: 2,7 pp.
- Tuning: run 9 z 20, dev 0,8570 po 3 epokach na 10 000 obrazach.

**Pliki**
- Wagi: `trained_models/mlp_deep.npz`. Konfiguracja, architektura jako kod i historia z `fit`: `trained_models/mlp_deep.json`. Wyniki tuningu: `outputs/tuning_mlp_deep.csv`.

## MLP głęboki z blokami residualnymi (3 bloki po 256) (`mlp_deep_res`): dev 0,9004

**Model**
- Flatten → Dense(256) → BN → ReLU → ResidualBlock[Dense(256)-BN-ReLU-Dense(256)-BN] → ReLU → Dropout(0,907) → ResidualBlock[Dense(256)-BN-ReLU-Dense(256)-BN] → ReLU → Dropout(0,907) → ResidualBlock[Dense(256)-BN-ReLU-Dense(256)-BN] → ReLU → Dropout(0,907) → Dense(10).
- Warstwy Conv i Dense (poza ostatnią) bez biasu, po nich BatchNorm i ReLU; ostatni Dense z biasem i inicjalizacją Xavier, logity.
- Łączna liczba parametrów: **600 074**.
- Regularyzacja: L2 λ = 0,00001, Dropout keepprob = 0,907 (to samo dla wszystkich warstw Dropout), BatchNorm, augmentacja: nie.

**Trening**
- Optymalizator: Adam, Beta1 = 0,9895, Beta2 = 0,999, eps = 1e-8.
- Learning rate: stały, lr = 0,00057.
- Epoki: 8. Batch size: 64. Dane: pełny train (55 000), float32.
- Seed: 850624 (wagi i kolejność danych).
- Czas całkowity: **0,4 min** (całe `fit`, z liczeniem train i dev po każdej epoce).

**Wyniki**
- Dokładność train (ostatnia epoka): 0,9328. Dokładność dev (ostatnia epoka): **0,9004**. Najlepszy dev 0,9004 w epoce 7.
- Luka train-dev: 3,2 pp.
- Tuning: run 0 z 20, dev 0,8594 po 3 epokach na 10 000 obrazach.

**Pliki**
- Wagi: `trained_models/mlp_deep_res.npz`. Konfiguracja, architektura jako kod i historia z `fit`: `trained_models/mlp_deep_res.json`. Wyniki tuningu: `outputs/tuning_mlp_deep_res.csv`.

## CNN, 2 bloki Conv-BN-ReLU-MaxPool (32, 64) (`conv`): dev 0,9118

**Model**
- Conv(32, f=3) → BN → ReLU → MaxPool → Dropout(0,991) → Conv(64, f=3) → BN → ReLU → MaxPool → Dropout(0,991) → Flatten → Dense(10).
- Warstwy Conv i Dense (poza ostatnią) bez biasu, po nich BatchNorm i ReLU; ostatni Dense z biasem i inicjalizacją Xavier, logity.
- Łączna liczba parametrów: **50 282**.
- Regularyzacja: L2 λ = 0,00001, Dropout keepprob = 0,991 (to samo dla wszystkich warstw Dropout), BatchNorm, augmentacja: nie.

**Trening**
- Optymalizator: Adam, Beta1 = 0,8671, Beta2 = 0,999, eps = 1e-8.
- Learning rate: stały, lr = 0,00882.
- Epoki: 8. Batch size: 64. Dane: pełny train (55 000), float32.
- Seed: 970323 (wagi i kolejność danych).
- Czas całkowity: **4,7 min** (całe `fit`, z liczeniem train i dev po każdej epoce).

**Wyniki**
- Dokładność train (ostatnia epoka): 0,9410. Dokładność dev (ostatnia epoka): **0,9118**. Najlepszy dev 0,9178 w epoce 5.
- Luka train-dev: 2,9 pp.
- Tuning: run 18 z 20, dev 0,8808 po 3 epokach na 10 000 obrazach.

**Pliki**
- Wagi: `trained_models/conv.npz`. Konfiguracja, architektura jako kod i historia z `fit`: `trained_models/conv.json`. Wyniki tuningu: `outputs/tuning_conv.csv`.

## CNN głęboki, 3 etapy po 2 konwolucje (32, 64, 128) (`conv_deep`): dev 0,9262

**Model**
- Conv(32, f=3) → BN → ReLU → Conv(32, f=3) → BN → ReLU → MaxPool → Dropout(0,911) → Conv(64, f=3) → BN → ReLU → Conv(64, f=3) → BN → ReLU → MaxPool → Dropout(0,911) → Conv(128, f=3) → BN → ReLU → Conv(128, f=3) → BN → ReLU → MaxPool → Dropout(0,911) → Flatten → Dense(10).
- Warstwy Conv i Dense (poza ostatnią) bez biasu, po nich BatchNorm i ReLU; ostatni Dense z biasem i inicjalizacją Xavier, logity.
- Łączna liczba parametrów: **298 410**.
- Regularyzacja: L2 λ = 0,07772, Dropout keepprob = 0,911 (to samo dla wszystkich warstw Dropout), BatchNorm, augmentacja: nie.

**Trening**
- Optymalizator: Adam, Beta1 = 0,8562, Beta2 = 0,999, eps = 1e-8.
- Learning rate: stały, lr = 0,00036.
- Epoki: 8. Batch size: 64. Dane: pełny train (55 000), float32.
- Seed: 827374 (wagi i kolejność danych).
- Czas całkowity: **13,6 min** (całe `fit`, z liczeniem train i dev po każdej epoce).

**Wyniki**
- Dokładność train (ostatnia epoka): 0,9522. Dokładność dev (ostatnia epoka): **0,9262**. Najlepszy dev 0,9262 w epoce 7.
- Luka train-dev: 2,6 pp.
- Tuning: run 19 z 20, dev 0,8838 po 3 epokach na 10 000 obrazach.
- Test (jednorazowo): **0,9271**, strata 0,2045.

**Pliki**
- Wagi: `trained_models/conv_deep.npz`. Konfiguracja, architektura jako kod i historia z `fit`: `trained_models/conv_deep.json`. Wyniki tuningu: `outputs/tuning_conv_deep.csv`.

## CNN głęboki z blokami residualnymi (32, 64, 128) (`conv_deep_res`): dev 0,9198

**Model**
- Conv(32, f=3) → BN → ReLU → ResidualBlock[Conv(32, f=3)-BN-ReLU-Conv(32, f=3)-BN] → ReLU → MaxPool → Dropout(0,968) → ResidualBlock[Conv(64, f=3)-BN-ReLU-Conv(64, f=3)-BN] + projekcja → ReLU → MaxPool → Dropout(0,968) → ResidualBlock[Conv(128, f=3)-BN-ReLU-Conv(128, f=3)-BN] + projekcja → ReLU → MaxPool → Dropout(0,968) → Flatten → Dense(10).
- Warstwy Conv i Dense (poza ostatnią) bez biasu, po nich BatchNorm i ReLU; ostatni Dense z biasem i inicjalizacją Xavier, logity.
- Łączna liczba parametrów: **318 122**.
- Regularyzacja: L2 λ = 0,00828, Dropout keepprob = 0,968 (to samo dla wszystkich warstw Dropout), BatchNorm, augmentacja: nie.

**Trening**
- Optymalizator: Adam, Beta1 = 0,9490, Beta2 = 0,999, eps = 1e-8.
- Learning rate: stały, lr = 0,00215.
- Epoki: 8. Batch size: 64. Dane: pełny train (55 000), float32.
- Seed: 649415 (wagi i kolejność danych).
- Czas całkowity: **18,6 min** (całe `fit`, z liczeniem train i dev po każdej epoce).

**Wyniki**
- Dokładność train (ostatnia epoka): 0,9499. Dokładność dev (ostatnia epoka): **0,9198**. Najlepszy dev 0,9244 w epoce 6.
- Luka train-dev: 3,0 pp.
- Tuning: run 1 z 20, dev 0,8722 po 3 epokach na 10 000 obrazach.

**Pliki**
- Wagi: `trained_models/conv_deep_res.npz`. Konfiguracja, architektura jako kod i historia z `fit`: `trained_models/conv_deep_res.json`. Wyniki tuningu: `outputs/tuning_conv_deep_res.csv`.
