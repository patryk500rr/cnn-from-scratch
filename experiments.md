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
