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
