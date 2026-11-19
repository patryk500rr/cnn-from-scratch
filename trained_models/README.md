# trained_models/ – wytrenowane modele (najlepsza konfiguracja z tuningu, pełny zbiór train)

Dla każdego z siedmiu modeli (`softmax_regression`, `mlp`, `mlp_deep`, `mlp_deep_res`, `conv`, `conv_deep`, `conv_deep_res`) są tu dwa pliki. Powstają w `src/main.ipynb` (sekcja „3. Trening końcowy”) przez `tuning.train_final`.

| Plik | Zawartość |
|---|---|
| `<model>.npz` | wagi, zapisane przez `Sequential.save` (W, b, gamma, beta, statystyki BN, wagi projekcji w resblockach) |
| `<model>.json` | konfiguracja hiperparametrów, architektura jako kod Pythona (`architecture`), liczba parametrów, czas treningu, końcowe metryki i pełna historia z `fit` (`history`: loss i accuracy na train i dev w każdej epoce) |

**Protokół:** pełny zbiór train (55 000), 8 epok, batch size 64, Adam ze stałym `lr`, float32. Dev (5 000) służy do wyboru. Zbiór testowy nie był użyty.

## Jak przywrócić model

1. Uruchom komórki z importami i z definicją `ARCHITECTURES` w `src/main.ipynb` (szablony warstw).
2. Wczytaj model razem z wagami:

```python
model = tuning.load_trained(ARCHITECTURES['conv_deep'], 'conv_deep')
J, acc = model.evaluate(SoftmaxCrossEntropy(), X_dev, Y_dev, batch_size=64)
```

`load_trained` odczytuje konfigurację z JSON-a, buduje model dokładnie jak przy treningu (`build_model`) i wywołuje `load_weights`. Szablon warstw musi być ten sam, co przy treningu. Pełny opis każdej architektury (kod warstw z konkretnymi wartościami) jest w `../architektury.md` i w polu `architecture` w JSON-ie.

Historię treningu (wynik `fit`) odczytasz przez `tuning.load_record('<model>')['history']`. Wyniki zbiorcze i opis modeli: `../experiments.md`.
