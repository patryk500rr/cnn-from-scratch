# Plan: CNN od zera w numpy

Twoje założenia są w większości dobre, ale pięć rzeczy trzeba skorygować, zanim zaczniesz.

1. **„Minuty, nie godziny" wymaga wektoryzacji i podzbioru danych.** Wersja naiwna z pętlami służy tylko do zrozumienia i do gradient checku na malutkich tensorach. Nie trenuj na niej niczego.
2. **Vanishing gradients i zysku ze skip connections nie zobaczysz na płytkiej CNN.** Taka sieć z ReLU i inicjalizacją He po prostu działa. Dlatego eksperyment z zanikaniem robisz na głębokiej sieci gęstej (20–30 warstw, jeden forward i backward, sekundy). ResNet oceniasz po normach gradientów i koszcie treningowym, a nie po tym, czy pobił prostą CNN na dev secie, bo na tym zbiorze raczej nie pobije.
3. **Gradient checking musi powstać przed warstwą conv.** Piszesz go na sieci gęstej, a potem sprawdzasz nim każdą nową warstwę od razu po napisaniu.
4. **Zakres to realnie 40–55 godzin**, czyli 2–3 tygodnie po około 3 godziny dziennie. Etap 2 (conv) zajmie najwięcej czasu.
5. **Różnice między wariantami będą rzędu 1–3 punktów procentowych.** Potrzebujesz stałych seedów i świadomości szumu: przy dev secie 2000 przykładów błąd standardowy dokładności to około 0,7 pp.

## Zbiór danych

| Kandydat | Format | Ocena |
|---|---|---|
| **Fashion-MNIST** (wybór) | 60k + 10k, 28×28×1, 10 klas | Sieć gęsta daje ok. 87–89%, prosta CNN ok. 90–92%. Overfitting łatwo wywołać, a jeden kanał oznacza szybki trening. |
| CIFAR-10 | 50k + 10k, 32×32×3, 10 klas | Dużo trudniejszy (prosta CNN ok. 60–70%) i kilka razy wolniejszy w numpy. Dobry jako rozszerzenie po Etapie 6. |
| MNIST | 28×28×1 | Za łatwy: sieć gęsta ma 98% i różnic nie widać. Odrzucam. |

Do testów jednostkowych przyda się `sklearn.datasets.load_digits` (1797 obrazków 8×8), bo wszystko liczy się na nim w ułamku sekundy.

**Wczytanie Fashion-MNIST** (dwie drogi):
- **Pliki IDX:** cztery pliki `.gz` z repozytorium `github.com/zalandoresearch/fashion-mnist` (katalog `data/fashion`, ok. 30 MB). Czytasz je przez `gzip.open` i `np.frombuffer` z `dtype=uint8`. Nagłówek ma 16 bajtów dla obrazów i 8 dla etykiet.
- **Przez sklearn:** `sklearn.datasets.fetch_openml` z nazwą `"Fashion-MNIST"`.

Po pierwszym wczytaniu zapisz dane przez `np.savez_compressed` i dalej używaj `np.load`.

**Podział:**
- Z oryginalnych 60k wydziel raz, ze stałym seedem, 5k na dev. Reszta to train.
- Oryginalne 10k to test. Dotykasz go jeden raz, na samym końcu.
- Do codziennych eksperymentów używaj podzbioru 10k train i 2k dev.
- Do finałowych porównań użyj pełnego dev 5k i większego train.

## Organizacja projektu

```
cnn_from_scratch/
  data/                  # .npz (poza gitem)
  src/
    data.py              # wczytanie, split, normalizacja, mini-batche, augmentacja
    layers/              # dense, conv, pool, flatten, activations,
                         # batchnorm, dropout, residual
    losses.py            # softmax + cross-entropy, składnik L2
    init.py              # He, Xavier, "zła" inicjalizacja do eksperymentów
    model.py             # Sequential
    optim.py             # SGD, Momentum, RMSprop, Adam, harmonogramy LR
    gradcheck.py
    train.py             # pętla treningowa, historia, statystyki gradientów
  tests/                 # kształty, gradient checki, naiwna vs wektoryzowana
  experiments/           # jeden skrypt/notebook na eksperyment, wyniki do plików
```

**Interfejs warstwy** (wspólny dla wszystkich):
- `forward(x, training)` zwraca wyjście i zapisuje cache.
- `backward(dout)` przyjmuje ∂J/∂wyjście, zwraca ∂J/∂wejście i zapisuje gradienty parametrów.
- Słowniki `params` i `grads` mają te same klucze i kształty.

`Sequential` trzyma listę warstw. Optymalizator dostaje pary (parametr, gradient) i trzyma własny stan dla każdego parametru.

**Konwencje do ustalenia teraz:**
- Układ tensorów to channels-last `(m, n_H, n_W, n_C)`, jak u Ng.
- W warstwach gęstych przykłady są wierszami `(m, n)`. To odwrotnie niż w kursie 1 Ng, ale spójnie z conv.
- Czynnik 1/m siedzi tylko w gradiencie funkcji kosztu. Warstwy sumują po batchu i nigdy nie dzielą.
- `dtype` jest konfigurowalny: `float32` do treningu, `float64` do gradient checku.

---

# Etap 1 — Dane i fundamenty (5–7 h)

## Krok 1. Dane i normalizacja

**Cel:** pipeline danych (zagadnienia 4, 13).

**Teoria:** normalizacja wyrównuje skale cech, więc powierzchnia kosztu jest mniej wydłużona i można użyć większego α. Dla obrazów skalujesz do [0,1], a potem odejmujesz średnią i dzielisz przez odchylenie:

```
x̃ = (x/255 − μ_train) / σ_train
```

μ i σ liczysz wyłącznie na train i te same wartości stosujesz do dev i test.

**Co zrobić:**
- `load_fashion_mnist()` zwraca X `(m, 28, 28, 1)` i y `(m,)`.
- `split(...)` ze stałym seedem.
- `Normalizer` z metodami `fit` i `transform`.
- `one_hot(y, 10)`.
- `iterate_minibatches(X, y, batch_size, shuffle, rng)` jako generator.

**Pułapki:**
- Arytmetyka na `uint8` się przepełnia, więc najpierw rzutuj na float.
- Liczenie μ i σ na całym zbiorze to wyciek danych.
- Tasuj X i y tą samą permutacją.
- Ostatni mini-batch może być mniejszy.

**Jak sprawdzić:**
- Wyświetl kilka obrazków z etykietami.
- Średnia train po normalizacji ≈ 0, odchylenie ≈ 1.
- Rozkład klas w train i dev jest zbliżony.
- Suma rozmiarów mini-batchy równa się m.

**Pytania kontrolne:**
- Dlaczego statystyki normalizacji liczymy tylko na train?
- Jak wyglądają poziomice kosztu bez normalizacji i co to znaczy dla gradient descent?
- Dlaczego dev i test muszą pochodzić z tego samego rozkładu?

## Krok 2. Warstwy bazowe: Dense, ReLU, Flatten, Softmax + cross-entropy

**Cel:** przepisać wczorajszą sieć na architekturę modułową (zagadnienia 1, 2).

**Teoria:**

```
Dense:   Z = A·W + b        A:(m,n_in)  W:(n_in,n_out)  b:(n_out,)
         dW = Aᵀ·dZ         db = Σ_i dZ[i,:]         dA = dZ·Wᵀ
ReLU:    dZ = dA ⊙ 1[Z>0]
Softmax: ŷ_k = exp(z_k) / Σ_j exp(z_j)
Koszt:   J = −(1/m) Σ_i Σ_k y_ik · log ŷ_ik
```

Wyprowadzenie gradientu softmax z cross-entropy:

```
∂ŷ_k/∂z_j = ŷ_k(δ_kj − ŷ_j)
∂L/∂z_j   = −Σ_k (y_k/ŷ_k) · ŷ_k(δ_kj − ŷ_j) = ŷ_j − y_j      (bo Σ_k y_k = 1)
∂J/∂Z     = (Ŷ − Y)/m
```

**Co zrobić:**
- Klasy `Dense`, `ReLU`, `Sigmoid`, `Tanh` (przydadzą się w kroku 12) i `Flatten` (backward to reshape do zapamiętanego kształtu).
- `SoftmaxCrossEntropy` z `forward(Z, Y)` zwracającym J i `backward()` zwracającym dZ.
- `Sequential` z `forward`, `backward` i `parameters()`.

**Pułapki:**
- Softmax licz stabilnie: odejmij maksimum w wierszu przed `exp`, a logarytm licz metodą log-sum-exp, nie jako `log(softmax)`.
- Softmax i koszt połącz w jeden moduł. Rozdzielone dają niestabilny jakobian.
- Uważaj na broadcasting `b`. `db` musi mieć kształt `b`.

**Jak sprawdzić:**
- Asercje `grads[k].shape == params[k].shape` dla wszystkich parametrów.
- Wiersze softmaxu sumują się do 1.
- Dla losowych wag J ≈ ln 10 ≈ 2,30.

**Pytania kontrolne:**
- Skąd bierze się prosta postać ŷ − y?
- Dlaczego dW ma kształt W i z jakiej reguły to wynika?
- Co by się stało, gdyby 1/m było i w koszcie, i w warstwie?

## Krok 3. Gradient checking

**Cel:** narzędzie weryfikacji backpropu, używane do końca projektu (zagadnienie 6).

**Teoria:** przybliżenie dwustronne ma błąd O(ε²), jednostronne O(ε):

```
g_num[i] = (J(θ + ε·e_i) − J(θ − ε·e_i)) / (2ε)
błąd     = ‖g − g_num‖₂ / (‖g‖₂ + ‖g_num‖₂)
```

Przy ε ≈ 1e-5 i float64 błąd poniżej 1e-7 oznacza poprawny backprop. Okolice 1e-5 są podejrzane, a powyżej 1e-3 masz błąd w kodzie.

**Co zrobić:**
- `grad_check(model, loss, X, Y, eps, n_samples)`.
- Dla każdego parametru losuje do n_samples współrzędnych, zaburza je w miejscu, liczy J dwa razy i przywraca wartość.
- Zwraca błąd względny osobno dla każdego parametru, żeby było widać, która warstwa jest zła.
- Dodaj wariant sprawdzający gradient względem wejścia X. Testuje on ścieżkę `dA` w każdej warstwie.

**Pułapki:**
- float32 daje fałszywe alarmy. Używaj wyłącznie float64.
- Zapomniane przywrócenie zaburzonej wagi psuje kolejne pomiary.
- Koszt w checku musi zawierać wszystko, co gradient (później także L2).
- Losowość (dropout) trzeba wyłączyć albo zamrozić seedem.
- Załamania ReLU i remisy w max-poolu dają sporadyczne odchyłki.
- Używaj malutkich danych, np. m=4 i kilku neuronów.

**Jak sprawdzić:**
- Wszystkie parametry sieci Dense-ReLU-Dense mają błąd poniżej 1e-7.
- Celowo zepsuj backward (np. usuń transpozycję) i sprawdź, że check to wykrywa. Bez tego nie wiesz, czy check w ogóle działa.

**Pytania kontrolne:**
- Dlaczego różnica dwustronna jest dokładniejsza?
- Dlaczego błąd względny, a nie bezwzględny?
- Dlaczego nie używamy gradientu numerycznego do treningu?

## Krok 4. Mini-batch SGD i baseline

**Cel:** pętla treningowa i punkt odniesienia w postaci sieci gęstej (zagadnienia 7, 13).

**Teoria:**
- Batch GD daje gładki, ale drogi krok. SGD z m=1 jest zaszumione i nie korzysta z wektoryzacji. Mini-batch 32–128 to kompromis.
- Epoka to jedno przejście po danych, czyli ⌈m/batch⌉ kroków.
- Koszt na mini-batchach oscyluje i to jest normalne.

**Co zrobić:**
- Optymalizator `SGD` z metodą `step`.
- Funkcja `fit(model, loss, optimizer, data, epochs, batch_size)`.
- Historia zapisuje koszt każdego kroku oraz koszt i dokładność train i dev po każdej epoce.
- `evaluate` działa batchami w trybie `training=False`.
- Baseline: Flatten → Dense(128) → ReLU → Dense(10).

**Pułapki:**
- Ewaluacja całego zbioru jednym forwardem zje pamięć przy conv.
- Tasuj dane co epokę.
- Przy podzbiorze 10k dokładność train licz na tym samym podzbiorze.

**Jak sprawdzić:**
- Test przeuczenia na 32 przykładach: koszt spada niemal do zera, dokładność osiąga 100%. To najważniejszy test całej pętli.
- Baseline na 10k osiąga po kilku epokach około 83–87% na dev.

**Eksperyment:** batch size 1, 32, 256 i całe 10k przy tej samej liczbie epok. Porównaj krzywe kosztu i czas epoki.

**Pytania kontrolne:**
- Dlaczego koszt mini-batchowy oscyluje?
- Co się zmienia przy batch = m i batch = 1?
- Co oznacza sytuacja, w której model nie umie przeuczyć 32 przykładów?

---

# Etap 2 — Warstwy konwolucyjne (10–14 h)

## Krok 5. Conv forward, wersja naiwna

**Cel:** konwolucja i arytmetyka wymiarów (zagadnienia 2, 12).

**Teoria:**

```
n_H = ⌊(n_H_prev + 2p − f)/s⌋ + 1        (analogicznie n_W)
Z[i,h,w,c] = Σ_{a,b,k} A_pad[i, h·s+a, w·s+b, k] · W[a,b,k,c] + b[c]
```

Kształty: `W` to `(f, f, n_C_prev, n_C)`, `b` to `(n_C,)`. Padding „same" przy s=1 to p = (f−1)/2, „valid" to p = 0.

**Co zrobić:**
- `zero_pad(A, pad)` przez `np.pad`, tylko po osiach H i W.
- `conv_forward_naive(A_prev, W, b, stride, pad)` z pętlami po h i w (ewentualnie też po c). Wycinek `(m, f, f, n_C_prev)` mnożysz przez filtr i sumujesz po trzech osiach.
- Cache zawiera `A_prev`, `stride`, `pad`.
- Klasa `Conv2D` opakowuje funkcję.

**Pułapki:**
- Narożnik okna to `h·s`, nie `h`.
- Padowanie osi batcha lub kanałów to błąd.
- Wymiary niepodzielne przez stride po cichu ucinają brzeg. Dodaj asercję.

**Jak sprawdzić:**
- Kształty wyjścia dla kilku kombinacji (f, s, p) zgadzają się ze wzorem.
- Filtr z jedynką w środku przy „same" zwraca wejście.
- Ręcznie policzony przykład 4×4 z filtrem 2×2.
- Filtr Sobela na obrazku z Fashion-MNIST pokazuje krawędzie.

**Pytania kontrolne:**
- Skąd się bierze wzór na n_H?
- Ile parametrów ma warstwa i od czego ta liczba nie zależy?
- Po co padding (dwa powody)?

## Krok 6. Conv backward

**Cel:** najtrudniejszy gradient w projekcie (zagadnienia 1, 2).

**Teoria:** Z zależy od W liniowo, a każda waga `W[a,b,k,c]` bierze udział w każdej pozycji okna, więc jej gradient to suma po wszystkich pozycjach i przykładach:

```
dW[a,b,k,c]          = Σ_{i,h,w} A_pad[i, h·s+a, w·s+b, k] · dZ[i,h,w,c]
db[c]                = Σ_{i,h,w} dZ[i,h,w,c]
dA_pad[i,h·s+a,w·s+b,k] += Σ_c W[a,b,k,c] · dZ[i,h,w,c]     (dla każdego h, w, a, b)
dA_prev              = dA_pad bez ramki paddingu
```

W ostatnim wzorze stoi „+=", bo piksel wejścia należy do wielu nakładających się okien i zbiera gradient z każdego. Dla s=1 jest to pełna konwolucja dZ z filtrem obróconym o 180°.

**Co zrobić:**
- `conv_backward_naive(dZ, cache)` zwraca `dA_prev`, `dW`, `db`.
- Pętle są te same co w forwardzie. W każdej pozycji dodajesz wkład do `dW` i do wycinka `dA_pad`.

**Pułapki:**
- Użycie „=" zamiast „+=" przy `dA_pad`.
- Przy p=0 wycinek `[pad:-pad]` zwraca pustą tablicę. Obsłuż ten przypadek osobno.
- Sumowanie po złych osiach przy `dW`.

**Jak sprawdzić:**
- Gradient check dla W, b i wejścia na tensorze `(2, 5, 5, 2)` z f=3, w konfiguracjach (s=1, p=0), (s=1, p=1), (s=2, p=1).
- Błąd poniżej 1e-7 w każdej.
- Potem check całej sieci Conv → ReLU → Flatten → Dense → koszt.

**Pytania kontrolne:**
- Dlaczego w dW sumujemy po pozycjach, a w dA akumulujemy?
- Jak stride zmienia przepływ gradientu do wejścia?
- Jaki kształt ma każdy z trzech gradientów i dlaczego?

## Krok 7. Pooling

**Cel:** max i average pooling w obie strony (zagadnienia 1, 2).

**Teoria:**
- Wymiar wyjścia to n_H = ⌊(n_H_prev − f)/s⌋ + 1. Warstwa nie ma parametrów.
- Max pooling przekazuje gradient tylko do elementu maksymalnego w oknie (maska).
- Average pooling daje każdemu elementowi okna dZ/(f·f).
- Global average pooling uśrednia po całym H×W i daje `(m, n_C)`. Zastępuje Flatten z dużą warstwą gęstą.

**Co zrobić:**
- `MaxPool2D` i `AvgPool2D` z pętlami po h i w.
- `GlobalAvgPool`.
- W max-poolu zapamiętaj wejście albo maskę.

**Pułapki:**
- Przy remisie maska przez `==` puści gradient do kilku elementów naraz. W gradient checku używaj losowych danych float64.
- Przy s < f okna nachodzą na siebie, więc znów potrzebne jest „+=".

**Jak sprawdzić:**
- Kształty.
- Gradient check względem wejścia.
- Dla avg-pool suma `dA_prev` równa się sumie `dZ`.

**Pytania kontrolne:**
- Dlaczego pooling nie ma parametrów, a i tak wpływa na liczbę parametrów sieci?
- Co pooling daje poza zmniejszeniem wymiaru?
- Czym można go zastąpić?

## Krok 8. Wektoryzacja

**Cel:** przyspieszenie o 2–3 rzędy wielkości. Zrób ten krok dopiero wtedy, gdy wersja naiwna przechodzi gradient check, a przed pierwszym treningiem CNN.

**Teoria:** idea im2col to rozłożenie wszystkich okien na dodatkowe osie tensora. Konwolucja staje się wtedy jednym iloczynem tensorowym, czyli jednym wywołaniem BLAS.

**Co zrobić:**
- **Forward:** `np.lib.stride_tricks.sliding_window_view` po osiach H i W daje widok `(m, H', W', n_C_prev, f, f)`. Stride realizujesz krokiem `::s`. Wynik to `np.tensordot` lub `np.einsum` z W. Sprawdź kolejność osi widoku, bo osie okna lądują na końcu.
- **dW:** ten sam widok okien pomnożony przez dZ.
- **dA_pad:** najprostsza poprawna metoda to pętla tylko po f×f przesunięciach (a, b). Dla każdego dodajesz `dZ · W[a,b]ᵀ` do odpowiednio przesuniętego wycinka z krokiem s. To 9 iteracji dla filtra 3×3.
- **Pooling:** gdy s = f, wystarczy reshape do `(m, H/f, f, W/f, f, C)` i `max` lub `mean` po dwóch osiach.

**Pułapki:**
- `sliding_window_view` zwraca widok tylko do odczytu. Nie pisz do niego.
- `einsum` bez `optimize=True` bywa wolny. Porównaj z `tensordot`.
- Trenuj we float32.

**Jak sprawdzić:**
- `np.allclose` między wersją naiwną a wektoryzowaną dla forward i wszystkich gradientów, na kilku konfiguracjach.
- Gradient check wersji szybkiej.
- Pomiar czasu obu wersji na batchu 64.

**Pytania kontrolne:**
- Dlaczego jedno wywołanie numpy jest szybsze od pętli robiącej te same mnożenia?
- Ile pamięci zajęłaby faktyczna kopia macierzy im2col dla batcha 64×28×28×16 z f=3?
- Dlaczego zachowujemy wersję naiwną?

## Krok 9. Liczba parametrów i pierwsza CNN

**Cel:** zagadnienia 12 i 2.

**Teoria:**

```
conv:  (f·f·n_C_prev + 1)·n_C        dense: (n_in + 1)·n_out
```

- Parameter sharing i rzadkie połączenia sprawiają, że liczba parametrów conv nie zależy od rozmiaru obrazu.
- Dwie warstwy 3×3 mają pole recepcyjne jednej 5×5 przy 18 zamiast 25 wag na parę kanałów i dodatkowej nieliniowości.
- Konwolucja 1×1 to warstwa gęsta po kanałach, stosowana w każdym pikselu. Tanio zmienia liczbę kanałów.
- Przykład bottlenecku: 5×5 z 64 do 32 kanałów to 51 200 wag. Najpierw 1×1 do 16 kanałów, potem 5×5 do 32, to 1 024 + 12 800 = 13 824 wag.
- Stride i padding nie zmieniają liczby parametrów własnej warstwy. Zmieniają rozmiar map, więc koszt obliczeń i rozmiar pierwszej warstwy gęstej.

**Co zrobić:**
- `summary(model, input_shape)` wypisuje dla każdej warstwy kształt wyjścia i liczbę parametrów.
- Policz ręcznie i sprawdź funkcją tabelę:
  - Dense(128) na 784 wejściach,
  - conv 3×3 i 5×5 przy 8, 16, 32 kanałach,
  - wariant z 1×1,
  - Flatten + Dense kontra GlobalAvgPool + Dense.
- Zbuduj pierwszą CNN: Conv(8, 3×3, same) → ReLU → MaxPool(2) → Conv(16, 3×3, same) → ReLU → MaxPool(2) → Flatten → Dense(64) → ReLU → Dense(10).
- Na razie inicjalizuj małymi losowymi wagami. Krok 12 to poprawi.

**Pułapki:**
- Większość parametrów małej CNN siedzi w pierwszej warstwie gęstej po Flatten, tutaj 7·7·16·64.
- Koszt obliczeń jest odwrotnie: dominuje conv.

**Jak sprawdzić:**
- Gradient check całej CNN na 4 przykładach.
- Przeuczenie 32 przykładów.
- Trening na 10k powinien dać dev wyraźnie powyżej baseline'u przy mniejszej liczbie parametrów, orientacyjnie 87–90%.
- Epoka powinna trwać kilkanaście do kilkudziesięciu sekund. Jeśli trwa minuty, wróć do kroku 8.

**Pytania kontrolne:**
- Dlaczego CNN z mniejszą liczbą parametrów wygrywa z siecią gęstą na obrazach?
- Co robi konwolucja 1×1 i kiedy jej użyć?
- Jak stride w pierwszej warstwie wpływa na parametry i czas reszty sieci?

---

# Etap 3 — Optymalizacja (5–7 h)

## Krok 10. Exponentially weighted averages

**Cel:** zagadnienie 8, fundament pod momentum, Adama i batch norm.

**Teoria:**

```
v_t = β·v_{t−1} + (1−β)·θ_t,   v_0 = 0
v_t = (1−β)·Σ_{k=0}^{t−1} β^k · θ_{t−k}            (rozwinięcie rekurencji)
v̂_t = v_t / (1 − β^t)                              (bias correction)
```

- Średnia obejmuje w przybliżeniu ostatnie 1/(1−β) wartości, bo β^(1/(1−β)) ≈ 1/e.
- Start od zera zaniża początkowe wartości. Suma wag po t krokach to 1−β^t i przez nią dzielimy.

**Co zrobić:**
- Klasa lub funkcja `ewa(series, beta, bias_correction)`.
- Zastosuj ją do zapisanej w kroku 9 serii kosztów mini-batchowych.

**Eksperyment:**
- Jeden wykres: surowy koszt i EWA dla β = 0,5, 0,9, 0,98, z korekcją i bez.
- Zaobserwuj opóźnienie przy dużym β i zaniżony początek bez korekcji.

**Pułapki:**
- W korekcji t liczysz od 1.
- Korekcja dotyczy wartości odczytywanej, nie stanu v.

**Jak sprawdzić:**
- Dla stałej serii θ=c wersja z korekcją zwraca dokładnie c od pierwszego kroku.

**Pytania kontrolne:**
- Skąd się bierze „około 1/(1−β) kroków"?
- Kiedy bias correction ma znaczenie, a kiedy nie?
- Dlaczego EWA zamiast zwykłej średniej kroczącej?

## Krok 11. Momentum, RMSprop, Adam, learning rate decay

**Cel:** zagadnienie 9.

**Teoria** (g to gradient, wszystkie operacje element po elemencie):

```
Momentum:  v = β·v + (1−β)·g               θ −= α·v
RMSprop:   s = β₂·s + (1−β₂)·g²            θ −= α·g/(√s + ε)
Adam:      v = β₁·v + (1−β₁)·g      s = β₂·s + (1−β₂)·g²
           v̂ = v/(1−β₁ᵗ)            ŝ = s/(1−β₂ᵗ)
           θ −= α·v̂/(√ŝ + ε)               (β₁=0,9  β₂=0,999  ε=1e-8)
Decay:     α = α₀/(1 + k·epoka)   lub   α₀·γ^epoka   lub schodkowo
```

- Momentum uśrednia gradienty, więc oscylacje w poprzek doliny się znoszą, a ruch wzdłuż niej się sumuje.
- RMSprop dzieli krok przez typową wielkość gradientu danej współrzędnej, czyli daje każdemu parametrowi własną skalę.
- Adam łączy oba mechanizmy.

**Co zrobić:**
- Klasy `Momentum`, `RMSprop`, `Adam` ze wspólnym interfejsem `step`.
- Stan (v, s, t) jest osobny dla każdego parametru i ma kształt tego parametru.
- Harmonogram LR to osobny obiekt wywoływany co epokę.

**Pułapki:**
- Stan musi być przypisany do parametru po identyfikatorze, a nie po kolejności, która mogłaby się zmienić.
- W Adamie t rośnie raz na krok, nie raz na parametr.
- ε stoi poza pierwiastkiem.
- Porównanie przy wspólnym α jest nieuczciwe. SGD i momentum lubią α rzędu 0,01–0,1, a Adam i RMSprop rzędu 1e-3.

**Jak sprawdzić:**
- Momentum z β=0 daje identyczne wagi jak SGD.
- Ręcznie przelicz pierwszy krok Adama dla jednego skalara. Wynik to około α·sign(g).
- Przeuczenie 32 przykładów każdym optymalizatorem.

**Eksperyment:**
- Ta sama CNN, ten sam seed inicjalizacji i kolejności batchy, 5 epok.
- Warianty: SGD, Momentum, RMSprop, Adam. Dla każdego wybierz najlepsze α z 3–4 wartości na skali logarytmicznej.
- Narysuj wygładzony koszt treningowy (EWA z kroku 10) i dokładność dev.
- Osobno: Adam ze stałym α kontra Adam z decay. Popatrz na końcówkę treningu.

**Pytania kontrolne:**
- Dlaczego momentum pozwala na większe α?
- Co w Adamie robi dzielenie przez √ŝ i dlaczego pierwszy krok ma wielkość około α?
- Po co decay, skoro Adam „sam się skaluje"?

---

# Etap 4 — Stabilność gradientów i batch norm (7–9 h)

## Krok 12. Vanishing i exploding gradients, inicjalizacja, clipping

**Cel:** zagadnienie 5, z problemem wywołanym własnoręcznie.

**Teoria:** gradient w warstwie l to iloczyn jakobianów wszystkich warstw nad nią:

```
∂J/∂a^[l] = Π_{k>l} (W^[k]ᵀ · diag(g'(z^[k]))) · ∂J/∂a^[L]
```

- Jeśli czynniki mają typowo normę poniżej 1, iloczyn zanika wykładniczo z głębokością. Powyżej 1 wybucha.
- Sigmoid ma g' ≤ 1/4, a tanh g' ≤ 1. Nasycenie daje g' ≈ 0.
- ReLU ma g' ∈ {0, 1}, więc nie tłumi aktywnych ścieżek.

Inicjalizacja ma utrzymać wariancję sygnału:

```
Var(z^[l]) = n_in · Var(w) · E[(a^[l−1])²]
ReLU:       E[a²] = ½·Var(z)   ⇒   Var(w) = 2/n_in                       (He)
tanh:       Var(w) = 1/n_in   lub   2/(n_in + n_out)                      (Xavier)
conv:       n_in = f·f·n_C_prev
Clipping:   g ← g · min(1, c/‖g‖₂)        (norma liczona po wszystkich parametrach)
```

Clipping leczy tylko wybuchanie, nie zanikanie.

**Co zrobić:**
- W `init.py` funkcje `he`, `xavier`, `scaled_normal(scale)`.
- Wybór inicjalizacji jako argument warstw.
- W `train.py` funkcja zbierająca dla każdej warstwy ‖dW‖ oraz odchylenie standardowe aktywacji.
- Opcja `clip_norm` w optymalizatorze lub pętli.

**Eksperyment A** (zanikanie i wybuchanie, tani, bo to jeden forward i backward):
- Sieć gęsta: 20–30 warstw po 64 neurony.
- Aktywacje: sigmoid, tanh, ReLU.
- Inicjalizacje: N(0, 0,01²), N(0, 1), Xavier, He.
- Wykres ‖dW^[l]‖ w funkcji numeru warstwy na skali logarytmicznej, drugi dla odchylenia aktywacji.
- Oczekiwany wynik: przy sigmoidzie spadek o wiele rzędów wielkości w stronę wejścia, przy N(0,1) z ReLU wzrost i NaN.

**Eksperyment B:**
- Spróbuj trenować najgorszy i najlepszy wariant przez 2 epoki.
- Wariant wybuchający puść z clippingiem i bez.

**Eksperyment C:**
- Twoja CNN z kroku 9 z wagami He pomnożonymi przez 10 i przez 0,1.

**Pułapki:**
- Porównuj normy przy tej samej liczbie parametrów w warstwie albo normalizuj przez √(liczba elementów).
- He pasuje do ReLU, Xavier do tanh. Zamiana to częsty błąd.
- NaN w koszcie zwykle oznacza wybuch, nie błąd w softmaxie.

**Jak sprawdzić:**
- Empiryczna wariancja wag z `he` wynosi około 2/n_in.
- Przy He z ReLU odchylenie aktywacji jest w przybliżeniu stałe przez wszystkie 30 warstw.

**Pytania kontrolne:**
- Dlaczego zanikanie jest wykładnicze względem głębokości?
- Skąd czynnik 2 w He?
- Których problemów clipping nie rozwiąże?

## Krok 13. Batch normalization

**Cel:** zagadnienie 10 i ponownie 5.

**Teoria (forward w trybie treningu).** N to liczba elementów, po których uśredniasz: m dla Dense oraz m·n_H·n_W dla conv, gdzie statystyki liczysz po osiach (0, 1, 2), osobno dla każdego kanału.

```
μ = (1/N)·Σ x        σ² = (1/N)·Σ (x−μ)²
x̂ = (x−μ)/√(σ²+ε)    y = γ·x̂ + β
running_μ ← mom·running_μ + (1−mom)·μ      (to samo dla σ²; jest to EWA z kroku 10)
```

W trybie testu x̂ = (x − running_μ)/√(running_σ² + ε). Warstwa jest wtedy stałym przekształceniem afinicznym i wynik dla przykładu nie zależy od reszty batcha.

**Backward.** μ i σ² zależą od wszystkich x w batchu, więc gradient płynie trzema ścieżkami:

```
dγ = Σ dy·x̂        dβ = Σ dy        dx̂ = dy·γ
dσ² = Σ dx̂·(x−μ) · (−½)·(σ²+ε)^(−3/2)
dμ  = −Σ dx̂/√(σ²+ε)  +  dσ²·(−2/N)·Σ(x−μ)        (drugi składnik = 0)
dx  = dx̂/√(σ²+ε) + dσ²·2(x−μ)/N + dμ/N
```

Po uproszczeniu:

```
dx = (N·dx̂ − Σdx̂ − x̂·Σ(dx̂·x̂)) / (N·√(σ²+ε))
```

Zaimplementuj wersję krokową, a zwartą wyprowadź na kartce i użyj jako drugiej implementacji do porównania.

**Co zrobić:**
- Klasa `BatchNorm` z parametrem `axes`, czyli jedna klasa dla Dense i conv.
- γ i β mają kształt `(n_C,)`, γ startuje od 1, β od 0.
- Bufory running nie są parametrami i optymalizator ich nie dotyka.
- Kolejność w sieci: Conv → BN → ReLU.

**Pułapki:**
- Bias w warstwie przed BN jest zbędny, bo odejmowanie μ go kasuje. Wyłącz go.
- Statystyki batcha użyte przy ewaluacji dają wyniki zależne od składu batcha.
- Zapomniany `training=False` przy ewaluacji.
- L2 nie stosuj do γ i β.
- Mały batch (poniżej 8) daje zaszumione statystyki.

**Jak sprawdzić:**
- W trybie treningu z γ=1 i β=0 wyjście ma średnią ≈ 0 i odchylenie ≈ 1 na kanał.
- Gradient check w trybie treningu (float64) dla γ, β i wejścia.
- Σ dx po osiach normalizacji ≈ 0. Zastanów się, dlaczego tak musi być.
- Wersja krokowa i zwarta dają to samo.
- Po treningu dokładność w trybie testu jest bliska tej w trybie treningu.

**Eksperymenty:**
- Powtórz eksperyment A z kroku 12 z BN po każdej warstwie.
- CNN z BN i bez: zbieżność przy α normalnym i 10× większym.
- Batch size 4 kontra 64 z BN.

**Pytania kontrolne:**
- Dlaczego potrzebne są γ i β, skoro dopiero co znormalizowaliśmy?
- Dlaczego Σ dx = 0?
- Co by się stało na teście z batchem o rozmiarze 1 bez running averages?

---

# Etap 5 — Regularyzacja, diagnostyka, tuning (6–8 h)

## Krok 14. Bias i variance, L2, dropout, augmentacja

**Cel:** zagadnienia 3 i 13.

**Teoria.** Diagnoza idzie w tej kolejności:
- Wysoki błąd train oznacza bias. Lekarstwo: większa sieć, dłuższy trening, lepszy optymalizator.
- Duża luka między train a dev oznacza variance. Lekarstwo: więcej danych, regularyzacja, augmentacja.
- Punktem odniesienia jest najlepszy osiągalny wynik. Dla Fashion-MNIST dobre sieci mają około 93–96%.

```
L2:       J_reg = J + (λ/2m)·Σ_l ‖W^[l]‖²        dW += (λ/m)·W
Dropout:  D = (rand < keep_prob)     A ← A⊙D / keep_prob
          backward: dA ← dA⊙D / keep_prob          (test: warstwa jest identycznością)
```

- Tutaj m to rozmiar mini-batcha. L2 nie obejmuje b, γ, β.
- Inverted dropout dzieli przez keep_prob, żeby wartość oczekiwana aktywacji była ta sama w treningu i teście.
- Augmentacja dla ubrań: odbicie poziome, losowe przesunięcie o ±2 piksele z dopełnieniem zerami. Losujesz ją na bieżąco dla każdego mini-batcha, tylko na train.

**Co zrobić:**
- Parametr `l2` w koszcie i gradientach.
- Klasa `Dropout` z własnym generatorem losowym.
- `augment(X_batch, rng)` wpięte w generator mini-batchy.
- `plot_learning_curves(history)`.

**Pułapki:**
- Dodanie L2 do gradientu, ale nie do kosztu (lub odwrotnie). Gradient check to wykryje.
- Inna maska dropout w backward niż w forward.
- Dropout włączony przy ewaluacji.
- Koszt z dropoutem nie maleje monotonicznie, więc debuguj z keep_prob = 1.
- Augmentacja po normalizacji wymaga dopełnienia wartością „tła po normalizacji", nie zerem.

**Jak sprawdzić:**
- Gradient check z λ > 0.
- Gradient check z dropoutem przy zamrożonym seedzie maski.
- Średnia aktywacji po dropoucie w treningu ≈ średnia bez niego.

**Eksperyment:**
1. Wywołaj overfitting: 2 000 przykładów train, szersza CNN, 30 epok. Powinno wyjść train blisko 100% i dev wyraźnie niżej.
2. Dodawaj po kolei: L2 (λ z 3 wartości na skali logarytmicznej), dropout przed warstwą gęstą (keep_prob 0,5 i 0,8), augmentację, wreszcie 10k danych.
3. Zrób tabelę z dokładnością train, dev i luką.
4. Narysuj histogram wag z L2 i bez.

**Pytania kontrolne:**
- Dlaczego L2 nazywa się „weight decay" (rozpisz krok aktualizacji)?
- Dlaczego dzielimy przez keep_prob?
- Masz train 99% i dev 88%. Co robisz najpierw, a czego nie robisz?

## Krok 15. Tuning hiperparametrów

**Cel:** zagadnienie 11.

**Teoria.** Kolejność ważności:
1. α.
2. β momentum, rozmiar mini-batcha, liczba filtrów i neuronów.
3. Liczba warstw i decay.
4. β₁, β₂, ε prawie nigdy.

Random search bije grid, bo przy n próbach testuje n różnych wartości każdego hiperparametru, a grid tylko √n. Zwykle liczą się jeden lub dwa z nich.

Skale:

```
α = 10^r,       r ~ U(−4, −1)
λ = 10^r,       r ~ U(−5, −2)
1−β = 10^r,     r ~ U(−3, −1)
keep_prob, liczba warstw: skala liniowa
```

Strategia coarse-to-fine: szeroki zakres, potem zawężenie wokół najlepszych. Wybierasz na dev, nigdy na test.

**Co zrobić:**
- `sample_config(rng)`.
- `run_experiment(config)` zwraca metryki i zapisuje wynik do pliku CSV lub JSON od razu po każdym przebiegu.

**Eksperyment:**
- Random search po (α, λ, keep_prob, liczba filtrów): 15–20 przebiegów po 3 epoki na 10k.
- Wykres punktowy dev accuracy względem α na osi logarytmicznej.
- Druga runda: zawężony zakres, 8–10 przebiegów.
- Dla porównania grid 3×3 po (α, λ) przy tym samym budżecie.
- Najlepszą konfigurację puść z 3 seedami, żeby zobaczyć szum.

**Pułapki:**
- Losowanie α liniowo z [0,0001, 0,1] daje 90% prób w ostatniej dekadzie.
- 3 epoki faworyzują duże α, więc finalistów sprawdź dłużej.
- Różnice poniżej szumu między seedami nic nie znaczą.
- Całość zajmie 1–2 godziny obliczeń. Puść w tle.

**Jak sprawdzić:**
- Histogram wylosowanych log₁₀(α) jest płaski.
- Wyniki są odtwarzalne z zapisanego configu i seeda.

**Pytania kontrolne:**
- Dlaczego skala logarytmiczna dla α i dlaczego losujemy 1−β, a nie β?
- Kiedy grid search ma sens?
- Dlaczego wielokrotny wybór na dev „zużywa" dev set?

---

# Etap 6 — ResNet i projektowanie architektury (8–10 h)

## Krok 16. Blok rezydualny

**Cel:** zagadnienie 14 oraz 5 i 2.

**Teoria:**

```
a_out = g(F(a_in) + a_in)      F = Conv → BN → ReLU → Conv → BN
∂a_out/∂a_in = g'(·) · (∂F/∂a_in + I)
```

- Składnik I daje gradientowi drogę omijającą wagi. Iloczyn z kroku 12 nie może zaniknąć do zera przez samo mnożenie małych jakobianów.
- Blokowi łatwo nauczyć się identyczności: wystarczy F ≈ 0, do czego L2 samo popycha wagi. Dodanie bloku nie powinno więc pogorszyć kosztu treningowego.
- Backward przez węzeł sumy: gradient kopiuje się do obu gałęzi, a na wejściu bloku oba się sumują:

```
dS = dA_out ⊙ g'(S)                 S = F(a_in) + shortcut(a_in)
dA_in = backward_F(dS) + backward_shortcut(dS)
```

- Gdy wymiary się nie zgadzają (zmiana liczby kanałów lub stride w gałęzi głównej), skrót staje się projekcją: Conv 1×1 z tym samym stride i docelową liczbą kanałów, opcjonalnie z BN:

```
a_out = g(F(a_in) + W_s * a_in)
```

- Alternatywa bez parametrów to podpróbkowanie i dopełnienie kanałów zerami.

**Co zrobić:**
- Klasa `ResidualBlock(main_path, shortcut=None)`. Sama jest warstwą, zawiera listy podwarstw i wystawia ich parametry oraz gradienty z unikalnymi nazwami.
- Fabryki `identity_block(n_C)` i `conv_block(n_C_in, n_C_out, stride)`.
- Mała sieć: Conv(16) → 2–3 etapy po 1–2 bloki z 16, 32, 64 kanałami (stride 2 między etapami) → GlobalAvgPool → Dense(10).

**Pułapki:**
- Wejście bloku jest potrzebne obu gałęziom. Żadna warstwa nie może modyfikować tensorów w miejscu.
- Zapomniane dodanie gradientu skrótu w backward. Gradient check względem wejścia to wykryje.
- ReLU stoi po sumie, nie przed nią.
- Asercja zgodności kształtów F(a) i skrótu powinna mieć czytelny komunikat.
- Optymalizator musi widzieć parametry zagnieżdżone.

**Jak sprawdzić:**
- Gradient check `identity_block` i `conv_block` dla parametrów i wejścia. Przy stride 2 użyj wejścia 8×8.
- Po wyzerowaniu wag ostatniej konwolucji gałęzi, bez BN, blok zwraca ReLU(wejście).
- Przeuczenie 32 przykładów całą siecią.

**Eksperyment:**
- Dwie sieci o tej samej głębokości (np. 6 bloków, czyli około 13 warstw conv) i szerokości 8–16 kanałów: zwykła i ze skrótami.
- Każda w wersji bez BN i z BN.
- Zmierz ‖dW‖ dla każdej warstwy w iteracji 0 oraz koszt treningowy po 2–3 epokach na 5k przykładów.
- Bez BN różnica powinna być wyraźna. Z BN i He przy tej głębokości będzie mała. Oryginalny efekt degradacji pojawia się przy 30+ warstwach.
- Jeśli chcesz go zobaczyć, zwiększ głębokość przy zmniejszonej rozdzielczości (stride 2 w pierwszej warstwie) i pogódź się z dłuższym treningiem.

**Pytania kontrolne:**
- Dlaczego „+I" w jakobianie pomaga?
- Dlaczego głębsza zwykła sieć może mieć gorszy błąd treningowy i dlaczego to nie jest overfitting?
- Kiedy potrzebna jest projekcja 1×1 i ile dodaje parametrów?

## Krok 17. Eksperymenty z architekturą i finał

**Cel:** zagadnienia 13 i 12 oraz synteza.

**Teoria:**
- Typowy wzorzec: w głąb sieci H i W maleją, a liczba kanałów rośnie.
- Głębokość i szerokość zwiększasz, gdy masz bias. Gdy masz variance, regularyzujesz zamiast zmniejszać sieć.
- Jedna zmiana naraz, reszta zamrożona, ten sam seed.

**Eksperymenty** (każdy: 10k train, najlepszy optymalizator z kroku 11, stała liczba epok). Tabela: liczba parametrów, dokładność train, dev, czas epoki.
- Głębokość: 2, 3 i 5 warstw conv.
- Szerokość: 8/16, 16/32, 32/64 kanałów.
- Filtr 3×3 kontra 5×5, oraz dwa 3×3 kontra jeden 5×5.
- MaxPool kontra konwolucja ze stride 2.
- Flatten + Dense kontra GlobalAvgPool.
- Bottleneck 1×1 przed drogą konwolucją.

**Finał:**
- Wybierz model po dev.
- Wytrenuj na większym train (30–55k, może potrwać kilkadziesiąt minut).
- Narysuj krzywe uczenia.
- Raz policz dokładność na test.
- Zrób macierz pomyłek. Koszule, T-shirty i swetry będą się mylić.
- Obejrzyj kilkanaście błędnych przykładów (error analysis z kursu 3).

**Pułapki:**
- Porównywanie modeli po różnej liczbie epok lub z różnym α.
- Wyciąganie wniosków z różnic 0,5 pp.
- Wracanie do strojenia po obejrzeniu wyniku na test.

**Jak sprawdzić:**
- Wynik test jest bliski dev. Jeśli jest wyraźnie gorszy, przeuczyłeś się do dev.
- Rozsądny cel to 90–92% na test. Każdy punkt powyżej kosztuje dużo czasu CPU.

**Pytania kontrolne:**
- Który eksperyment dał najlepszy stosunek zysku do liczby parametrów i dlaczego?
- Po czym poznałeś, że model ma problem z bias, a po czym, że z variance?
- Co byś zmienił, przechodząc na CIFAR-10?

---

## Harmonogram

| Etap | Kroki | Czas |
|---|---|---|
| 1. Dane i fundamenty | 1–4 | 5–7 h |
| 2. Warstwy konwolucyjne | 5–9 | 10–14 h |
| 3. Optymalizacja | 10–11 | 5–7 h |
| 4. Stabilność gradientów i batch norm | 12–13 | 7–9 h |
| 5. Regularyzacja, diagnostyka, tuning | 14–15 | 6–8 h (plus obliczenia w tle) |
| 6. ResNet i architektura | 16–17 | 8–10 h |
| **Razem** | | **ok. 40–55 h** |

## Pokrycie zagadnień

| # | Zagadnienie | Kroki |
|---|---|---|
| 1 | Liczenie gradientów | 2, 6, 7, 13, 16 |
| 2 | Wymiary tensorów | 2, 5, 6, 7, 9, 16 |
| 3 | Regularyzacja | 14 (strojenie w 15) |
| 4 | Normalizacja wejścia | 1 |
| 5 | Vanishing i exploding gradients | 12, 13, 16 |
| 6 | Gradient checking | 3, potem w 6, 7, 8, 9, 13, 14, 16 |
| 7 | Mini-batch GD | 1, 4 |
| 8 | EWA i bias correction | 10 (użyte w 11, 13) |
| 9 | Optymalizatory i LR decay | 11 |
| 10 | Batch normalization | 13 |
| 11 | Tuning hiperparametrów | 15 |
| 12 | Liczba parametrów conv, 1×1 | 5, 9, 17 |
| 13 | Architektura, bias/variance, podział danych | 1, 4, 14, 17 |
| 14 | Skip connections | 16 |
