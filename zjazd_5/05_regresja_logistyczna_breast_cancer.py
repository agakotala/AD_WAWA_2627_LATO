# Importujemy bibliotekę numpy i nadajemy jej skrót np.
# NumPy służy do wykonywania obliczeń numerycznych i pracy z tablicami.
# W tym skrypcie użyjemy jej m.in. do porównania dwóch tablic
# oraz obliczenia funkcji wykładniczej dla współczynników modelu.
import numpy as np

# Importujemy bibliotekę pandas i nadajemy jej skrót pd.
# Pandas służy do pracy z danymi tabelarycznymi.
# Tutaj wykorzystamy ją do tworzenia tabeli z wynikami
# oraz obiektu Series zawierającego współczynniki modelu.
import pandas as pd

# Importujemy funkcję train_test_split.
# Służy ona do podziału danych na zbiór treningowy i testowy.
from sklearn.model_selection import train_test_split

# Importujemy StandardScaler.
# Standaryzuje on cechy, dzięki czemu wartości o różnych skalach
# zostają przeliczone do porównywalnego zakresu.
from sklearn.preprocessing import StandardScaler

# Importujemy funkcję load_breast_cancer.
# Pozwala ona wczytać wbudowany zbiór danych dotyczący diagnostyki raka piersi.
# Zbiór zawiera cechy liczbowe opisujące badane przypadki oraz dwie klasy.
from sklearn.datasets import load_breast_cancer

# Importujemy LogisticRegression.
# Regresja logistyczna jest modelem klasyfikacyjnym,
# który może przewidywać jedną z dwóch klas.
from sklearn.linear_model import LogisticRegression

# Importujemy Pipeline.
# Pipeline pozwala połączyć skalowanie danych i model
# w jeden spójny proces wykonywany zawsze w tej samej kolejności.
from sklearn.pipeline import Pipeline

# Importujemy accuracy_score.
# Accuracy mierzy udział poprawnych predykcji wśród wszystkich predykcji modelu.
from sklearn.metrics import accuracy_score


# Wczytujemy zbiór Breast Cancer Wisconsin w formacie pandas.
# Parametr as_frame=True powoduje, że cechy i etykiety są obiektami pandas.
zbior = load_breast_cancer(as_frame=True)


# X zawiera wszystkie cechy wejściowe opisujące obserwacje.
# Są to liczbowe parametry wyznaczone na podstawie obrazów komórek.
X = zbior.data

# y zawiera etykiety klas.
# W oryginalnym zbiorze sklearn:
# 0 oznacza nowotwór złośliwy (malignant),
# 1 oznacza nowotwór łagodny (benign).
y = zbior.target  #0 rak zlosliwy, 1 rak łagodny


# Sprawdzamy, jakie numery są przypisane do poszczególnych klas.
print("Nazwy klas:")

# zbior.target_names zawiera tekstowe nazwy klas.
# enumerate() pozwala jednocześnie pobrać numer klasy i jej nazwę.
for numer, nazwa in enumerate(zbior.target_names):
    # Dzięki temu otrzymujemy mapowanie numeru 0 i 1 na nazwę klasy.
    print(numer, "=", nazwa)


# Pokazujemy początkowe obserwacje cech wejściowych.
print("Pierwsze wiersze cech:")

# head() domyślnie zwraca pierwszych 5 wierszy DataFrame X.
# Pozwala szybko sprawdzić nazwy cech i przykładowe wartości.
print(X.head())


# Sprawdzamy liczebność obu klas w całym zbiorze.
print("Liczba obserwacji w klasach:")

# value_counts() liczy wystąpienia każdej klasy,
# a sort_index() układa wynik według numerów klas: 0, następnie 1.
# Pozwala to sprawdzić, czy klasy mają podobną liczebność.
print(y.value_counts().sort_index())


# Dzielimy dane na zbiór treningowy i testowy.
# Model nauczy się zależności na części treningowej,
# a część testowa posłuży do niezależnej oceny jego predykcji.
X_train, X_test, y_train, y_test = train_test_split(
    # Cechy wejściowe.
    X,

    # Odpowiadające im prawdziwe etykiety klas.
    y,

    # 25% danych przeznaczamy do testowania,
    # a pozostałe 75% do trenowania modelu.
    test_size=0.25,

    # Stałe ziarno losowości zapewnia taki sam podział przy kolejnych uruchomieniach.
    random_state=42,

    # stratify=y zachowuje proporcje klasy złośliwej i łagodnej
    # w zbiorze treningowym oraz testowym.
    stratify=y,
)


# Sprawdzamy rozmiar zbioru treningowego.
# shape zwraca liczbę obserwacji oraz liczbę cech.
print("Rozmiar X_train:", X_train.shape)

# Sprawdzamy rozmiar zbioru testowego.
print("Rozmiar X_test:", X_test.shape)


# Tworzymy pipeline składający się z dwóch kroków:
# skalowania cech oraz regresji logistycznej.
model = Pipeline(steps=[

    # StandardScaler oblicza średnią i odchylenie standardowe
    # dla każdej cechy na zbiorze treningowym, a następnie standaryzuje dane.
    ("skalowanie", StandardScaler()),

    # LogisticRegression jest właściwym modelem klasyfikacyjnym.
    # max_iter=1000 zwiększa maksymalną liczbę iteracji optymalizacji,
    # a random_state=42 pomaga zachować powtarzalność tam, gdzie występuje losowość.
    ("model", LogisticRegression(max_iter=1000, random_state=42)),
])


# Uczymy cały pipeline na danych treningowych.
# Najpierw dopasowywany jest StandardScaler,
# a następnie regresja logistyczna na przeskalowanych danych.
model.fit(X_train, y_train)


# predict() zwraca gotową przewidzianą klasę 0 lub 1 dla każdego przypadku w X_test.
y_pred = model.predict(X_test)


# Obliczamy accuracy przez porównanie prawdziwych klas y_test
# z klasami przewidzianymi przez model.
accuracy = accuracy_score(y_test, y_pred)

# Zaokrąglamy wynik do trzech miejsc po przecinku.
print("Accuracy:", round(accuracy, 3))


# predict_proba() zwraca prawdopodobieństwo przynależności
# każdej obserwacji do każdej z dwóch klas.
# Każdy wiersz wyniku zawiera dwie wartości, których suma wynosi 1.
prawdopodobienstwa = model.predict_proba(X_test)


# [:, 0] oznacza: wszystkie wiersze i pierwsza kolumna.
# Pobieramy więc prawdopodobieństwo klasy 0 dla każdej obserwacji testowej.
proba_klasa_0 = prawdopodobienstwa[:, 0]

# [:, 1] pobiera drugą kolumnę,
# czyli prawdopodobieństwo klasy 1 dla każdej obserwacji.
proba_klasa_1 = prawdopodobienstwa[:, 1]


# Tworzymy tabelę, która łączy prawdziwe klasy,
# prawdopodobieństwa obu klas oraz końcową predykcję modelu.
wyniki = pd.DataFrame({
    # Prawdziwa klasa ze zbioru testowego.
    "klasa_prawdziwa": y_test.values,

    # Prawdopodobieństwo, że dana obserwacja należy do klasy 0.
    "P(klasa=0)": proba_klasa_0,

    # Prawdopodobieństwo, że dana obserwacja należy do klasy 1.
    "P(klasa=1)": proba_klasa_1,

    # Klasa zwrócona bezpośrednio przez predict().
    "klasa_predict": y_pred,
})


# Tworzymy słownik pozwalający zamienić numery klas na nazwy tekstowe.
mapa_klas = {
    # Numerowi 0 przypisujemy pierwszą nazwę z target_names, czyli malignant.
    0: zbior.target_names[0],

    # Numerowi 1 przypisujemy drugą nazwę, czyli benign.
    1: zbior.target_names[1],
}


# Dodajemy kolumnę z tekstową nazwą prawdziwej klasy.
# map(mapa_klas) zamienia wartość 0 lub 1 na odpowiadającą jej nazwę.
wyniki["prawdziwa_nazwa"] = wyniki["klasa_prawdziwa"].map(mapa_klas)

# Dodajemy analogiczną nazwę dla klasy przewidzianej przez model.
wyniki["przewidziana_nazwa"] = wyniki["klasa_predict"].map(mapa_klas)


# Pokazujemy pierwszych 10 przypadków wraz z prawdopodobieństwami obu klas.
print("Pierwsze 10 predykcji i prawdopodobieństw:")
print(wyniki.head(10))


# Ręcznie tworzymy predykcję przy progu 0.5 dla klasy 1.
# proba_klasa_1 >= 0.5 tworzy wartości True/False.
# astype(int) zamienia False na 0, a True na 1.
# Jeśli prawdopodobieństwo klasy 1 wynosi co najmniej 0.5,
# przypisujemy obserwacji klasę 1, w przeciwnym razie klasę 0.
y_pred_prog_05 = (proba_klasa_1 >= 0.5).astype(int)


# Sprawdzamy, czy ręczne użycie progu 0.5 daje taki sam wynik jak predict().
print("Czy ręczny próg 0.5 daje te same klasy co predict()?")

# np.array_equal() zwraca True, jeśli dwie tablice mają identyczne elementy
# w tych samych miejscach, w przeciwnym razie zwraca False.
print(np.array_equal(y_pred, y_pred_prog_05))


# Tworzymy predykcje dla progu 0.3.
# Niższy próg ułatwia przypisanie obserwacji do klasy 1,
# ponieważ wystarczy mniejsze prawdopodobieństwo tej klasy.
y_pred_prog_03 = (proba_klasa_1 >= 0.3).astype(int)

# Tworzymy predykcje dla wyższego progu 0.7.
# Aby otrzymać klasę 1, model musi być bardziej pewny tej klasy.
y_pred_prog_07 = (proba_klasa_1 >= 0.7).astype(int)

# Tworzymy dodatkowy wariant dla progu 0.4.
y_pred_prog_04 = (proba_klasa_1 >= 0.4).astype(int)


# Obliczamy accuracy dla ręcznej klasyfikacji z progiem 0.3.
accuracy_03 = accuracy_score(y_test, y_pred_prog_03)

# Obliczamy accuracy dla ręcznej klasyfikacji z progiem 0.7.
accuracy_07 = accuracy_score(y_test, y_pred_prog_07)


# Porównujemy skuteczność klasyfikacji dla dwóch różnych progów decyzyjnych.
print("Accuracy dla progu 0.3:", round(accuracy_03, 3))
print("Accuracy dla progu 0.7:", round(accuracy_07, 3))


# Sprawdzamy, jak zmiana progu wpływa na liczbę obserwacji przypisanych do klasy 1.
print("Liczba przewidzianej klasy 1 dla różnych progów:")

# Tablice zawierają wartości 0 i 1, więc sum() zlicza liczbę jedynek.
# int() zamienia wynik na zwykłą liczbę całkowitą.
print("Próg 0.3:", int(y_pred_prog_03.sum()))
print("Próg 0.4:", int(y_pred_prog_04.sum()))
print("Próg 0.5:", int(y_pred_prog_05.sum()))
print("Próg 0.7:", int(y_pred_prog_07.sum()))


# Z pipeline pobieramy bezpośrednio wytrenowany krok regresji logistycznej.
# named_steps pozwala odwołać się do elementu pipeline po jego nazwie.
regresja_logistyczna = model.named_steps["model"]


# model.coef_ zawiera współczynniki regresji logistycznej.
# [0] wybiera zestaw współczynników dla klasy dodatniej w klasyfikacji binarnej.
# Tworzymy Series, aby każdy współczynnik miał jako indeks nazwę odpowiedniej cechy.
wspolczynniki = pd.Series(
    regresja_logistyczna.coef_[0],
    index=X.columns,
)


# Sortujemy współczynniki malejąco i pokazujemy 8 największych dodatnich wartości.
# Dodatni współczynnik zwiększa logarytm szans przypisania do klasy 1
# przy wzroście danej cechy, przy pozostałych warunkach modelu bez zmian.
print("Największe dodatnie współczynniki:")
print(wspolczynniki.sort_values(ascending=False).head(8))


# Sortujemy współczynniki rosnąco i pokazujemy 8 najbardziej ujemnych.
# Ujemny współczynnik działa w kierunku zmniejszenia szans klasy 1
# przy wzroście danej cechy, przy pozostałych warunkach bez zmian.
print("Najbardziej ujemne współczynniki:")
print(wspolczynniki.sort_values().head(8))


# Funkcja np.exp() podnosi liczbę e do potęgi każdego współczynnika.
# W regresji logistycznej w ten sposób otrzymujemy iloraz szans, czyli odds ratio.
# Wartość większa od 1 wskazuje wzrost szans klasy 1,
# a wartość mniejsza od 1 ich spadek dla wzrostu cechy o jednostkę w skali używanej przez model.
iloraz_szans = np.exp(wspolczynniki)

# Pokazujemy największe ilorazy szans po posortowaniu malejącym.
print("Przykładowe największe ilorazy szans:")
print(iloraz_szans.sort_values(ascending=False).head(8))


# Wyraz wolny, czyli intercept, jest stałą częścią równania regresji logistycznej.
print("Wyraz wolny:")

# intercept_ jest tablicą, dlatego [0] pobiera jej pierwszy element.
# round(..., 3) zaokrągla wartość do trzech miejsc po przecinku.
print(round(regresja_logistyczna.intercept_[0], 3))


print("Interpretacja:")

# Pomimo słowa "regresja" model służy tutaj do przewidywania klas.
print("1. Regresja logistyczna jest modelem klasyfikacyjnym, mimo słowa 'regresja' w nazwie.")

# predict() zwraca już wybraną klasę, a predict_proba() pokazuje stopień pewności modelu
# w postaci prawdopodobieństw przypisanych do poszczególnych klas.
print("2. predict() zwraca gotową klasę, natomiast predict_proba() zwraca prawdopodobieństwo klas.")

# Współczynniki opisują zależności wykorzystywane przez model,
# ale same w sobie nie dowodzą, że dana cecha powoduje zmianę klasy.
print("3. Sam współczynnik nie oznacza automatycznie związku przyczynowego między cechą na klasą.")
