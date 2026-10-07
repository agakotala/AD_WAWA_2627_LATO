# Z modułu sklearn.datasets importujemy funkcję load_iris.
# Funkcja load_iris pozwala wczytać gotowy, wbudowany zbiór danych Iris.
# Jest to popularny dataset wykorzystywany do nauki analizy danych i klasyfikacji.
from sklearn.datasets import load_iris

# Z modułu sklearn.model_selection importujemy funkcję train_test_split.
# Funkcja ta służy do podziału danych na zbiór treningowy i testowy.
# Dzięki temu można później uczyć model na jednej części danych
# i sprawdzać jego działanie na danych, których wcześniej nie widział.
from sklearn.model_selection import train_test_split


# Wczytujemy wbudowany zbiór danych Iris.
# Parametr as_frame=True powoduje, że dane zostają zwrócone
# w formacie biblioteki pandas, czyli jako DataFrame i Series.
# Dzięki temu można korzystać z metod takich jak head(), tail(),
# describe(), mean() czy info().
zbior = load_iris(as_frame=True)


# Z obiektu zbior pobieramy pełną tabelę danych.
# zbior.frame zawiera zarówno cechy, jak i kolumnę target, czyli etykietę klasy.
# Metoda .copy() tworzy kopię danych, żeby uniknąć przypadkowej modyfikacji oryginalnego obiektu.
dane = zbior.frame.copy()


# Znak \n oznacza przejście do nowej linii,
# dzięki czemu wyniki w konsoli są bardziej czytelne.
print("\nCały zbiór danych:")

# Metoda to_string() zamienia cały DataFrame na reprezentację tekstową.
# Parametr index=False powoduje, że numery indeksów wierszy
# nie są wyświetlane obok danych.
# Dzięki temu można zobaczyć wszystkie obserwacje znajdujące się w zbiorze.
print(dane.to_string(index=False))


# zbior.DESCR zawiera tekstowy opis zbioru Iris.
# Można w nim znaleźć informacje m.in. o liczbie obserwacji,
# liczbie cech, nazwach cech, klasach oraz pochodzeniu danych.
print(zbior.DESCR)


print("Pierwsze 10 wierszy danych:")

# Metoda head(10) pobiera pierwszych 10 wierszy tabeli.
# Pozwala szybko sprawdzić strukturę zbioru,
# nazwy kolumn oraz przykładowe wartości.
print(dane.head(10))


print("\nNazwy kolumn:")

# Atrybut columns zawiera nazwy wszystkich kolumn w DataFrame.
# W zbiorze Iris znajdują się cztery kolumny z cechami
# oraz kolumna target zawierająca etykietę klasy.
print(dane.columns)


print("Informacje o danych:")

# Metoda info() pokazuje podstawowe informacje o strukturze DataFrame.
# Można dzięki niej sprawdzić:
# - liczbę wierszy,
# - liczbę kolumn,
# - nazwy kolumn,
# - typ danych w każdej kolumnie,
# - liczbę wartości niepustych.
# Pomaga to również sprawdzić, czy w zbiorze występują brakujące dane.
dane.info()


print("Podstawowe statystyki:")

# Metoda describe() oblicza podstawowe statystyki
# dla kolumn zawierających wartości liczbowe.
# Wynik zawiera między innymi:
# count - liczbę wartości,
# mean - średnią arytmetyczną,
# std - odchylenie standardowe,
# min - wartość minimalną,
# 25%, 50%, 75% - kwartyle,
# max - wartość maksymalną.
# Pozwala to szybko ocenić zakres i rozkład danych.
print(dane.describe())


print("Liczba obserwacji w każdej klasie:")

# dane["target"] wybiera kolumnę target,
# która zawiera numery gatunków: 0, 1 i 2.
# value_counts() zlicza, ile razy występuje każda wartość.
# sort_index() sortuje wynik według numerów klas.
# Dzięki temu można sprawdzić, ile obserwacji należy do każdej klasy.
print(dane["target"].value_counts().sort_index())


print("nazwy klas:")

# zbior.target_names zawiera tekstowe nazwy gatunków irysa.
# enumerate() pozwala jednocześnie pobrać numer klasy oraz jej nazwę.
# numer oznacza indeks klasy, czyli 0, 1 lub 2,
# a nazwa oznacza nazwę odpowiadającego jej gatunku.
for numer, nazwa in enumerate(zbior.target_names):

    # W wyniku otrzymujemy:
    # 0 = setosa
    # 1 = versicolor
    # 2 = virginica
    # Dzięki temu wiadomo, co oznaczają liczby zapisane w kolumnie target.
    print(numer, "=", nazwa)


# Tworzymy zmienną X zawierającą cechy wejściowe.
# zbior.data zawiera tylko cztery kolumny z pomiarami kwiatów,
# bez kolumny target.
# Są to informacje, na podstawie których model może rozpoznawać gatunek.
X = zbior.data


# Tworzymy zmienną y zawierającą etykiety klas.
# zbior.target przechowuje poprawny gatunek każdego kwiatu
# zapisany jako wartość 0, 1 albo 2.
# Jest to zmienna, którą model będzie próbował przewidywać.
y = zbior.target


# Atrybut shape zwraca rozmiar danych.
# Dla X pierwsza wartość oznacza liczbę obserwacji,
# a druga oznacza liczbę cech.
# W zbiorze Iris mamy 150 obserwacji i 4 cechy,
# dlatego wynik wynosi (150, 4).
print("Kształt X:", X.shape)


# y zawiera jedną etykietę dla każdej obserwacji.
# Ponieważ w zbiorze znajduje się 150 kwiatów,
# y zawiera 150 wartości.
# Wynik (150,) oznacza jednowymiarową serię 150 elementów.
print("Kształt y:", y.shape)


# Funkcja train_test_split dzieli dane na część treningową i testową.
# Zwraca cztery obiekty:
# X_train - cechy danych treningowych,
# X_test - cechy danych testowych,
# y_train - etykiety danych treningowych,
# y_test - etykiety danych testowych.
X_train, X_test, y_train, y_test = train_test_split(

    # X zawiera wszystkie cechy wejściowe.
    # Zostaną one podzielone na część treningową i testową.
    X,

    # y zawiera etykiety odpowiadające obserwacjom z X.
    # Muszą zostać podzielone w taki sam sposób,
    # aby każda obserwacja nadal miała przypisaną poprawną klasę.
    y,

    # test_size=0.3 oznacza, że 30% wszystkich danych
    # zostanie przeznaczone do zbioru testowego.
    # Pozostałe 70% danych zostanie przeznaczone do treningu.
    test_size=0.3,  # zadanie 3 - zmiana test_size z 0.25 na 0.3

    # random_state=42 ustala ziarno losowości.
    # Dzięki temu przy każdym uruchomieniu programu
    # otrzymamy dokładnie taki sam podział danych.
    # Ułatwia to powtarzanie i porównywanie wyników.
    random_state=42,

    # stratify=y powoduje zachowanie proporcji poszczególnych klas
    # w zbiorze treningowym i testowym.
    # Jest to ważne w klasyfikacji, ponieważ każda klasa
    # powinna być odpowiednio reprezentowana w obu częściach danych.
    stratify=y,
)


# X_train.shape pokazuje liczbę obserwacji treningowych
# oraz liczbę cech każdej obserwacji.
# 70% ze 150 obserwacji daje 105 obserwacji treningowych.
# Każda obserwacja posiada 4 cechy, dlatego wynik to (105, 4).
print("Rozmiar zbioru treningowego:", X_train.shape)


# X_test.shape pokazuje liczbę obserwacji testowych
# oraz liczbę cech każdej obserwacji.
# 30% ze 150 obserwacji daje 45 obserwacji testowych.
# Każda obserwacja posiada 4 cechy, dlatego wynik to (45, 4).
print("Rozmiar zbioru testowego:", X_test.shape)


print("Wnioski:")

# Obserwacja oznacza pojedynczy przykład znajdujący się w zbiorze danych.
# W tym przypadku jedną obserwacją jest jeden kwiat irysa.
print("1. Każdy wiersz to jedna obserwacja, czyli jeden kwiat.")


# Cechy X są informacjami wejściowymi opisującymi każdy kwiat.
# W zbiorze Iris są to cztery pomiary elementów kwiatu.
print("2. Kolumny z pomiarami to cechy X.")


# Kolumna target jest zmienną docelową.
# Określa prawidłowy gatunek danego kwiatu
# i jest wykorzystywana jako etykieta y.
print("3. Kolumna target to etykieta y, czyli gatunek irysa.")


# Podział danych pozwala wykorzystać jedną część do nauki modelu,
# a drugą część do późniejszego sprawdzenia jego działania
# na przykładach, których model wcześniej nie widział.
print("4. Dane dzielimy na treningowe i testowe, aby sprawdzić model na nowych przykładach.")


# Zadanie 1 polega na wyświetleniu ostatnich pięciu rekordów zbioru.
print("Ostatnie 5 wierszy:")

# Metoda tail() domyślnie zwraca ostatnich 5 wierszy DataFrame.
# Jest przeciwieństwem metody head(), która pobiera początkowe wiersze.
# Dzięki temu można szybko sprawdzić końcowe rekordy zbioru.
print(dane.tail())


# Zadanie 2 polega na obliczeniu średniej wartości każdej cechy.
print("Średnia wartość każdej cechy:")

# Metoda mean() oblicza średnią arytmetyczną
# osobno dla każdej kolumny znajdującej się w X.
# Ponieważ X zawiera cztery cechy, otrzymamy cztery średnie.
# Pozwala to określić przeciętną wartość każdego pomiaru w całym zbiorze.
print(X.mean())