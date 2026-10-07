# Importujemy moduł pyplot z biblioteki matplotlib i nadajemy mu skrót plt.
# Matplotlib służy do tworzenia wykresów i wizualizacji danych.
# Moduł pyplot zawiera funkcje potrzebne m.in. do tworzenia wykresów punktowych,
# ustawiania tytułów, opisów osi oraz wyświetlania gotowego wykresu.
import matplotlib.pyplot as plt

# Importujemy bibliotekę pandas i nadajemy jej skrót pd.
# Pandas służy do pracy z danymi tabelarycznymi, czyli np. tabelami podobnymi do arkuszy Excel.
# W tym skrypcie pandas jest dostępny do pracy z DataFrame, chociaż większość tabel
# otrzymujemy bezpośrednio ze zbioru Iris.
import pandas as pd

# Z modułu sklearn.datasets importujemy funkcję load_iris.
# Funkcja load_iris pozwala wczytać gotowy, wbudowany zbiór danych Iris.
# Dataset zawiera pomiary kwiatów należących do trzech gatunków irysa.
from sklearn.datasets import load_iris


# Wczytujemy zbiór danych Iris.
# Parametr as_frame=True powoduje, że dane zostaną zwrócone w formacie pandas.
# Dzięki temu możemy wygodnie odwoływać się do kolumn po ich nazwach.
zbior = load_iris(as_frame=True)


# Pobieramy pełną tabelę danych ze zbioru Iris.
# zbior.frame zawiera cztery cechy opisujące kwiat oraz kolumnę target z numerem klasy.
# Metoda copy() tworzy kopię tabeli, aby można było bezpiecznie dodawać nowe kolumny.
dane = zbior.frame.copy()


# Tworzymy nową kolumnę o nazwie "gatunek".
# dane["target"] zawiera numery klas: 0, 1 i 2.
# enumerate(zbior.target_names) tworzy pary numer-klasa, np. 0-setosa.
# dict(...) zamienia te pary na słownik, a map() zamienia numery klas
# na czytelne nazwy gatunków: setosa, versicolor i virginica.
dane["gatunek"] = dane["target"].map((dict(enumerate(zbior.target_names))))


# Pokazujemy początkowe rekordy, aby sprawdzić wygląd tabeli po dodaniu kolumny "gatunek".
print("Pierwsze wiersze:")

# Metoda head() domyślnie zwraca pierwszych 5 wierszy DataFrame.
# Dzięki temu możemy sprawdzić, czy nowa kolumna została poprawnie utworzona.
print(dane.head())


# Poniżej obliczamy średnie wartości cech osobno dla każdego gatunku irysa.
print("Średnie wartości cech w grupach:")

# groupby("gatunek") dzieli dane na grupy według nazwy gatunku.
# [zbior.feature_names] wybiera tylko cztery kolumny z cechami liczbowymi.
# mean() oblicza średnią każdej cechy osobno dla setosa, versicolor i virginica.
# Dzięki temu możemy porównać typowe wartości pomiarów między gatunkami.
print(dane.groupby("gatunek")[zbior.feature_names].mean())


# Tworzymy nową figurę, na której powstanie pierwszy wykres.
# figsize=(8, 6) określa rozmiar wykresu: 8 cali szerokości i 6 cali wysokości.
plt.figure(figsize=(8, 6))


# Pętla przechodzi kolejno po wszystkich nazwach gatunków ze zbioru Iris.
# Dla każdego gatunku narysujemy osobną grupę punktów na tym samym wykresie.
for gatunek in zbior.target_names:

    # Wybieramy z tabeli tylko te wiersze, które należą do aktualnego gatunku.
    # Dzięki temu każdy gatunek może zostać narysowany jako osobna grupa punktów.
    podzbior = dane[dane["gatunek"] == gatunek]

    # Tworzymy wykres rozrzutu dla aktualnego gatunku.
    # Każdy punkt reprezentuje jeden kwiat.
    plt.scatter(
        # Na osi X umieszczamy długość płatka kwiatu w centymetrach.
        podzbior["petal length (cm)"],

        # Na osi Y umieszczamy szerokość płatka kwiatu w centymetrach.
        podzbior["petal width (cm)"],

        # label=gatunek przypisuje aktualnej grupie nazwę gatunku.
        # Nazwa ta zostanie później pokazana w legendzie.
        label=gatunek,

        # alpha określa przezroczystość punktów.
        # Wartość 0.75 sprawia, że punkty są lekko przezroczyste,
        # co ułatwia zauważenie miejsc, w których nachodzą na siebie.
        alpha=0.75,
    )


# Ustawiamy opis osi poziomej, która przedstawia długość płatka.
plt.xlabel("Długość płatka")

# Ustawiamy opis osi pionowej, która przedstawia szerokość płatka.
plt.ylabel("Szerokość płatka")


# Nadajemy wykresowi tytuł opisujący porównywane cechy.
plt.title("Iris - długość i szerokość płatka")

# Dodajemy legendę, aby było wiadomo, które punkty należą do którego gatunku.
plt.legend()

# tight_layout() automatycznie poprawia rozmieszczenie elementów wykresu.
# Dzięki temu tytuł, opisy osi i legenda nie powinny na siebie nachodzić.
plt.tight_layout()

# Wyświetlamy przygotowany wykres.
plt.show()


# Tworzymy drugą, niezależną figurę do porównania cech działki kielicha.
plt.figure(figsize=(8, 6))


# Ponownie przechodzimy po wszystkich trzech gatunkach irysa.
for gatunek in zbior.target_names:

    # Dla aktualnego gatunku wybieramy tylko odpowiadające mu obserwacje.
    podzbior = dane[dane["gatunek"] == gatunek]

    # Tworzymy wykres rozrzutu dla długości i szerokości działki kielicha.
    plt.scatter(
        # Na osi X umieszczamy długość działki kielicha.
        podzbior["sepal length (cm)"],

        # Na osi Y umieszczamy szerokość działki kielicha.
        podzbior["sepal width (cm)"],

        # Nazwa gatunku będzie widoczna w legendzie.
        label=gatunek,

        # Ustawiamy częściową przezroczystość punktów.
        alpha=0.75,
    )


# Opisujemy oś X jako długość działki kielicha.
plt.xlabel("Długość działki kielicha")

# Opisujemy oś Y jako szerokość działki kielicha.
plt.ylabel("Szerokość działki kielicha")


# Ustawiamy tytuł drugiego wykresu.
plt.title("Iris - długość i szerokość działki kielicha")

# Dodajemy legendę z nazwami gatunków.
plt.legend()

# Automatycznie dopasowujemy marginesy i rozmieszczenie elementów.
plt.tight_layout()

# Wyświetlamy drugi wykres.
plt.show()


# Poniższe wnioski opisują, jak interpretować wykresy rozrzutu.
print("Interpretacja:")

# Wykres rozrzutu pozwala wizualnie ocenić, czy obserwacje różnych klas
# tworzą oddzielne grupy, czy mocno się ze sobą mieszają.
print("1. Wykresy rozrzutu pomagają zobaczyć, czy klasy da się rozdzielić.")

# W zbiorze Iris cechy związane z płatkami zwykle tworzą wyraźniejsze grupy
# niż cechy związane z działkami kielicha.
print("2. Cechy płatka zwykle lepiej rozdzielają gatunki niż cechy działki kielicha")

# Jeżeli klasy są dobrze rozdzielone na podstawie wybranych cech,
# model klasyfikacyjny może mieć łatwiejsze zadanie podczas przewidywania gatunku.
print("3. Jeśli klasy są dobrze rozdzielone na wykresie, prosty model może działać dobrze.")


# Zadanie 1 porównujemy długość działki kielicha z długością płatka.

# Tworzymy nową figurę dla dodatkowego wykresu.
plt.figure(figsize=(8, 6))


# Przechodzimy kolejno po wszystkich gatunkach irysa.
for gatunek in zbior.target_names:

    # Wybieramy tylko obserwacje należące do aktualnego gatunku.
    podzbior = dane[dane["gatunek"] == gatunek]

    # Tworzymy wykres rozrzutu dla dwóch innych cech niż wcześniej.
    plt.scatter(
        # Na osi X umieszczamy długość działki kielicha.
        podzbior["sepal length (cm)"],

        # Na osi Y umieszczamy długość płatka.
        podzbior["petal length (cm)"],

        # Przypisujemy grupie nazwę gatunku do wykorzystania w legendzie.
        label=gatunek,

        # Ustawiamy lekko przezroczyste punkty.
        alpha=0.75,
    )


# Ustawiamy opis osi poziomej.
plt.xlabel("Długość działki kielicha")

# Ustawiamy opis osi pionowej.
plt.ylabel("Długość płatka")


# Nadajemy wykresowi tytuł opisujący zestawione cechy.
plt.title("Iris - długość działki kielicha i długość płatka")

# Dodajemy legendę z nazwami gatunków.
plt.legend()

# Poprawiamy rozmieszczenie elementów na wykresie.
plt.tight_layout()

# Wyświetlamy gotowy wykres.
plt.show()
