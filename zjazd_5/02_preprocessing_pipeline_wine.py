# Importujemy bibliotekę numpy i nadajemy jej skrót np.
# NumPy służy do wykonywania obliczeń numerycznych i pracy z tablicami.
# W tym skrypcie użyjemy jej do losowania indeksów oraz tworzenia wartości NaN.
import numpy as np


# Importujemy ColumnTransformer.
# Pozwala on zastosować określone operacje przetwarzania tylko do wybranych kolumn.
# Jest szczególnie przydatny, gdy różne typy danych wymagają różnych transformacji.
from sklearn.compose import ColumnTransformer

# Importujemy funkcję load_wine.
# Pozwala ona wczytać wbudowany zbiór danych Wine,
# używany m.in. do zadań klasyfikacji różnych klas win.
from sklearn.datasets import load_wine

# Importujemy SimpleImputer.
# Służy on do uzupełniania brakujących wartości w danych.
# W tym skrypcie braki będą uzupełniane medianą danej cechy.
from sklearn.impute import SimpleImputer

# Importujemy model LogisticRegression.
# Regresja logistyczna jest modelem klasyfikacyjnym,
# który w tym przypadku będzie przewidywał klasę wina.
from sklearn.linear_model import LogisticRegression

# Importujemy accuracy_score.
# Ta metryka oblicza, jaki procent predykcji modelu jest poprawny.
from sklearn.metrics import accuracy_score

# Importujemy train_test_split.
# Funkcja służy do podziału danych na część treningową i testową.
from sklearn.model_selection import train_test_split

# Importujemy Pipeline.
# Pipeline pozwala połączyć kilka kolejnych etapów w jeden proces,
# np. uzupełnianie braków, skalowanie i trenowanie modelu.
from sklearn.pipeline import Pipeline

# Importujemy StandardScaler.
# StandardScaler standaryzuje cechy tak, aby miały średnią około 0
# i odchylenie standardowe około 1.
from sklearn.preprocessing import StandardScaler


# Wczytujemy zbiór danych Wine.
# Parametr as_frame=True powoduje zwrócenie danych w formacie pandas.
zbior = load_wine(as_frame=True)



# można wyświetlić pełny opis zbioru Wine,
# zawierający m.in. informacje o cechach, klasach i liczbie obserwacji.
#print(zbior.DESCR) #wyświetlanie charakterystyki zbioru danych


# Tworzymy zmienną X zawierającą wszystkie cechy wejściowe.
# Używamy copy(), ponieważ za chwilę celowo zmodyfikujemy dane,
# wprowadzając do nich brakujące wartości.
X = zbior.data.copy()

# Tworzymy zmienną y zawierającą etykiety klas.
# Są to poprawne odpowiedzi określające klasę każdej obserwacji.
y = zbior.target


# Tworzymy generator liczb losowych NumPy.
# seed=42 ustala ziarno losowości, dzięki czemu za każdym uruchomieniem
# zostaną wylosowane te same indeksy i otrzymamy powtarzalny eksperyment.
generator = np.random.default_rng(seed=42)


# Pętla przechodzi tylko po pierwszych czterech kolumnach X.
# X.columns[:4] oznacza wybór kolumn od pierwszej do czwartej.
# W tych kolumnach celowo utworzymy brakujące dane.
for kolumna in X.columns[:4]:

    # Losujemy 5 różnych indeksów wierszy z tabeli X.
    # size=5 oznacza, że losujemy pięć indeksów.
    # replace=False oznacza losowanie bez powtórzeń,
    # więc ten sam indeks nie zostanie wybrany dwa razy w jednej kolumnie.
    indeksy = generator.choice(X.index, size=5, replace=False)

    # W wybranych wierszach aktualnej kolumny wpisujemy np.nan.
    # np.nan oznacza brakującą wartość numeryczną.
    # Robimy to celowo, aby później przećwiczyć uzupełnianie braków danych.
    X.loc[indeksy, kolumna] = np.nan


# Pokazujemy fragment danych po sztucznym wprowadzeniu brakujących wartości.
print("Pierwsze wiersze z brakami:")

# head(15) zwraca pierwszych 15 wierszy tabeli.
# Pozwala sprawdzić, czy w danych pojawiły się wartości NaN.
print(X.head(15))


# Sprawdzamy, ile brakujących wartości występuje w każdej kolumnie.
print("Liczba braków w każdej kolumnie:")

# isna() zamienia każdą komórkę na wartość True, jeśli jest brakiem,
# lub False, jeśli zawiera prawidłową wartość.
# sum() zlicza wartości True osobno w każdej kolumnie,
# dzięki czemu otrzymujemy liczbę braków dla każdej cechy.
print(X.isna().sum())


# Dzielimy dane na zbiór treningowy i testowy.
# Zbiór treningowy posłuży do dopasowania preprocessingu i modelu,
# a zbiór testowy do późniejszej oceny działania na nowych danych.
X_train, X_test, y_train, y_test = train_test_split(
    # X zawiera cechy wejściowe, w tym utworzone wcześniej braki danych.
    X,

    # y zawiera poprawne etykiety klas.
    y,

    # 30% danych zostanie przeznaczone do testowania,
    # a pozostałe 70% do treningu.
    test_size=0.3,

    # Ustalamy stałe ziarno losowości,
    # aby podział był taki sam przy każdym uruchomieniu programu.
    random_state=42,

    # stratify=y zachowuje podobne proporcje klas
    # w zbiorze treningowym i testowym.
    stratify=y,
)


# Pobieramy nazwy wszystkich kolumn z X i zamieniamy je na zwykłą listę.
# Ponieważ wszystkie cechy w zbiorze Wine są liczbowe,
# lista ta będzie wykorzystana jako lista kolumn poddawanych preprocessingowi.
kolumny_liczbowe = X.columns.tolist()


# Tworzymy obiekt ColumnTransformer odpowiedzialny za przetwarzanie kolumn.
# W tym przypadku wszystkie kolumny liczbowe otrzymają ten sam zestaw operacji.
preprocessor = ColumnTransformer(transformers=[
    (
        # "num" jest nazwą tego transformera.
        # Nazwa może być później pomocna przy odwoływaniu się do elementów pipeline.
        "num",

        # Dla kolumn liczbowych tworzymy osobny Pipeline z dwoma krokami:
        # uzupełnianiem braków oraz skalowaniem cech.
        Pipeline(steps=[

            # SimpleImputer(strategy="median") zastępuje brakujące wartości
            # medianą obliczoną dla danej kolumny na danych treningowych.
            # Mediana jest odporna na wartości odstające i często sprawdza się
            # przy uzupełnianiu braków w danych liczbowych.
            ("imputer", SimpleImputer(strategy="median")),

            # StandardScaler() standaryzuje dane po uzupełnieniu braków.
            # Dzięki temu cechy o różnych skalach stają się bardziej porównywalne.
            ("scaler", StandardScaler()),
        ]),

        # Określamy, że powyższy pipeline ma zostać zastosowany
        # do wszystkich kolumn znajdujących się na liście kolumny_liczbowe.
        kolumny_liczbowe,
    )
])


# Tworzymy główny Pipeline łączący preprocessing z modelem klasyfikacyjnym.
# Dzięki temu podczas fit() najpierw zostaną przygotowane dane,
# a dopiero później na przetworzonych danych zostanie wytrenowany model.
model = Pipeline(steps=[

    # Pierwszym krokiem jest wcześniej utworzony preprocessor.
    ("preprocessing", preprocessor),

    # Drugim krokiem jest regresja logistyczna.
    # max_iter=2000 zwiększa maksymalną liczbę iteracji algorytmu uczącego,
    # a random_state=42 ustala powtarzalność tam, gdzie model korzysta z losowości.
    ("model", LogisticRegression(max_iter=2000, random_state=42)),
])


# Uczymy cały pipeline na danych treningowych.
# Pipeline najpierw oblicza mediany i parametry skalowania tylko na X_train,
# a następnie uczy regresję logistyczną na przetworzonych danych.
# Dzięki temu informacje ze zbioru testowego nie przeciekają do procesu treningu.
model.fit(X_train, y_train)

# Za pomocą wytrenowanego pipeline przewidujemy klasy dla X_test.
# Pipeline automatycznie wykonuje na danych testowych te same transformacje,
# a następnie przekazuje wynik do modelu regresji logistycznej.
y_pred = model.predict(X_test)


# Obliczamy accuracy, czyli udział poprawnie przewidzianych klas
# wśród wszystkich obserwacji ze zbioru testowego.
accuracy = accuracy_score(y_test, y_pred)


# round(accuracy, 3) zaokrągla wartość accuracy do trzech miejsc po przecinku.
print("ACCURACY:", round(accuracy, 3))


# Sprawdzamy nazwy cech po wykonaniu transformacji przez ColumnTransformer.
print("Nazwy kolumn po transformacji:")

# model.named_steps["preprocessing"] pobiera krok preprocessingu z głównego Pipeline.
# get_feature_names_out() zwraca nazwy cech po transformacji.
# Prefix "num__" informuje, że dana cecha została przetworzona przez transformer o nazwie "num".
print(model.named_steps["preprocessing"].get_feature_names_out())


print("Wnioski:")

# Uzupełnianie braków odbywa się wewnątrz pipeline po podziale danych,
# więc mediana jest wyznaczana na zbiorze treningowym, a nie na całym zbiorze.
print("1. Braki danych zostały uzupełnione medianą liczoną tylko na treningu.")

# Standaryzacja zmienia skalę cech, co pomaga modelom wrażliwym na różne zakresy wartości.
print("2. Cechy liczbowe zostały wystandaryzowane")

# Pipeline wymusza stałą kolejność operacji i stosuje te same transformacje
# zarówno podczas treningu, jak i podczas przewidywania nowych danych.
print("3. Pipeline pilnuje poprawnej kolejności: imputacja, skalowanie, model")
