# Importujemy bibliotekę pandas i nadajemy jej skrót pd.
# Pandas będzie używany do tworzenia tabel z wynikami predykcji i klasteryzacji.
import pandas as pd

# Importujemy funkcję load_iris do wczytania gotowego zbioru danych Iris.
from sklearn.datasets import load_iris

# Importujemy LogisticRegression, czyli model klasyfikacji nadzorowanej.
# Model będzie uczył się na cechach X oraz poprawnych etykietach y.
from sklearn.linear_model import LogisticRegression

# Importujemy KMeans.
# Jest to algorytm uczenia nienadzorowanego, który dzieli obserwacje na klastry.
from sklearn.cluster import KMeans

# Importujemy StandardScaler do standaryzacji cech.
# Skalowanie jest ważne szczególnie wtedy, gdy algorytm korzysta z odległości między obserwacjami.
from sklearn.preprocessing import StandardScaler

# Importujemy Pipeline.
# Pipeline pozwala połączyć skalowanie i model w jeden uporządkowany proces.
from sklearn.pipeline import Pipeline

# Importujemy funkcję do podziału danych na część treningową i testową.
from sklearn.model_selection import train_test_split

# Importujemy accuracy_score do oceny klasyfikacji nadzorowanej.
# Accuracy określa odsetek poprawnie przewidzianych klas.
from sklearn.metrics import accuracy_score

# Importujemy adjusted_rand_score, czyli ARI.
# ARI porównuje dwa podziały obserwacji na grupy, uwzględniając przypadkowe zgodności.
from sklearn.metrics import adjusted_rand_score

# Importujemy silhouette_score.
# Metryka silhouette ocenia jakość klastrów na podstawie ich zwartości i rozdzielenia.
from sklearn.metrics import silhouette_score

# Importujemy klasyfikator KNN.
# KNeighborsClassifier przypisuje klasę na podstawie klas najbliższych sąsiadów.
from sklearn.neighbors import KNeighborsClassifier


# Wczytujemy zbiór Iris w formacie pandas.
zbior = load_iris(as_frame=True)

# X zawiera cztery cechy opisujące kwiaty.
X = zbior.data

# y zawiera poprawne etykiety klas: 0, 1 lub 2.
y = zbior.target


# Pokazujemy pierwsze obserwacje cech wejściowych.
print("Pierwsze 5 cech X:")

# head() domyślnie zwraca pierwszych 5 wierszy tabeli X.
print(X.head())


# Pokazujemy pierwsze etykiety klas.
print("Pierwsze 5 etykiet y:")

# head() zwraca pierwszych 5 wartości zmiennej y.
print(y.head())


# X.shape pokazuje liczbę obserwacji i liczbę cech.
print("Kształt X:", X.shape)

# y.shape pokazuje liczbę etykiet.
print("Kształt y:", y.shape)

# zbior.target_names zawiera nazwy trzech gatunków.
# list(...) zamienia tablicę nazw na zwykłą listę Pythona.
print("Nazwy klas:", list(zbior.target_names))


# Dzielimy dane na część treningową i testową.
# Ten podział będzie wykorzystany przez modele uczenia nadzorowanego.
X_train, X_test, y_train, y_test = train_test_split(
    # Cechy wejściowe.
    X,

    # Poprawne etykiety klas.
    y,

    # 25% obserwacji trafia do testu, a 75% do treningu.
    test_size=0.25,

    # Stałe ziarno losowości zapewnia powtarzalny podział.
    random_state=42,

    # Zachowujemy proporcje klas w obu częściach danych.
    stratify=y,
)


# Sprawdzamy rozmiar danych treningowych.
print("Rozmiar X_train:", X_train.shape)

# Sprawdzamy rozmiar danych testowych.
print("Rozmiar X_test:", X_test.shape)


# Tworzymy pipeline dla pierwszego modelu nadzorowanego.
# Najpierw dane są skalowane, a następnie trafiają do regresji logistycznej.
model_nadzorowany = Pipeline(steps=[
    # StandardScaler ujednolica skalę wszystkich cech.
    ("skalowanie", StandardScaler()),

    # LogisticRegression uczy się przewidywać jedną z trzech klas Iris.
    # max_iter=2000 zwiększa limit iteracji potrzebnych do dopasowania modelu.
    ("model", LogisticRegression(max_iter=2000, random_state=42)),
])


# Uczymy pipeline na danych treningowych.
# W uczeniu nadzorowanym fit() otrzymuje zarówno X_train, jak i y_train.
model_nadzorowany.fit(X_train, y_train)


# Przewidujemy klasy dla obserwacji ze zbioru testowego.
y_pred = model_nadzorowany.predict(X_test)


# Obliczamy accuracy przez porównanie prawdziwych etykiet z przewidywaniami.
accuracy = accuracy_score(y_test, y_pred)


print("=== Uczenie nadzorowane ===")
print("Model: regresja logistyczna")
print("Podczas fit() użyto: X_train oraz y_train")

# round(..., 3) zaokrągla accuracy do trzech miejsc po przecinku.
print("Accuracy:", round(accuracy, 3))


# Tworzymy tabelę, w której zestawiamy prawdziwą i przewidzianą klasę.
wyniki_nadzorowane = pd.DataFrame({
    # y_test.values zawiera rzeczywiste etykiety ze zbioru testowego.
    "klasa_prawdziwa": y_test.values,

    # y_pred zawiera klasy przewidziane przez regresję logistyczną.
    "klasa_przewidziana": y_pred,
})


# Dodajemy kolumnę logiczną "poprawna".
# Dla każdej obserwacji sprawdzamy, czy klasa prawdziwa jest równa przewidzianej.
# Wynik True oznacza poprawną predykcję, a False błędną.
wyniki_nadzorowane["poprawna"] = (
    wyniki_nadzorowane["klasa_prawdziwa"] == wyniki_nadzorowane["klasa_przewidziana"]
)


# Pokazujemy pierwszych 10 wyników klasyfikacji nadzorowanej.
print("Pierwsze predykcje modelu nadzorowanego:")
print(wyniki_nadzorowane.head(10))


# Te dwie linie ponownie wyświetlają ten sam fragment tabeli.
# Kod pozostaje bez zmian względem oryginału, dlatego wynik pojawi się drugi raz.
print("Pierwsze predykcje modelu nadzorowanego:")
print(wyniki_nadzorowane.head(10))


# Tworzymy drugi model uczenia nadzorowanego - KNN.
# Tak jak wcześniej, najpierw skalujemy dane, a później uruchamiamy klasyfikator.
model_knn = Pipeline(steps=[
    # Standaryzacja jest szczególnie ważna dla KNN,
    # ponieważ model porównuje odległości między obserwacjami.
    ("skalowanie", StandardScaler()),

    # n_neighbors=5 oznacza, że podczas klasyfikacji
    # model bierze pod uwagę pięciu najbliższych sąsiadów.
    ("model", KNeighborsClassifier(n_neighbors=5)),
])


# Uczymy model KNN na tych samych danych treningowych.
model_knn.fit(X_train, y_train)


# Przewidujemy klasy dla zbioru testowego.
y_pred_knn = model_knn.predict(X_test)

# Obliczamy accuracy dla modelu KNN.
accuracy_knn = accuracy_score(y_test, y_pred_knn)


# Wyświetlamy podstawowe informacje i wynik drugiego modelu.
print("=== Uczenie nadzorowane KNN ===")
print("Model: KNN")
print("Liczba sąsiadów: 5")
print("Podczas fit() użyto: X_train oraz y_train")
print("Accuracy:", round(accuracy_knn, 3))


# Tworzymy tabelę porównującą prawdziwe etykiety z predykcjami KNN.
wyniki_knn = pd.DataFrame({
    "klasa_prawdziwa": y_test.values,
    "klasa_przewidziana": y_pred_knn,
})


# Dodajemy informację, czy dana predykcja KNN była poprawna.
wyniki_knn["poprawna"] = (
    wyniki_knn["klasa_prawdziwa"] == wyniki_knn["klasa_przewidziana"]
)


# Pokazujemy pierwszych 10 wyników modelu KNN.
print("Pierwsze predykcje modelu knn:")
print(wyniki_knn.head(10))


# Od tego miejsca przechodzimy do uczenia nienadzorowanego.
# W tej części algorytm nie dostaje etykiet y podczas tworzenia klastrów.
#nienadzorowane


# Tworzymy osobny obiekt StandardScaler.
# Będzie użyty do przeskalowania wszystkich cech przed KMeans.
skaler = StandardScaler()


# fit_transform(X) wykonuje dwie operacje naraz:
# fit() oblicza parametry skalowania na całym X,
# a transform() przekształca dane do ustandaryzowanej postaci.
# Wynik jest zapisany jako X_scaled.
X_scaled = skaler.fit_transform(X)


# Tworzymy model KMeans do grupowania obserwacji bez użycia etykiet y.
model_nienadzorowany = KMeans(
    # n_clusters=3 oznacza, że algorytm ma utworzyć trzy klastry.
    # Jest to ta sama liczba co liczba gatunków w Iris,
    # ale KMeans nie zna ich nazw ani prawdziwych etykiet.
    n_clusters=3,

    # random_state=42 zapewnia powtarzalność losowej inicjalizacji centroidów.
    random_state=42,

    # n_init=10 oznacza, że algorytm uruchomi proces klasteryzacji 10 razy
    # z różnymi inicjalizacjami i wybierze najlepszy wynik.
    n_init=10,
)


# Uczymy KMeans wyłącznie na przeskalowanych cechach X.
# Nie przekazujemy tutaj y, dlatego jest to uczenie nienadzorowane.
model_nienadzorowany.fit(X_scaled)


# labels_ zawiera numer klastra przypisany każdej obserwacji przez KMeans.
# Numery klastrów są umowne i nie muszą odpowiadać numerom prawdziwych klas.
klastry = model_nienadzorowany.labels_


# Pokazujemy pierwszych 20 numerów klastrów przypisanych przez KMeans.
print("Pierwsze 20 numerów klastrów:")
print(klastry[:20])


# Obliczamy silhouette score.
# Metryka ocenia, jak dobrze obserwacje pasują do własnego klastra
# w porównaniu z innymi klastrami.
# Wartości bliższe 1 oznaczają zwykle lepiej rozdzielone grupy.
silhouette = silhouette_score(X_scaled, klastry)


# Obliczamy ARI, porównując prawdziwe klasy y z klastrami KMeans.
# y nie było używane do treningu KMeans - służy tutaj wyłącznie do oceny wyniku po fakcie.
# ARI bliskie 1 oznacza dużą zgodność dwóch podziałów.
ari = adjusted_rand_score(y, klastry)


# Wyświetlamy podsumowanie uczenia nienadzorowanego.
print("=== Uczenie nienadzorowane ===")
print("Model: KMeans")
print("Podczas fit() użyto: TYLKO X")
print("Silhouette:", round(silhouette, 3))
print("ARI:", round(ari, 3))


# Tworzymy tabelę zestawiającą prawdziwą klasę z numerem klastra KMeans.
# Pozwala to zobaczyć, jak grupowanie algorytmu ma się do rzeczywistych gatunków.
wyniki_nienadzorowane = pd.DataFrame({
    "klasa_prawdziwa": y.values,
    "klaster_KMeans": klastry,
})


# Pokazujemy pierwszych 10 przypisań do klastrów.
print("Pierwsze przypisania do klastrów:")
print(wyniki_nienadzorowane.head(10))


# Tworzymy tabelę krzyżową porównującą prawdziwe klasy i klastry.
print("Klasa prawdziwa, a klaster KMeans:")

# pd.crosstab() zlicza, ile obserwacji każdej prawdziwej klasy
# trafiło do poszczególnych klastrów KMeans.
print(pd.crosstab(
    wyniki_nienadzorowane["klasa_prawdziwa"],
    wyniki_nienadzorowane["klaster_KMeans"],
))


# Poniższe wnioski podsumowują różnicę między uczeniem nadzorowanym i nienadzorowanym.
print("Wnioski:")

# W nadzorowanej klasyfikacji model zna poprawne odpowiedzi podczas treningu.
print("1. W uczeniu nadzorowanym model podczas treningu korzysta z cech X i etykiet y.")

# Klasyfikacja zwraca kategorię, natomiast regresja przewiduje wartość liczbową.
print("2. Klasyfikacja przewiduje kategorię, a regresja przewiduje wartość liczbową")

# W uczeniu nienadzorowanym algorytm sam szuka struktury w cechach X.
print("3. W uczeniu nienadzorowanym algorytm nie otrzymuje poprawnych odpowiedzi y.")

# KMeans grupuje podobne obserwacje bez informacji o nazwach prawdziwych klas.
print("4. KMeans tworzy klastry na podstawie podobieństwa obserwacji, a nie nazw klas.")

# ARI używamy dopiero po klasteryzacji, żeby sprawdzić zgodność klastrów z prawdziwymi klasami.
print("5. ARI wykorzystaliśmy tylko do porównania wyniku KMeans z prawdziwymi klasami po treningu.")
