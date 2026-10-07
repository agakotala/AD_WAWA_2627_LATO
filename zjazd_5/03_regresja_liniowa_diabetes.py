# Importujemy moduł pyplot z biblioteki matplotlib i nadajemy mu skrót plt.
# Służy on do tworzenia wykresów i wizualnego porównywania wyników modelu.
import matplotlib.pyplot as plt


# Importujemy bibliotekę pandas i nadajemy jej skrót pd.
# Pandas będzie używany do tworzenia Series i DataFrame z wynikami modelu.
import pandas as pd

# Importujemy funkcję load_diabetes.
# Pozwala ona wczytać wbudowany zbiór danych Diabetes przeznaczony do regresji.
from sklearn.datasets import load_diabetes

# Importujemy model LinearRegression.
# Regresja liniowa służy do przewidywania wartości liczbowych na podstawie cech wejściowych.
from sklearn.linear_model import LinearRegression

# Importujemy metrykę mean_squared_error, czyli MSE.
# MSE mierzy średni kwadrat błędu między wartościami rzeczywistymi i przewidywanymi.
from sklearn.metrics import mean_squared_error

# Importujemy metrykę r2_score.
# R2 pokazuje, jaka część zmienności wartości docelowej jest wyjaśniana przez model.
from sklearn.metrics import r2_score

# Importujemy metrykę mean_absolute_error, czyli MAE.
# MAE mierzy przeciętną bezwzględną różnicę między wartością rzeczywistą i przewidywaną.
from sklearn.metrics import mean_absolute_error

# Importujemy funkcję train_test_split.
# Służy ona do podziału danych na zbiór treningowy i testowy.
from sklearn.model_selection import train_test_split


# Wczytujemy zbiór danych Diabetes.
# Parametr as_frame=True powoduje zwrócenie danych w formacie pandas.
zbior = load_diabetes(as_frame=True)


# Tworzymy X zawierające wszystkie cechy wejściowe.
# Każdy wiersz odpowiada jednej obserwacji, a kolumny opisują cechy pacjenta.
X = zbior.data

# Tworzymy y zawierające zmienną docelową.
# Jest to wartość liczbowa, którą model regresji będzie próbował przewidywać.
y = zbior.target


# Pokazujemy przykładowe obserwacje z pełnej tabeli danych.
print("Pierwsze wiersze danych:")

# zbior.frame zawiera zarówno cechy, jak i kolumnę docelową.
# head(10) zwraca pierwszych 10 wierszy tabeli.
print(zbior.frame.head(10))


# X.shape zwraca liczbę obserwacji oraz liczbę cech wejściowych.
print("Kształt cech X:", X.shape)

# y.shape pokazuje, ile wartości docelowych znajduje się w zbiorze.
# Liczba ta powinna odpowiadać liczbie wierszy w X.
print("Kształt celu y:", y.shape)


# Dzielimy dane na zbiór treningowy i testowy.
# Model nauczy się zależności na danych treningowych,
# a jego skuteczność sprawdzimy na danych testowych.
X_train, X_test, y_train, y_test = train_test_split(
    # Cechy wejściowe.
    X,

    # Wartości docelowe odpowiadające obserwacjom z X.
    y,

    # 30% danych trafia do zbioru testowego,
    # a pozostałe 70% do zbioru treningowego.
    test_size=0.3,

    # Ustalamy stałe ziarno losowości,
    # aby podział danych był powtarzalny.
    random_state=42,
)


# Tworzymy podstawowy model regresji liniowej.
# Model będzie szukał liniowej zależności między cechami X a wartością y.
model = LinearRegression()


# Uczymy model na zbiorze treningowym.
# fit() wyznacza współczynniki dla poszczególnych cech oraz wyraz wolny.
model.fit(X_train, y_train)


# Korzystając z wytrenowanego modelu przewidujemy wartości y dla X_test.
# Wynik zapisujemy w y_pred.
y_pred = model.predict(X_test)


# Obliczamy MAE, czyli średni błąd bezwzględny.
# Im mniejsza wartość MAE, tym przeciętnie mniejsze są błędy predykcji.
mae = mean_absolute_error(y_test, y_pred)

# Obliczamy MSE, czyli średni błąd kwadratowy.
# Większe błędy są tutaj karane mocniej, ponieważ różnica jest podnoszona do kwadratu.
mse = mean_squared_error(y_test, y_pred)

# Obliczamy współczynnik R2.
# Wartość bliższa 1 oznacza lepsze dopasowanie modelu do danych.
r2 = r2_score(y_test, y_pred)


# Poniżej analizujemy współczynniki wyznaczone przez regresję liniową.
print("Współczynniki modelu:")

# model.coef_ zawiera współczynniki przypisane poszczególnym cechom.
# Tworzymy z nich obiekt Series i jako indeks ustawiamy nazwy kolumn X,
# aby łatwo było sprawdzić, który współczynnik należy do której cechy.
wspolczynniki = pd.Series(model.coef_, index=X.columns)

# sort_values(ascending=False) sortuje współczynniki od największego do najmniejszego.
# Dodatni współczynnik oznacza, że przy pozostałych cechach bez zmian
# wzrost danej cechy jest związany ze wzrostem przewidywanej wartości.
print(wspolczynniki.sort_values(ascending=False))


# Wyraz wolny jest wartością bazową równania regresji.
print("Wyraz wolny modelu:")

# model.intercept_ zawiera wyraz wolny, a round(..., 3) zaokrągla go do 3 miejsc po przecinku.
print(round(model.intercept_, 3))


# Wyświetlamy obliczone wcześniej metryki oceny modelu.
print("Metryki oceny:")

# MAE pokazuje przeciętną wielkość błędu w jednostkach zmiennej docelowej.
print("MAE:", round(mae, 3))

# MSE jest średnią z kwadratów błędów, więc mocniej reaguje na duże pomyłki.
print("MSE:", round(mse, 3))

# R2 informuje o stopniu wyjaśnienia zmienności wartości docelowej przez model.
print("R2:", round(r2, 3))


# Tworzymy tabelę porównującą wartości rzeczywiste z przewidywaniami modelu.
wyniki = pd.DataFrame({
    # y_test.values pobiera rzeczywiste wartości ze zbioru testowego bez indeksów pandas.
    "rzeczywiste": y_test.values,

    # y_pred zawiera wartości przewidziane przez model regresji liniowej.
    "przewidywane": y_pred,
})


# Dodajemy kolumnę "blad" jako różnicę wartości rzeczywistej i przewidywanej.
# Dodatni błąd oznacza, że model przewidział za małą wartość,
# a ujemny, że przewidział za dużą.
wyniki["blad"] = wyniki["rzeczywiste"] - wyniki["przewidywane"]


# Pokazujemy pierwszych 10 porównań wartości rzeczywistych i przewidywanych.
print("Pierwsze 10 wyników")
print(wyniki.head(10))


# Tworzymy drugi model regresji liniowej.
# Parametr positive=True wymusza, aby wszystkie współczynniki cech były nieujemne.
# Oznacza to, że model nie może przypisać żadnej cesze ujemnego współczynnika.
model_dodatni = LinearRegression(positive=True)


# Uczymy model z ograniczeniem dodatnich współczynników na tych samych danych treningowych.
model_dodatni.fit(X_train, y_train)


# Przewidujemy wartości dla zbioru testowego za pomocą modelu z dodatnimi współczynnikami.
y_pred_dodatni = model_dodatni.predict(X_test)


# Ponownie liczymy MAE, tym razem dla modelu z positive=True.
# Nadpisujemy wcześniejszą zmienną mae nową wartością.
mae = mean_absolute_error(y_test, y_pred_dodatni)

# Liczymy MSE dla drugiego modelu i również nadpisujemy wcześniejszą wartość mse.
mse = mean_squared_error(y_test, y_pred_dodatni)

# Liczymy R2 dla modelu z dodatnimi współczynnikami.
r2 = r2_score(y_test, y_pred_dodatni)


# Sprawdzamy współczynniki drugiego modelu.
print("Współczynniki modelu dodatniego:")

# Tworzymy Series z nowych współczynników i przypisujemy im nazwy cech.
# Ponieważ użyto positive=True, współczynniki nie powinny być ujemne.
wspolczynniki = pd.Series(model_dodatni.coef_, index=X.columns)

# Sortujemy współczynniki od największego do najmniejszego.
print(wspolczynniki.sort_values(ascending=False))


# Sprawdzamy wyraz wolny modelu z dodatnimi współczynnikami.
print("Wyraz wolny modelu dodatniego:")
print(round(model_dodatni.intercept_, 3))


# Pokazujemy metryki drugiego modelu, aby można było porównać go z modelem podstawowym.
print("Metryki oceny modelu dodatniego:")
print("MAE:", round(mae, 3))
print("MSE:", round(mse, 3))
print("R2:", round(r2, 3))


# Tworzymy nową tabelę wyników dla modelu z dodatnimi współczynnikami.
# Zmienna wyniki zostaje zastąpiona nowym DataFrame.
wyniki = pd.DataFrame({
    "rzeczywiste": y_test.values,
    "przewidywane": y_pred_dodatni,
})


# Ponownie obliczamy błąd jako różnicę między wartością rzeczywistą i przewidywaną.
wyniki["blad"] = wyniki["rzeczywiste"] - wyniki["przewidywane"]


# Pokazujemy pierwszych 10 wyników drugiego modelu.
print("Pierwsze 10 wyników modelu dodatniego:")
print(wyniki.head(10))


# Tworzymy figurę o rozmiarze 12 na 8 cali dla pierwszego wykresu.
plt.figure(figsize=(12, 8))

# Tworzymy wykres rozrzutu wartości rzeczywistych i przewidywanych.
# Oś X zawiera wartości rzeczywiste, a oś Y przewidywania modelu.
# alpha=0.7 ustawia lekką przezroczystość punktów.
plt.scatter(y_test, y_pred, alpha=0.7)


# Rysujemy linię idealnych predykcji y = x.
# Jeśli przewidywanie jest dokładnie równe wartości rzeczywistej,
# punkt powinien znaleźć się właśnie na tej przekątnej.
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()])


# Opisujemy oś poziomą jako wartości rzeczywiste.
plt.xlabel("Wartości rzeczywiste")

# Opisujemy oś pionową jako wartości przewidywane.
plt.ylabel("Wartości przewidywane")


# Nadajemy wykresowi tytuł.
plt.title("Regresja liniowa: rzeczywiste vs przewidywane")


# tight_layout() poprawia rozmieszczenie elementów wykresu.
plt.tight_layout()


# Wyświetlamy wykres pierwszego modelu.
plt.show()


# Tworzymy osobną figurę dla modelu z dodatnimi współczynnikami.
plt.figure(figsize=(12, 8))

# Rysujemy punkty: wartości rzeczywiste kontra przewidywania drugiego modelu.
plt.scatter(y_test, y_pred_dodatni, alpha=0.7)


# Dodajemy tę samą linię idealnych predykcji y = x.
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()])


# Opisujemy osie wykresu.
plt.xlabel("Wartości rzeczywiste")
plt.ylabel("Wartości przewidywane")


# Nadajemy tytuł informujący, że jest to model z dodatnimi współczynnikami.
plt.title("Regresja liniowa dodatnie współczynniki: rzeczywiste vs przewidywane")


# Poprawiamy rozmieszczenie elementów wykresu.
plt.tight_layout()


# Wyświetlamy drugi wykres.
plt.show()


print("Interpretacja")

# Im bliżej punkt leży linii y=x, tym mniejszy jest błąd dla danej obserwacji.
print("1. Im bliżej punkty są na linii przekątnej, tym lepsze predykcje")

# MAE daje łatwą do interpretacji średnią wielkość błędu.
print("2. MAE mówi o przeciętnym błędzie w jednostkach zmiennej docelowej")

# R2 pokazuje, jak dobrze model wyjaśnia zmienność wartości docelowej.
print("3. R2 pokazuje, jaka część zmienności została wyjaśniona")
