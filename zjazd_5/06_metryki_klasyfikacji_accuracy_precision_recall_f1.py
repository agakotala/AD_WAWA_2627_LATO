# Importujemy bibliotekę pandas i nadajemy jej skrót pd.
# Pandas posłuży do utworzenia tabeli porównującej prawdziwe i przewidziane klasy.
import pandas as pd

# Importujemy funkcję load_breast_cancer.
# Pozwala ona wczytać wbudowany zbiór danych Breast Cancer Wisconsin.
from sklearn.datasets import load_breast_cancer

# Importujemy LogisticRegression, czyli model klasyfikacji binarnej.
from sklearn.linear_model import LogisticRegression

# Importujemy train_test_split do podziału danych na treningowe i testowe.
from sklearn.model_selection import train_test_split

# Importujemy StandardScaler do standaryzacji cech przed uczeniem modelu.
from sklearn.preprocessing import StandardScaler

# Importujemy Pipeline, który połączy skalowanie i regresję logistyczną
# w jeden uporządkowany proces.
from sklearn.pipeline import Pipeline

# Importujemy accuracy_score.
# Accuracy określa, jaka część wszystkich predykcji była poprawna.
from sklearn.metrics import accuracy_score

# Importujemy precision_score.
# Precision mówi, jaka część przypadków przewidzianych jako pozytywne
# rzeczywiście należy do klasy pozytywnej.
from sklearn.metrics import precision_score

# Importujemy recall_score.
# Recall mówi, jaką część wszystkich rzeczywistych przypadków pozytywnych
# model poprawnie wykrył.
from sklearn.metrics import recall_score

# Importujemy f1_score.
# F1 jest średnią harmoniczną precision i recall
# i pozwala połączyć obie te metryki w jedną wartość.
from sklearn.metrics import f1_score

# Importujemy confusion_matrix.
# Macierz pomyłek pokazuje liczbę wyników TN, FP, FN i TP.
from sklearn.metrics import confusion_matrix

# Importujemy classification_report.
# Funkcja tworzy zbiorczy raport zawierający m.in. precision, recall i F1 dla klas.
from sklearn.metrics import classification_report


# Wczytujemy zbiór Breast Cancer Wisconsin w formacie pandas.
zbior = load_breast_cancer(as_frame=True)

# X zawiera wszystkie cechy wejściowe opisujące badane przypadki.
X = zbior.data

# W oryginalnym zbiorze target=0 oznacza nowotwór złośliwy,
# a target=1 oznacza nowotwór łagodny.
# Warunek zbior.target == 0 zwraca True dla przypadków złośliwych.
# astype(int) zamienia True na 1 i False na 0.
# Dzięki temu w tym skrypcie klasa pozytywna 1 oznacza nowotwór złośliwy,
# co ułatwia interpretację recall w kontekście wykrywania choroby.
y = (zbior.target == 0).astype(int)


# Poniżej zapisujemy jawnie sposób kodowania klas używany w tym skrypcie.
print("KODOWANIE W TYM SKRYPCIE:")
print("1 - klasa pozytywna - nowotwór złośliwy")
print("0 - klasa negatywna - nowotwór łagodny")


# Sprawdzamy liczebność obu klas po zmianie kodowania.
print("Liczba obserwacji w klasach:")

# value_counts() zlicza wystąpienia 0 i 1,
# a sort_index() porządkuje wynik według numeru klasy.
print(y.value_counts().sort_index())


# Dzielimy dane na zbiór treningowy i testowy.
X_train, X_test, y_train, y_test = train_test_split(
    # Cechy wejściowe.
    X,

    # Etykiety z nowym kodowaniem klas.
    y,

    # 25% danych przeznaczamy na test, a 75% na trening.
    test_size=0.25,

    # Ustalamy stałe ziarno losowości dla powtarzalnego podziału.
    random_state=42,

    # Zachowujemy proporcje klasy pozytywnej i negatywnej w obu częściach danych.
    stratify=y,
)


# Tworzymy pipeline składający się ze skalowania i modelu klasyfikacyjnego.
model = Pipeline(steps=[
    # StandardScaler ujednolica skalę poszczególnych cech.
    ("skalowanie", StandardScaler()),

    # LogisticRegression uczy się przewidywać klasę 0 lub 1.
    # max_iter=1000 zwiększa maksymalną liczbę iteracji optymalizacji.
    ("model", LogisticRegression(max_iter=1000, random_state=42)),
])


# Uczymy pipeline na danych treningowych.
# Model otrzymuje cechy X_train oraz odpowiadające im prawdziwe etykiety y_train.
model.fit(X_train, y_train)


# Przewidujemy klasy dla danych testowych.
y_pred = model.predict(X_test)


# Tworzymy macierz pomyłek przez porównanie y_test z y_pred.
# Dla klasyfikacji binarnej otrzymujemy macierz 2x2.
macierz = confusion_matrix(y_test, y_pred)


# ravel() spłaszcza macierz 2x2 do czterech wartości.
# Przy standardowym układzie confusion_matrix są to kolejno:
# TN - True Negative,
# FP - False Positive,
# FN - False Negative,
# TP - True Positive.
tn, fp, fn, tp = macierz.ravel()


# Pokazujemy pełną macierz pomyłek.
print("Macierz pomyłek:")
print(macierz)


# Poniżej opisujemy znaczenie każdej z czterech wartości macierzy.
print("Elementy macierzy:")

# TN oznacza przypadek naprawdę łagodny, który model poprawnie przewidział jako łagodny.
print("TN - True Negative:", tn, "-> poprawnie przewidziany łagodny")

# FP oznacza przypadek łagodny błędnie uznany za złośliwy.
# Jest to fałszywy alarm modelu.
print("FP - False Positive:", fp, "-> łagodny błędnie uznany za złośliwy")

# FN oznacza przypadek złośliwy błędnie uznany za łagodny.
# W kontekście diagnostycznym jest to szczególnie istotny rodzaj błędu,
# ponieważ model nie wykrył przypadku należącego do klasy pozytywnej.
print("FN - False Negative:", fn, "-> złośliwy błędnie uznany za łagodny")

# TP oznacza przypadek złośliwy poprawnie wykryty jako złośliwy.
print("TP - True Positive:", tp, "-> poprawnie przewidziany jako złośliwy")


# Accuracy oblicza udział wszystkich poprawnych predykcji,
# czyli zarówno poprawnych pozytywnych, jak i poprawnych negatywnych.
accuracy = accuracy_score(y_test, y_pred)


# Precision odpowiada na pytanie:
# spośród przypadków przewidzianych jako złośliwe, ile rzeczywiście było złośliwych?
precision = precision_score(y_test, y_pred)


# Recall odpowiada na pytanie:
# spośród wszystkich rzeczywiście złośliwych przypadków, ile model wykrył?
recall = recall_score(y_test, y_pred)


# F1 łączy precision i recall w jedną metrykę.
# Jest wysokie wtedy, gdy oba te wyniki są jednocześnie wysokie.
f1 = f1_score(y_test, y_pred)


# Pokazujemy metryki obliczone gotowymi funkcjami biblioteki scikit-learn.
print("METRYKI SCIKIT-LEARN:")
print("Accuracy:", round(accuracy, 3))
print("Precision:", round(precision, 3))
print("Recall:", round(recall, 3))
print("F1 Score:", round(f1, 3))


# Poniżej obliczamy te same metryki ręcznie na podstawie TN, FP, FN i TP.

# Accuracy = (TP + TN) / liczba wszystkich przypadków.
# Licznik zawiera wszystkie poprawne predykcje.
accuracy_recznie = (tp + tn) / (tp+ tn + fp + fn)

# Precision = TP / (TP + FP).
# Warunek if zabezpiecza przed dzieleniem przez zero,
# gdyby model nie przewidział ani jednego przypadku pozytywnego.
precision_recznie = tp / (tp + fp) if (tp + fp) > 0 else 0

# Recall = TP / (TP + FN).
# Pokazuje udział poprawnie wykrytych przypadków pozytywnych.
# Warunek również zabezpiecza przed dzieleniem przez zero.
recall_recznie = tp / (tp + fn) if (tp + fn) > 0 else 0

# F1 obliczamy ze wzoru 2 * precision * recall / (precision + recall).
# W liczniku oryginalnego kodu użyto zmiennej precision obliczonej przez scikit-learn,
# a w mianowniku precision_recznie i recall_recznie.
# Dla tych samych y_test i y_pred wartość precision jest taka sama jak precision_recznie,
# więc wynik powinien odpowiadać ręcznemu obliczeniu F1.
f1_recznie = (
    2 * precision * recall_recznie
    / (precision_recznie + recall_recznie)
    if (precision_recznie + recall_recznie) > 0
    else 0
)


# Pokazujemy wartości metryk policzonych ze wzorów.
# Możemy porównać je z wynikami funkcji scikit-learn.
print("METRYKI POLICZONE RĘCZNIE:")
print("Accuracy:", round(accuracy_recznie, 3))
print("Precision:", round(precision_recznie, 3))
print("Recall:", round(recall_recznie, 3))
print("F1 Score:", round(f1_recznie, 3))


# classification_report tworzy bardziej rozbudowane zestawienie metryk dla obu klas.
print("CLASSIFICATION REPORT:")

# Do funkcji przekazujemy prawdziwe etykiety i predykcje.
# target_names przypisuje czytelne nazwy klas do wartości 0 i 1.
# Raport zawiera m.in. precision, recall, F1-score i support,
# gdzie support oznacza liczbę rzeczywistych obserwacji danej klasy.
print(classification_report(
    y_test,
    y_pred,
    target_names=["łagodna (0)", "złośliwa (1)"]
))


# Tworzymy DataFrame pozwalający przeanalizować każdą predykcję osobno.
wyniki = pd.DataFrame({
    # Prawdziwa klasa ze zbioru testowego.
    "prawdziwa_klasa": y_test.values,

    # Klasa przewidziana przez model.
    "przewidziana_klasa": y_pred,
})


# Dodajemy kolumnę informującą, czy predykcja była poprawna.
# Porównanie dwóch kolumn zwraca True dla poprawnej predykcji
# i False dla błędnej.
wyniki["poprawna"] = (
    wyniki["prawdziwa_klasa"] == wyniki["przewidziana_klasa"]
)


# Definiujemy funkcję, która dla pojedynczego wiersza tabeli
# określa typ wyniku: TP, TN, FP albo FN.
def typ_wyniku(wiersz):

    # Jeśli prawdziwa klasa to 1 i model również przewidział 1,
    # mamy True Positive, czyli poprawnie wykryty przypadek złośliwy.
    if wiersz["prawdziwa_klasa"] == 1 and wiersz["przewidziana_klasa"] == 1:
        return "TP"

    # Jeśli prawdziwa klasa to 0 i model przewidział 0,
    # mamy True Negative, czyli poprawnie rozpoznany przypadek łagodny.
    if wiersz["prawdziwa_klasa"] == 0 and wiersz["przewidziana_klasa"] == 0:
        return "TN"

    # Jeśli prawdziwa klasa to 0, ale model przewidział 1,
    # mamy False Positive, czyli przypadek łagodny uznany za złośliwy.
    if wiersz["prawdziwa_klasa"] == 0 and wiersz["przewidziana_klasa"] == 1:
        return "FP"

    # Jeżeli żaden z powyższych warunków nie został spełniony,
    # pozostaje przypadek: prawdziwa klasa 1, przewidziana 0,
    # czyli False Negative.
    return "FN"


# apply() uruchamia funkcję typ_wyniku dla każdego wiersza tabeli wyniki.
# axis=1 oznacza wykonywanie funkcji wiersz po wierszu.
# Zwrócone oznaczenia zapisujemy w nowej kolumnie "typ_wyniku".
wyniki["typ_wyniku"] = wyniki.apply(typ_wyniku, axis=1)


# Pokazujemy pierwszych 20 przypadków wraz z informacją,
# czy wynik był poprawny oraz jaki typ wyniku reprezentuje.
print("Pierwsze 20 predykcji:")
print(wyniki.head(20))


# value_counts() zlicza, ile razy wystąpiło TP, TN, FP i FN.
# Jest to inny sposób podsumowania informacji zawartych w macierzy pomyłek.
print("Liczebność typów wyników:")
print(wyniki["typ_wyniku"].value_counts())

print("\nINTERPRETACJA:")

# Accuracy określa, jaka część wszystkich predykcji modelu była poprawna.
# Uwzględnia zarówno poprawnie rozpoznane przypadki pozytywne,
# jak i poprawnie rozpoznane przypadki negatywne.
print("1. Accuracy pokazuje udział wszystkich poprawnych predykcji.")


# Precision informuje, jaka część przypadków przewidzianych przez model
# jako pozytywne rzeczywiście należy do klasy pozytywnej.
# Wysokie precision oznacza, że model rzadko generuje fałszywe alarmy.
print("2. Precision mówi, jak wiarygodne są przewidywania klasy pozytywnej.")


# Recall określa, jaką część wszystkich rzeczywistych przypadków pozytywnych
# model poprawnie rozpoznał jako pozytywne.
# Wysoki recall oznacza, że model rzadko pomija przypadki pozytywne.
print("3. Recall mówi, jak dużą część prawdziwych przypadków pozytywnych model wykrył.")


# F1-score łączy precision i recall w jedną wartość.
# Jest średnią harmoniczną tych dwóch metryk,
# dlatego osiąga wysoką wartość tylko wtedy,
# gdy zarówno precision, jak i recall są wysokie.
print("4. F1-score łączy precision i recall i jest wysokie tylko wtedy, gdy oba wyniki są dobre.")


# False Positive oznacza sytuację, w której model przewidział klasę pozytywną,
# mimo że prawdziwa klasa była negatywna.
# Można to interpretować jako fałszywy alarm.
#
# False Negative oznacza sytuację, w której model przewidział klasę negatywną,
# mimo że przypadek w rzeczywistości należał do klasy pozytywnej.
# Oznacza to przeoczenie przypadku pozytywnego.
print("5. False Positive to fałszywy alarm, a False Negative to przeoczony przypadek pozytywny.")


# Precision i recall są obliczane względem klasy uznanej za pozytywną.
# Jeżeli zmienimy sposób kodowania klas i wybierzemy inną klasę jako pozytywną,
# zmieni się również znaczenie tych metryk.
# Dlatego przed interpretacją wyników trzeba wiedzieć,
# która wartość oznacza klasę pozytywną.
print("6. Wybór klasy pozytywnej ma bezpośredni wpływ na interpretację precision i recall.")


# Jeśli jedna klasa występuje znacznie częściej niż druga,
# model może osiągnąć wysokie accuracy przez częste przewidywanie klasy dominującej.
# Mimo wysokiego accuracy może wtedy słabo rozpoznawać rzadszą klasę.
# Dlatego przy niezbalansowanych danych warto analizować również
# precision, recall, F1-score i macierz pomyłek.
print("7. Przy nierównych klasach samo accuracy może wyglądać dobrze mimo słabego wykrywania rzadszej klasy.")


# Macierz pomyłek pokazuje dokładną liczbę wyników:
# True Positive, True Negative, False Positive i False Negative.
# Dzięki temu można zobaczyć, jakie rodzaje błędów popełnia model.
# Sama wartość accuracy podaje tylko ogólny procent poprawnych predykcji
# i nie pokazuje, które klasy były najczęściej mylone.
print("8. Macierz pomyłek pokazuje szczegóły, których pojedyncza liczba accuracy nie pokazuje.")