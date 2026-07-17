
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import cross_val_score, train_test_split

from sklearn.metrics import confusion_matrix



# Carico i csv

train = pd.read_csv("cats_dataset.csv", dtype=str)

test = pd.read_csv("test_set.csv", dtype=str)



# Tolgo i gatti senza razza nel train

train = train.dropna(subset=["razza"])

train = train[train["razza"].str.lower() != "nan"]



# Mi salvo quante righe ha il train per rispaccare i dati dopo

divisore = len(train)



# Unisco tutto temporaneamente 

# dei testi in numeri una volta sola per entrambi i file

df_totale = pd.concat([train, test], ignore_index=True)



# Sistemo i numeri con la virgola e i valori nulli

df_totale["peso_kg"] = df_totale["peso_kg"].str.replace(",", ".", regex=False)

df_totale["peso_kg"] = pd.to_numeric(df_totale["peso_kg"], errors="coerce").fillna(0)

df_totale["eta_anni"] = pd.to_numeric(df_totale["eta_anni"], errors="coerce").fillna(0)



# Converto tutte le colonne di testo in numeri usando le categorie di pandas

colonne_testo = ["sesso", "lunghezza_pelo", "colore_mantello", "livello_attivita", "frequenza_miagolio", "sterilizzato", "patologia", "classe"]



for col in colonne_testo:

    df_totale[col] = df_totale[col].fillna("vuoto").astype("category").cat.codes



# Converto anche la razza

df_totale["razza_num"] = df_totale["razza"].astype("category").cat.codes



# Ricavo i nomi delle razze reali per dopo associandoli ai numeri

mappa_razze = dict(enumerate(df_totale["razza"].astype("category").cat.categories))



# Ora riseparo il dataset in train e test sfruttando l'indice iniziale

df_train_pulito = df_totale.iloc[:divisore].copy()

df_test_pulito = df_totale.iloc[divisore:].copy()



# Creo le matrici X e y per scikit-learn

colonne_features = ["eta_anni", "peso_kg"] + colonne_testo

X = df_train_pulito[colonne_features].values

y = df_train_pulito["razza_num"].values

X_test = df_test_pulito[colonne_features].values



# Modello Gradient Boosting standard

modello = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)

modello.fit(X, y)



# Stampo la precisione del modello con la cross-validation

precisione = cross_val_score(modello, X, y, cv=5)

print(f"Accuratezza media: {precisione.mean() * 100:.2f}%")



# Faccio le predizioni sui gatti del test set

pred_numeri = modello.predict(X_test)

pred_nomi = [mappa_razze[num] for num in pred_numeri]



# Salvo il file csv con i risultati

risultati = pd.DataFrame({

    "ID": df_test_pulito["ID"],

    "razza_prevista": pred_nomi

})

risultati.to_csv("predictions.csv", index=False)

print("File predictions.csv salvato!")





# GRAFICI



# 1. Distribuzione gatti per razza nel train

plt.figure(figsize=(8, 4))

df_train_pulito["razza"].value_counts().plot(kind="bar", color="skyblue", edgecolor="black")

plt.title("Distribuzione delle razze")

plt.xticks(rotation=15)

plt.tight_layout()

plt.savefig("grafici/01_distribuzione_razze.png")

plt.close()



# 2. Correlazione età e peso

plt.figure(figsize=(5, 4))

sns.heatmap(df_train_pulito[["eta_anni", "peso_kg"]].corr(), annot=True, cmap="coolwarm")

plt.title("Correlazione Età - Peso")

plt.tight_layout()

plt.savefig("grafici/02_heatmap_correlazione.png")

plt.close()



# 3. Matrice di confusione

X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

modello_test = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)

modello_test.fit(X_tr, y_tr)



matrice = confusion_matrix(y_val, modello_test.predict(X_val))

plt.figure(figsize=(7, 5))

nomi_classi = list(mappa_razze.values())

sns.heatmap(matrice, annot=True, fmt="d", cmap="Blues", xticklabels=nomi_classi, yticklabels=nomi_classi)

plt.title("Matrice di Confusione")

plt.ylabel("Reale")

plt.xlabel("Predetto")

plt.tight_layout()

plt.savefig("grafici/03_matrice_confusione.png")

plt.close()



# 4. Scatter plot peso vs età fatto con seaborn che gestisce i colori da solo

plt.figure(figsize=(8, 5))

sns.scatterplot(data=df_train_pulito, x="eta_anni", y="peso_kg", hue="razza", alpha=0.7)

plt.title("Peso vs Età per Razza")

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()

plt.savefig("grafici/04_peso_vs_eta_razza.png")

plt.close()



# 5. Importanza delle colonne per l'algoritmo

plt.figure(figsize=(8, 4))

importanze = modello.feature_importances_

indici = np.argsort(importanze)

plt.barh([colonne_features[i] for i in indici], importanze[indici], color="teal")

plt.title("Quali caratteristiche contano di più?")

plt.tight_layout()

plt.savefig("grafici/05_feature_importance.png")

plt.close()



print("grafici esportati")