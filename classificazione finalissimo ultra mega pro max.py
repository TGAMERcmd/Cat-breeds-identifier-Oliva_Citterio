import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import confusion_matrix




# carico i dataset
train = pd.read_csv("cats_dataset.csv", dtype=str)
test = pd.read_csv("test_set.csv", dtype=str)

print("train:", train.shape)
print("test:", test.shape)

# rimuovo le righe senza razza perché non servono per allenare il modello
train = train.dropna(subset=["razza"])
train = train[train["razza"].str.lower() != "nan"]

print("righe train dopo pulizia:", len(train))

# salvo quante righe ha il train così dopo posso riseparare i due dataset
n_train = len(train)

# unisco tutto così converto i testi in numeri una volta sola
tutto = pd.concat([train, test], ignore_index=True)

# sistemo peso ed età che a volte hanno la virgola invece del punto
tutto["peso_kg"] = tutto["peso_kg"].str.replace(",", ".", regex=False)
tutto["peso_kg"] = pd.to_numeric(tutto["peso_kg"], errors="coerce").fillna(0)
tutto["eta_anni"] = pd.to_numeric(tutto["eta_anni"], errors="coerce").fillna(0)

# converto le colonne di testo in numeri
colonne_testo = ["sesso", "lunghezza_pelo", "colore_mantello", "livello_attivita",
                 "frequenza_miagolio", "sterilizzato", "patologia"]

for col in colonne_testo:
    tutto[col] = tutto[col].fillna("vuoto").astype("category").cat.codes

# converto anche la razza che è quello che voglio predire
tutto["razza_num"] = tutto["razza"].fillna("sconosciuta").astype("category").cat.codes
mappa_razze = dict(enumerate(tutto["razza"].fillna("sconosciuta").astype("category").cat.categories))

# riseparo train e test
df_train = tutto.iloc[:n_train].copy()
df_test = tutto.iloc[n_train:].copy()

# preparo i dati per sklearn
features = ["eta_anni", "peso_kg"] + colonne_testo

X = df_train[features].values
y = df_train["razza_num"].values
X_test = df_test[features].values

print("features usate:", features)

# alleno il modello
modello = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
modello.fit(X, y)

# vedo quanto è preciso con la cross validation (l'ho visto in un tutorial)
scores = cross_val_score(modello, X, y, cv=5)
print(f"accuratezza media: {scores.mean() * 100:.2f}%")

# faccio le predizioni sul test set
pred_numeri = modello.predict(X_test)
pred_nomi = [mappa_razze[n] for n in pred_numeri]

# salvo i risultati
risultati = pd.DataFrame({
    "ID": df_test["ID"],
    "razza_prevista": pred_nomi
})
risultati.to_csv("predictions.csv", index=False)
print("predictions.csv salvato!")


# ---- grafici ----

# 1) quanti gatti per razza ci sono nel train
plt.figure(figsize=(8, 4))
df_train["razza"].value_counts().plot(kind="bar", color="skyblue", edgecolor="black")
plt.title("quante gatti per razza")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("grafici/01_razze.png")
plt.close()

# 2) correlazione tra età e peso
plt.figure(figsize=(5, 4))
sns.heatmap(df_train[["eta_anni", "peso_kg"]].corr(), annot=True, cmap="coolwarm")
plt.title("età vs peso")
plt.tight_layout()
plt.savefig("grafici/02_correlazione.png")
plt.close()

# 3) matrice di confusione
# splitTo train/val per vedere gli errori del modello
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

modello2 = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
modello2.fit(X_tr, y_tr)

# FIX: uso solo le classi che esistono davvero in y_val, non tutte della mappa
# altrimenti la matrice ha righe/colonne in più e non torna
classi_val = sorted(set(y_val))
nomi_classi_val = [mappa_razze[c] for c in classi_val]

matrice = confusion_matrix(y_val, modello2.predict(X_val), labels=classi_val)

plt.figure(figsize=(7, 5))
sns.heatmap(matrice, annot=True, fmt="d", cmap="Blues",
            xticklabels=nomi_classi_val, yticklabels=nomi_classi_val)
plt.title("matrice di confusione")
plt.ylabel("reale")
plt.xlabel("predetto")
plt.tight_layout()
plt.savefig("grafici/03_confusione.png")
plt.close()

# 4) scatter peso vs età colorato per razza
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df_train, x="eta_anni", y="peso_kg", hue="razza", alpha=0.7)
plt.title("peso vs età")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig("grafici/04_scatter.png")
plt.close()

# 5) quali feature contano di più per il modello
plt.figure(figsize=(8, 4))
importanze = modello.feature_importances_
ordine = np.argsort(importanze)
plt.barh([features[i] for i in ordine], importanze[ordine], color="teal")
plt.title("feature importance")
plt.tight_layout()
plt.savefig("grafici/05_importanza.png")
plt.close()

print("grafici salvati!")
