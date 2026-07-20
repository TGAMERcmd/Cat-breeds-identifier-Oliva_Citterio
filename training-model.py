"""
pandas per caricare i dati nelle tabelle
numpy per calcoli e ordinamento delle liste
matplot per fare grafici
seaborn per fare grafici
sklearn per allenamento del modello
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# costante per migliorare leggibilità andando a capo nelle stampe 
new_line = "\n"

# per caricare i dati di training e di test

train = pd.read_csv("cats_dataset.csv", dtype=str)
test = pd.read_csv("test_set.csv", dtype=str)

# pulizia del dataset per il training(i dati che non contengono una razza)
train = train.dropna(subset=["razza"])
train = train[train["razza"].str.lower() != "nan"]

# Per separare i dataset (che a breve unisco) devo salvarmi il numero di righe originali
numero_righe_originali = len(train)

# DataFrame (df), una struttura bidimensionale simile ad una tabella
"""
Uniamo temporaneamente i due dataset per:
* semplificare operazioni di pulizia
* per fare in modo che ogni dato uguale venga convertito nello stesso numero.
        - Esempio: Siamese = 1 
"""
df = pd.concat([train, test], ignore_index=True)

# pulizia dei dati perchè alcuni valori decimali hanno il punto, altri la virgola 
df["peso_kg"] = df["peso_kg"].str.replace(",", ".", regex=False)


# visto che il il modello non capisce le parole dobbiamo usare i numeri (.to_numeric)
df["peso_kg"] = pd.to_numeric(df["peso_kg"], errors="coerce").fillna(0)
df["eta_anni"] = pd.to_numeric(df["eta_anni"], errors="coerce").fillna(0)

# definisce le colonne del DataFrame
cols = ["sesso", "lunghezza_pelo", "colore_mantello", "livello_attivita",
        "frequenza_miagolio", "sterilizzato", "patologia"]

# converte le colonne in numeri (.cat.codes --> ad ogni valore unico viene assegnato un numero)
# per non perdere informazioni, i valori mancanti vengono riempiti con "vuoto"
for c in cols:
    df[c] = df[c].fillna("vuoto").astype("category").cat.codes

# converte la razza in numero
df["razza_num"] = df["razza"].fillna("sconosciuta").astype("category").cat.codes



# dizionario con "numero razza" --> "nome razza"
# usato per trasformare il numero che il modello predice in razza (stringa)
razze_map = dict(enumerate(df["razza"].fillna("sconosciuta").astype("category").cat.categories))


# separa i dataset (modifiche terminate)
train_df = df.iloc[:numero_righe_originali].copy()
test_df = df.iloc[numero_righe_originali:].copy()


# feat sono le "feature predittive", ovvero i parametri che passeremo al modello per fare le predizioni sulla razza del gatto.
feat = ["eta_anni" , "peso_kg"] + cols

"""
Noi utilizziamo una tecnica particolare. Il valore x rappresenta tutti i parametri da considerare, mentre
y rappresenta il nostro risultato, o predizione, cioè la razza espressa in numeri. All'inizio il modello studia i parametri X  e come trasformano y (dataset di training), 
poi, in assenza di Y (il dataset di test), il modello effettua le predizioni sul suo valore
"""
X = train_df[feat].values
y = train_df["razza_num"].values
X_test_final = test_df[feat].values #dataset di test

"""
In fase di training ci alleniamo solo su parte dei dati, in modo che poi possiamo verificare l'accuratezza del modello.
Per semplicità, lo chiameremo "dataset di controllo". 

Facciamo in modo che tutte le razze siano sia nel dataset di controllo che in quello su cui ci alleniamo, in modo che possiamo
testare l'accuratezza del modello su tutte le razze del dataset completo.

Dopo che abbiamo effettuato il controllo, alleneremo il modello anche sul dataset di controllo.


* test_size=0.2 --> divide il dataset completo 80-20. L'80% è la parte di dataset su cui alleniamo il modello, mentre il 20% è
il dataset di controllo
* stratify=y --> fa in modo che nel dataset di controllo ci siano gatti appartenenti a tutte le razze su cui il modello si è allenato
"""
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

"""
Abbiamo scelto il GradientBoosting perché funziona molto bene su dati tabulari.

Il GradientBoosting allena tanti alberi in sequenza (nel nostro caso, 100). Se un albero sbaglia, gli altri alberi lo correggono.
Questa auto-correzione consente di ottenere un modello molto più preciso, in quanto ogni albero impara dagli errori di un altro.

#! TODO: Chiarifica
* max_depth=4 --> quante domande può fare ogni albero
* random_state=42 --> imposta la casualità per ottenere risultati riproducibili
"""
clf = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
clf.fit(X_tr, y_tr)

# y_prediction_value sono le razze che il modello ha previsto per il dataset di controllo
y_pred_val = clf.predict(X_val)

# calcola l'accuratezza del modello e la stampa come percentuale. Per farlo, trova la quantità di previsioni corrette sul totale delle razze (nel dataset di controllo)
unfinished_model_accuracy = accuracy_score(y_val, y_pred_val)
print(f"{new_line}accuracy validation: {unfinished_model_accuracy * 100:.2f}%")

# divide i dati in 5 "pezzi" e fa l'operazione di prima su questi 5 "pezzi", per avere la misura dell'accuratezza più precisa
# ogni volta usa un gruppo diverso come test e gli altri 4 per allenarsi
cv_scores = cross_val_score(clf, X, y, cv=5)
print(f"cross-val media: {cv_scores.mean() * 100:.2f}% (+/- {cv_scores.std() * 100:.2f}%)")
print(f"fold scores: {[round(s*100,1) for s in cv_scores]}")

# converte ogni numero previsto nella rispettiva razza (Es. 1 --> Siamese) nel dataset di controllo
# Operazione necessaria per il calcolo della matrice di confusione
classi_val = sorted(set(y_val))
nomi_val = [razze_map[c] for c in classi_val]

"""
Stampa in modo dettagliato le info su ogni razza. 

Esempio, 0.90 per i gatti Persiani = 90% di sicurezza che il gatto è un Persiano.

Stampa un report dettagliato su razza usando 4 metriche precise: 
1 precision -- quando effettua una predizione, quante volte ha ragione (es. 0.90 = 90%)
2 recall -- su tutti i gatti appartenenti ad una razza, quanti ne ha trovati (es. 0.97 = ne trova 97 su 100)
3 f1-score -- media pesata tra precision e recall, il "voto finale"
4 support -- quanti gatti di una stessa razza ci sono nel dataset di controllo

Esempio per i Siamesi (non riflette la realtà)
1 - 90% --- 9 volte su 10 indovina correttamente che un gatto è un siamese
2 - 97% --- ha trovato il 97% dei gatti siamesi nel dataset
3 - 93.5% 
4 - 15 -- numero di gatti siamesi nel dataset
"""
print(new_line, classification_report(y_val, y_pred_val, labels=classi_val, target_names=nomi_val), sep="")

# alleno il modello su tutti i dati (quindi anche il dataset di controllo) 
modello_finale = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
modello_finale.fit(X, y)

# predice le razze del dataset di test e converte i numeri previsti nei nomi delle razze (es 1 --> siamese)
preds = modello_finale.predict(X_test_final)
pred_nomi = [razze_map[p] for p in preds]

# salva in un file csv l'ID del gatto e la rispettiva razza prevista
# utilizza il dizionario di prima
out = pd.DataFrame({
    "ID": test_df["ID"],
    "razza_prevista": pred_nomi
})
out.to_csv("predictions.csv", index=False)
print(f"{new_line}predictions.csv salvato")


# grafici 

# distribuzione delle razze nel training set 
plt.figure(figsize=(8, 4))
train_df["razza"].value_counts().plot(kind="bar", color="skyblue", edgecolor="black")
plt.title("razze nel dataset")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("grafici/razze.png")
plt.close()


# heatmap correlazione tra le feature numeriche  richiesto dal regolamento 
plt.figure(figsize=(5, 4))
sns.heatmap(train_df[["eta_anni", "peso_kg"]].corr(), annot=True, cmap="coolwarm")
plt.title("correlazione eta vs peso")
plt.tight_layout()
plt.savefig("grafici/correlazione.png")
plt.close()

# matrice della confusione





#!!!!! TODO __ FIX




# mostra gli errori che il modello effettua in base alla razza
# sulla diagonale ci sono i gatti classificati correttamente
# fuori dalla diagonale ci sono gli errori

mat = confusion_matrix(y_val, y_pred_val, labels=classi_val)
plt.figure(figsize=(7, 5))
sns.heatmap(mat, annot=True, fmt="d", cmap="Blues",
            xticklabels=nomi_val, yticklabels=nomi_val)
plt.title("matrice confusione")
plt.ylabel("reale")
plt.xlabel("predetto")
plt.tight_layout()
plt.savefig("grafici/confusione.png")
plt.close()


#  quali parametri ha pesato di più il modello per decidere la razza
plt.figure(figsize=(8, 4))
imp = modello_finale.feature_importances_
idx = np.argsort(imp)
plt.barh([feat[i] for i in idx], imp[idx], color="teal")
plt.title("feature importance")
plt.tight_layout()
plt.savefig("grafici/importanza.png")
plt.close()

# accuratezza su ogni fold della corss validation 
# un fold è uno dei 5 pezzi in cui viene diviso il dataset
plt.figure(figsize=(6, 3))
plt.bar(range(1, 6), cv_scores * 100, color="steelblue", edgecolor="black")
plt.axhline(cv_scores.mean() * 100, color="red", linestyle="--", label=f"media: {cv_scores.mean()*100:.1f}%")
plt.title("accuracy per fold")
plt.xlabel("fold")
plt.ylabel("accuracy %")
plt.legend()
plt.tight_layout()
plt.savefig("grafici/crossval.png")
plt.close()

print("grafici fatti")
