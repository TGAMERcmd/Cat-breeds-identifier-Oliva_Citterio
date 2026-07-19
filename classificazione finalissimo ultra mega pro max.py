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

#CONTROLLARE RECAL INVECE DI RECALL

# stampa le dimensioni dei due dataset (righe, colonne)
# serve a verificare che i file siano stati caricati correttamente
print(train.shape, test.shape) 

# pulizia del dataset per il training(i dati che non contengono una razza)
train = train.dropna(subset=["razza"])
train = train[train["razza"].str.lower() != "nan"]

# salvo quante righe ha il train così dopo posso riseparare i due dataset
n = len(train)

# DataFrame, una struttura bidimensionale simile ad una tabella
# temporaneamente uniamo train e test, in modo che la stessa stringa verrà 
# convertita nello stesso numero sia nel train che nel test altrimenti il modello si confonderebbe
df = pd.concat([train, test], ignore_index=True)

# pulizia dei dati - alcuni pesi hanno la virgola, altri i punti. Così li rendiamo tutti uguali usando pandas 
df["peso_kg"] = df["peso_kg"].str.replace(",", ".", regex=False)
df["peso_kg"] = pd.to_numeric(df["peso_kg"], errors="coerce").fillna(0)
df["eta_anni"] = pd.to_numeric(df["eta_anni"], errors="coerce").fillna(0)

# definisce le colonne del DataFrame, e le converte in numeri (.cat.codes)
# visto che il il modello non capisce le parole dobbiamo usare i numeri 
# cat.codes converte ogni valore unico in un numero 
# valori mancanti riempio con "vuoto" così non perdono informazione
cols = ["sesso", "lunghezza_pelo", "colore_mantello", "livello_attivita",
        "frequenza_miagolio", "sterilizzato", "patologia"]

for c in cols:
    df[c] = df[c].fillna("vuoto").astype("category").cat.codes
# converto anche la razza in numero, che è quello che il modello dovrà predire
df["razza_num"] = df["razza"].fillna("sconosciuta").astype("category").cat.codes

# dizionario con numero razza --> nome razza che verrà usato nuovamente per riconvertire le predizione sulla razza
razze_map = dict(enumerate(df["razza"].fillna("sconosciuta").astype("category").cat.categories))

# abbiamo finito con le modifiche ad entrambi i dataset, 
# ora li separiamo nuovamente
train_df = df.iloc[:n].copy()
test_df = df.iloc[n:].copy()

"""
feat sono le "feature predittive", ovvero i parametri che passeremo al modello per fare le predizioni
sulla razza del gatto.
"""
feat = ["eta_anni" , "peso_kg"] + cols

"""
Noi utilizziamo una tecnica particolare. Il valore x rappresenta tutti i parametri da considerare, mentre
y rappresenta il nostro risultato, o predizione, cioè la razza espressa in numeri. All'inizio il modello studia i parametri X  e come trasformano y (dataset di training), 
poi, in assenza di Y (il dataset di test), il modello effettua le predizioni sul suo valore
"""
X = train_df[feat].values
y = train_df["razza_num"].values
X_test_final = test_df[feat].values #dataset di test

# si allena sull'80% dei gatti (in fase di training -- test_size=0.2). Poi facciamo il controllo sul restante 20% per verificare quanto sbaglia 
# stratify=y garantisce che tutte le razze siano presenti sia nell'80% che nel 20% 
# altrimenti potrebbe capitare che una razza finisca tutta nell'80% e il modello non venga mai testato su di essa
#! TODO: Correggere spiegazione
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# abbiamo scelto GradientBoosting perché funziona molto bene su dati tabulari come in questa challange 
# allena tanti alberi in sequenza (100). Se uno sbaglia, gli altri alberi lo correggono.
# così otteniamo un modello molto più preciso
# gradient boosting allena 100 alberi decisionali in sequenza
#dopo
# ogni albero impara dagli errori di quello prima in tal modo che il modello possa migliorare iterazione dopo iterazione
# max_depth=4 quante domande può fare ogni albero
# random_state=42 fissa il caso così i risultati sono riproducibili
clf = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
clf.fit(X_tr, y_tr)

# y_prediction_value sono le razze che il modello ha previsto per il 20% di validazione
y_pred_val = clf.predict(X_val)

#! TODO: migliora commenti di questa sezione

# confronto le razze previste con quelle reali e calcolo la percentuale di quelle giuste e la stampo
acc = accuracy_score(y_val, y_pred_val)
print(f"{new_line}accuracy validation: {acc * 100:.2f}%")

# divide i dati in 5 "pezzi" e fa i test su questi 5 "pezzi", per avere un'accuratezza migliore
# ogni volta usa un gruppo diverso come test e gli altri 4 per allenarsi
cv_scores = cross_val_score(clf, X, y, cv=5)
print(f"cross-val media: {cv_scores.mean() * 100:.2f}% (+/- {cv_scores.std() * 100:.2f}%)")
print(f"fold scores: {[round(s*100,1) for s in cv_scores]}")

# trasforma i numeri in nomi di razze che esistono nel 20% di validazione e le converto in nomi
# serve per la matrice di confusione e il report sotto
classi_val = sorted(set(y_val))
nomi_val = [razze_map[c] for c in classi_val]
# stampa in modo dettagliato le info su ogni razza. Ad esempio, 0.90 per i gatti Persiani significa che quando 
# predice che un gatto è persiano è sicuro al 90%
#dopo
# stampa un report dettagliato per ogni razza  usando 4 metriche precise 
# 1 precision quando dice "è un Bengala" quante volte ha ragione (es. 0.90 = 90%)
# 2 recal di tutti i Bengala reali, quanti ne ha trovati (es. 0.97 = ne trova 97 su 100)
# 3 f1-score è la media bilanciata tra precision e recall, il voto finale possiamo dire
# 4 support quanti gatti di quella razza ci sono nel 20% di test
print(new_line, classification_report(y_val, y_pred_val, labels=classi_val, target_names=nomi_val), sep="")

## rialleno il modello su tutti i dati del train 100% non solo l 80%
modello_finale = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
modello_finale.fit(X, y)

# predice le razze del dataset di test  e converte i numeri in nomi
preds = modello_finale.predict(X_test_final)
pred_nomi = [razze_map[p] for p in preds]

# salva in un file csv l'ID del gatto e la sua razza prevista usando una tabella
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

# matrice della confusione cioè mostra fli errori del modello razza per razza
# sulla diagonale ci sono i gatti classificati correttamente
# fuori dalla diagonale ci sono gli erorri

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
