# Cat-Breed-Classifier_Oliva-Citterio

La soluzione di Oliva Riccardo e Tommaso Citterio alla challenge "Cat Breed Classifier", dove dobbiamo addestrare un modello di classificazione supervisionata in grado di predire la razza del gatto a partire dalle altre caratteristiche (fisiche e comportamentali).

# Preprarazione

Rimossi i record senza razza dal training set
Uniformati i valori decimali da virgola a punto
Convertite le colonne testuali in numeri con cat.codes
Train e test uniti temporaneamente durante la conversione, per garantire che la stessa stringa riceva lo stesso numero in entrambi i dataset


  
# Modello scelto

Per risolvere questa challenge abbiamo addestrato solo un modello.
Utilizziamo GradientBoostingClassifier (scikit-learn) con 100 alberi e max_depth=4, scelto da noi perché funziona molto bene su dati tabulari con feature miste e ogni albero impara dagli errori del precedente, corregendoli iterazione dopo iterazione, rendendo il modello ancora più preciso

# Risultati

- Accuracy validation set: 95.26%
- Accuracy cross-validation (5 fold): 96.63% 

Il dataset di training contiene 5 record con razza "Alien" che abbasa l'accuracy del modello. Nonostante ciò, **non** l'abbiamo scartato. 
Il modello li classifica come "outlier", evitando quindi di allucinare razze inesistenti
Rimuovendo Alien l'accuracy salirebbe al 97.46%, ma perderemmo la dimostrazione di robustezza.



## Librerie utilizzate

- pandas caricamento e manipolazione dati
- numpy  calcoli numerici
- matplotlib  grafici
- seaborn  grafici avanzati
- scikit-learn modello, valutazione e preparazione
