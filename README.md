# Cat-Breed-Classifier_Oliva-Citterio

La soluzione di Oliva Riccardo e Tommaso Citterio alla challenge "Cat Breed Classifier", dove dobbiamo addestrare un modello di classificazione supervisionata in grado di predire la razza del gatto a partire dalle altre caratteristiche (fisiche e comportamentali).

## Modello scelto

Per risolvere questa challenge abbiamo addestrato solo un modello.
Utilizziamo GradientBoostingClassifier (scikit-learn) con 100 alberi e max_depth=4, scelto da noi perché funziona molto bene su dati tabulari con feature miste

## Risultati

- Accuracy validation set: 95.26%
- Accuracy cross-validation (5 fold): 96.63% 

Il dataset di training contiene 5 record con razza "Alien" che abbasa l'accuracy del modello. Nonostante ciò, **non** l'abbiamo scartato. 
Il modello li classifica come "outlier", evitando quindi di allucinare razze inesistenti


## Librerie utilizzate

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
