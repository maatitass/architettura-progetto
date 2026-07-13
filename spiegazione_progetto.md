# Spiegazione Dettagliata del Progetto: Studio di Robustezza al Rumore

Di seguito viene riportato **tutto** il codice del notebook originale, suddiviso per blocchi logici e funzioni. Per soddisfare appieno le esigenze di comprensione, ogni blocco presenta:
- **🌍 Analisi Macroscopica**: Spiega il senso logico del blocco all'interno dell'intero progetto (il "perché").
- **🔬 Analisi Microscopica**: Analizza la sintassi, le funzioni, i parametri e le trasformazioni matematiche riga per riga (il "come").

---

## 1. Importazione delle Librerie

```python
import os
import math
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    recall_score, f1_score, roc_curve, roc_auc_score, log_loss
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score, normalized_mutual_info_score,
    silhouette_score, adjusted_mutual_info_score
)
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_class_weight
from scipy import stats
from scipy.stats.mstats import winsorize

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
print('Librerie caricate OK — TensorFlow', tf.__version__)
```

### 🌍 Analisi Macroscopica
L'obiettivo di questo primo blocco è preparare l'ambiente di lavoro. Un progetto di Machine Learning richiede strumenti specializzati per l'algebra lineare, la gestione dei dati in formato tabellare, il plotting statistico e ovviamente i framework per addestrare i modelli predittivi. Qui importiamo l'intero ecosistema necessario (Scikit-Learn per il ML tradizionale, TensorFlow/Keras per il Deep Learning, e librerie matematiche avanzate come SciPy) in un colpo solo.

### 🔬 Analisi Microscopica
- `import numpy as np` / `import pandas as pd`: Numpy è la base per i vettori e le matrici in Python. Pandas (pd) costruisce sopra numpy la struttura DataFrame, essenziale per interrogare e filtrare le tabelle (similmente a un database SQL o a Excel).
- `import matplotlib.pyplot as plt` / `import seaborn as sns`: Librerie per disegnare i grafici. Seaborn avvolge Matplotlib rendendo la sintassi più elegante e fornendo temi statistici pre-impostati.
- `from sklearn.model_selection import train_test_split`: Una funzione fondamentale che spacca i vettori di dati in modo probabilistico o stratificato per creare i set isolati di allenamento e validazione.
- `from sklearn.preprocessing import StandardScaler`: Una classe che memorizza la media e la deviazione standard del set di training per poter trasformare (scalare) le distribuzioni dei dati su una curva normale Z.
- `from sklearn.metrics import ...`: Importazione in blocco di tutti i misuratori di performance. `accuracy_score` per la purezza, `f1_score` per l'equilibrio recall-precision, `roc_auc_score` per il potenziale separatore, `confusion_matrix` per l'identificazione di falsi positivi/negativi.
- `from sklearn.ensemble import RandomForestClassifier`: Importa l'algoritmo Ensemble che aggrega moltissimi alberi decisionali (Tree) estratti tramite "bagging".
- `from sklearn.cluster import KMeans` e `from sklearn.decomposition import PCA`: Strumenti per l'Apprendimento Non Supervisionato (Unsupervised Learning). Il primo per dividere i dati alla cieca in `k` gruppi, il secondo per applicare una rotazione vettoriale (autovettori) in grado di schiacciare 10 dimensioni in solo 2 per poterle disegnare su schermo (Dimensionality Reduction).
- `from scipy.stats.mstats import winsorize`: Metodo statistico puro che blocca gli outlier appiattendoli ai percentili definiti, al posto di rimuoverli.
- `import tensorflow as tf` e sottomoduli `keras`: L'ecosistema di calcolo a tensori di Google. `Sequential` modella la rete come una "pila" di strati, `Dense` istanzia un layer in cui tutti i neuroni comunicano con il livello precedente. `Dropout`, `EarlyStopping`, e `l2` sono tecniche matematiche rigorose chiamate "Regolarizzatori" per impedire l'overfitting.
- `warnings.filterwarnings('ignore')`: Istruzione tecnica che sopprime log a schermo irrilevanti (es. warning di deprecazione di alcune funzioni interne) per mantenere il notebook pulito.
- `sns.set_style('whitegrid')`: Sovrascrive lo sfondo grigio di default dei plot inserendo una griglia bianca utile per stimare i valori ad occhio nei grafici a linea che verranno stampati dopo.

---

## 2. Caricamento e Preparazione del Dataset

```python
# Caricamento del dataset
col_names = ['fLength', 'fWidth', 'fSize', 'fConc', 'fConc1',
             'fAsym', 'fM3Long', 'fM3Trans', 'fAlpha', 'fDist', 'class']

df = pd.read_csv('magic04.data', names=col_names)
df['class'] = df['class'].map({'h': 0, 'g': 1})

print('Dataset caricato con successo in locale! Dimensioni:', df.shape)
```

### 🌍 Analisi Macroscopica
In questa cella avviene l'ingestione (Data Ingestion) dei dati grezzi dal file system alla RAM del computer. Inoltre, si risolve immediatamente il problema della classificazione del target: i modelli non possono ottimizzare derivate matematiche su stringhe testuali ('h', 'g'), quindi il target categoriale viene trasformato in un formato numerico binario (0, 1), ponendo le basi logiche del problema (0=Hadron, 1=Gamma).

### 🔬 Analisi Microscopica
- `col_names = [...]`: Crea una struttura Lista in Python. Poiché il set di dati pubblico di MAGIC Telescope non ha metadati intrinseci nella prima riga del file CSV (nessun header), definiamo esplicitamente le 11 colonne che rappresentano le variabili (features del telescopio di Cherenkov).
- `pd.read_csv('magic04.data', names=col_names)`: Il parser CSV di pandas scansiona il file riga per riga, separandolo tramite la virgola (default). Associa i dati grezzi alla maschera `col_names`.
- `df['class']`: Isola la "Serie" (colonna monodimensionale in pandas) contenente le etichette reali.
- `.map({'h': 0, 'g': 1})`: Questa funzione prende in input un dizionario Python. Cerca ogni occorrenza della chiave `'h'` e la rimpiazza col valore `0`. Sostituisce `'g'` con `1`. Questo rende il dataset completamente compatibile con funzioni di Loss quali la "Binary Cross-Entropy".
- `df.shape`: Un attributo di pandas che restituisce una Tupla `(righe, colonne)`. Il `print` mostra all'analista quante istanze sono effettivamente entrate in memoria (nel caso specifico, 19020 righe per 11 colonne).

---

## 3. Suddivisione del Dataset (Train, Val, Test)

```python
X = df.drop(['class'], axis=1)
y = df['class']
seed = 42

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=seed, stratify=y)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=seed, stratify=y_temp)

df_train = pd.concat([X_train, y_train], axis=1)
print(f'Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}')
```

### 🌍 Analisi Macroscopica
Un mantra del Machine Learning è che un modello non deve mai essere giudicato sui dati che ha usato per studiare (sarebbe come dare le soluzioni prima dell'esame). Questo blocco suddivide matematicamente l'intera mole di dati in 3 bacini non comunicanti: Training (per fare imparare la rete aggiornandone i pesi), Validation (per decidere quando fermare il training in anticipo e misurare i progressi epocali), e Test (set blindato, usato solo alla primissima fine per emettere il giudizio sulle metriche ufficiali).

### 🔬 Analisi Microscopica
- `X = df.drop(['class'], axis=1)`: Genera la matrice delle Features. `drop` distrugge la colonna `class` e l'argomento `axis=1` specifica che l'asse di cancellazione è verticale (la colonna), non orizzontale (la riga).
- `seed = 42`: Stabilisce l'entropia del modulo randomico di Python. "42" è la convenzione informatica standard, ma ciò che conta è che fissandolo, la funzione shuffle di sklearn taglierà le 19020 righe esattamente nello stesso split ad ogni avvio del notebook (riproducibilità accademica).
- Il primo `train_test_split`: Passa `X` (input) e `y` (target). `test_size=0.4` alloca il 60% dei dati (11412 righe) a `X_train` e `y_train`. Il resto (40%) va nel set `temp`. 
- `stratify=y`: Questo parametro è CRITICO. Se il dataset originario ha il 65% di classe 1 e il 35% di classe 0, l'istruzione `stratify` forza l'algoritmo di split a mantenere questa esatta asimmetria in tutti i set derivati, prevenendo set di allenamento sbilanciati che potrebbero deviare l'apprendimento.
- Il secondo `train_test_split`: Prende l'output temporaneo del passo prima (il 40% residuo) e applica un `test_size=0.5`. Spacca il 40% a metà, creando il 20% di validazione (`X_val, y_val`) e il 20% di Test purissimo (`X_test, y_test`).
- `pd.concat([X_train, y_train], axis=1)`: Salda di nuovo input e target assieme, ma *esclusivamente per le istanze di training*. Questo genera un dataframe dedicato solo all'Analisi Esplorativa (EDA). Se facessimo esplorazione dati sul set intero, le intuizioni visive violerebbero l'isolamento del Test set.

---

## 4. Analisi Esplorativa: Qualità Dati e De-duplicazione

```python
print('Valori mancanti per colonna:\n', df.isnull().sum())
print(f'\nDuplicati trovati prima della pulizia: {df_train.duplicated().sum()}')
# NB: la deduplica agisce solo su df_train, usato per l'EDA di questa sezione;
# X_train/y_train restano interi e i modelli si addestrano sull'intero training set.
df_train = df_train.drop_duplicates()
print('Dataset Training dopo rimozione duplicati:', df_train.shape)
```

### 🌍 Analisi Macroscopica
Questa è la primissima validazione di "sanità" dei dati (Data Cleaning). Cerca inconsistenze macroscopiche come i valori nulli (che farebbero mandare in crash matematico una rete neurale) e righe clonate. Rimuoviamo i duplicati solo a scopo analitico: questo eviterà che i grafici a torta o gli istogrammi presentino picchi irrealistici derivati da artefatti di campionamento del telescopio.

### 🔬 Analisi Microscopica
- `df.isnull()`: Crea una maschera Booleana (matrice di True e False) per tutto il dataset. True se c'è un valore `NaN` (Not a Number), False se la cella è valida.
- `.sum()`: Siccome in Python `True` vale 1 e `False` vale 0, sommando la colonna otteniamo il conteggio numerico esatto dei record corrotti per ogni sensore.
- `df_train.duplicated()`: Identifica record che hanno esattamente le stesse variazioni fino all'ultima cifra decimale in ogni singola feature, il che spesso implica che il sensore ha loggato due volte lo stesso medesimo evento fisico per latenze di sistema.
- `df_train.drop_duplicates()`: Cancella le righe marchiate come `True` nel comando precedente. Modifica l'oggetto ricreandolo compresso a 11371 righe. Il training set vero (`X_train`) resta illibato.

---

## 5. Analisi Esplorativa: Distribuzione (Sbilanciamento di Classe)

```python
signal_class = df_train['class'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(signal_class, labels=['gamma (1)', 'hadron (0)'],
        autopct='%1.1f%%', startangle=90, colors=['#ff9999', '#66b3ff'])
plt.title('Distribuzione del Target nel Training Set')
plt.axis('equal'); plt.show()
```

### 🌍 Analisi Macroscopica
Un aspetto essenziale del design di una Rete Neurale è sapere in anticipo se si dovrà combattere uno Sbilanciamento di Classe (Class Imbalance). In questo blocco si quantifica visivamente il rapporto di forza tra i segnali (Gamma) e il rumore di fondo cosmico (Hadron). Se una classe domina per il 90%, dovremo implementare la manipolazione dei pesi nel modello.

### 🔬 Analisi Microscopica
- `df_train['class'].value_counts()`: Una funzione pandas estremamente efficiente che calcola la frequenza assoluta. Restituisce una struttura associativa: chiave (classe 1 o 0) e valore (es. 7000 o 4000).
- `plt.figure(figsize=(6, 6))`: Prende l'engine Matplotlib, crea l'oggetto Canvas (Finestra di tracciato) dandogli proporzioni perfettamente quadrate (6 pollici per 6 pollici) per evitare distorsioni ellittiche della torta.
- `plt.pie(signal_class, ...)`: Genera il Pie Chart usando i conteggi generati sopra.
- `autopct='%1.1f%%'`: Espressione sintattica di interpolazione. Dichiara a Matplotlib di calcolare autonomamente le quote relative e formattarle come stringa Floating Point ad 1 singola cifra decimale (`.1f`), affiggendo il simbolo `%`.
- `colors=['#ff9999', '#66b3ff']`: Asssegna colori HTML esadecimali pastello personalizzati per i settori, per migliorare la leggibilità didattica.
- `plt.axis('equal')`: Assicura matematicamente che l'aspect ratio (rapporto x/y) del canvas mantenga la geometria circolare della torta al resising della finestra.

---

## 6. Analisi Esplorativa: Distribuzioni Univariate e Outlier (Boxplot)

```python
numeric_cols = [c for c in df_train.select_dtypes(include=['float64', 'int64']).columns if c != 'class']
n_cols = 3
n_rows = math.ceil(len(numeric_cols) / n_cols)
plt.figure(figsize=(15, 4 * n_rows))
for i, col in enumerate(numeric_cols):
    plt.subplot(n_rows, n_cols, i + 1)
    sns.histplot(df_train[col], kde=True, bins=30, edgecolor='black')
    plt.title(f'Distribuzione di {col}'); plt.xlabel(''); plt.ylabel('Frequenza')
plt.tight_layout(); plt.show()
```

```python
plt.figure(figsize=(15, 4 * n_rows))
for i, col in enumerate(numeric_cols):
    plt.subplot(n_rows, n_cols, i + 1)
    sns.boxplot(x='class', y=col, hue='class', data=df_train, palette='Set2', legend=False)
    plt.title(f'{col} vs Class')
plt.tight_layout(); plt.show()
```

### 🌍 Analisi Macroscopica
Questi due blocchi affrontano l'osservazione microscopica dei segnali di ciascun sensore.
Nel **primo blocco** (Istogrammi) si studia la curva statistica dei dati. Molti algoritmi preferiscono distribuzioni normali a campana; se le curve sono fortemente sbilanciate (skewed), potrebbero servire trasformazioni logaritmiche.
Nel **secondo blocco** (Boxplot) si valuta l'ispezione bivariata tra classe e feature per un obiettivo primario: scovare gli estremi aberranti (Outlier). Gli strumenti del telescopio MAGIC producono code anomale pesantissime e il boxplot lo denuncia graficamente (i puntini fuori dai "baffi"). Inoltre, permette di vedere a colpo d'occhio quali sensori (es. fAlpha) distanziano nettamente le due classi, offrendo un alto "potere predittivo".

### 🔬 Analisi Microscopica
- `numeric_cols = [c for c in ... if c != 'class']`: List Comprehension. Esplora il framework `.dtypes` di pandas che classifica i tipi di memoria per ogni colonna. Mantiene solo i numerici reali (`float64`, `int64`) ed esclude con un costrutto logico `if` la colonna di target, salvando una lista di 10 stringhe.
- `n_cols = 3` e `n_rows = math.ceil(len(numeric_cols) / n_cols)`: Matematica base per formare la griglia (Grid). Calcola che 10 feature divise in 3 colonne richiedono `3.33` righe. La funzione `math.ceil` del modulo nativo arrotonda per eccesso a `4` righe intere di grafici.
- `for i, col in enumerate(numeric_cols)`: Questo costrutto Python restituisce contemporaneamente l'indice (`i` da 0 a 9) e il nome della feature (`col`).
- `plt.subplot(n_rows, n_cols, i + 1)`: Dichiara quale sotto-finestra della matrice grafica è attiva in questa iterazione. L'indice richiede `+1` perché Matplotlib è 1-indexed, non 0-indexed per i subplot.
- `sns.histplot(..., kde=True, bins=30)`: Usa Seaborn per contare quanti campioni ricadono in 30 scompartimenti (bins) verticali. `kde=True` applica la Kernel Density Estimation, una curva derivativa complessa che traccia la stima di densità probabilistica spalmando l'istogramma grezzo, utilissima per localizzare code asimmetriche (Left/Right Skewness).
- `sns.boxplot(x='class', y=col, hue='class', ...)`: Configura il Boxplot orientato verticalmente. Mettendo `class` sia su asse x che come variante di colore (`hue`), Seaborn divide i dati di quella colonna in 2 blocchi fisicamente separati e cromaticamente distinti, permettendo di valutare visivamente la differenza di Mediana (la riga interna alla scatola) e il Quartile Range.
- `plt.tight_layout()`: Invoca un algoritmo interno alla libreria grafica che calcola l'ingombro dei font dei titoli e ridimensiona e ridistanzia i margini dei `subplot` per prevenire "accavallamenti" visivi dei testi.

---

## 7. Analisi Esplorativa: Matrice di Correlazione (Multicollinearità)

```python
plt.figure(figsize=(10, 8))
sns.heatmap(df_train[numeric_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
plt.title('Matrice di Correlazione'); plt.show()
```

### 🌍 Analisi Macroscopica
Nel Machine Learning, features fortemente ridondanti (multicollinearità) possono rallentare l'addestramento e confondere l'algoritmo sulle reali origini del segnale (effetto di masking). Questa griglia cromatica calcola il coefficiente di correlazione tra tutte le variabili contemporaneamente. Se due variabili hanno correlazione >0.8 o <-0.8 (colore rosso vivo o blu denso), dicono essenzialmente la stessa cosa, e potrebbero teoricamente essere fuse tramite PCA o rimosse.

### 🔬 Analisi Microscopica
- `df_train[numeric_cols].corr()`: Un metodo matematico potentissimo di pandas che esegue una correlazione di Pearson $r$ incrociata (pairwise) estraendo una matrice simmetrica N x N tra tutte le features fornite.
- `sns.heatmap(..., annot=True)`: Traduce la matrice numerica in una tabella a mappatura cromatica. `annot=True` sovrappone i numeri grezzi all'interno dei box colorati per poterne leggere il peso esatto.
- `fmt='.2f'`: Limita rigorosamente la stampa a video del float point a 2 cifre decimali onde evitare di saturare visivamente la casella con lunghe precisioni inutili.
- `cmap='coolwarm'`: Specifica la Color Map (tavolozza graduata) di Matplotlib. `coolwarm` è una mappa divergente, ideale per variazioni intorno allo zero: il blu intenso segnala correlazione negativa (proporzionalità inversa), il bianco assenza totale, e il rosso intenso correlazione positiva stretta (proporzionalità diretta).

---

## 8. Preprocessing dei Dati: Standardizzazione e Cast

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

y_train = y_train.astype('int64')
y_val   = y_val.astype('int64')
y_test  = y_test.astype('int64')

feature_names = list(X.columns)
print('Scaling completato. Feature:', feature_names)
```

### 🌍 Analisi Macroscopica
Le reti neurali computano i risultati moltiplicando vettori su matrici di pesi. Se una colonna (es. fSize) viaggia intorno a valori = 10.000 e una (es. fAsym) viaggia intorno a 0.5, i gradienti in retro-propagazione "esplodono" (Vanishing/Exploding Gradient) disorientando l'ottimizzatore Adam. La **Standardizzazione** forza tutte le colonne ad avere medie centrate sullo zero e ampiezza uguale a 1, un livellamento che assicura stabilità assoluta durante il training.

### 🔬 Analisi Microscopica
- `scaler = StandardScaler()`: Istanzia la classe di SciKit Learn. L'equazione matematica che applica ad ogni singola cella è: $Z = \frac{(x - \mu)}{\sigma}$. (Sottrae la media della colonna, e divide per la deviazione standard della colonna).
- `scaler.fit_transform(X_train)`: Operazione composita. `fit` analizza l'intera matrice e calcola le Medie $\mu$ e le Dev.Standard $\sigma$ di quel campione specifico, custodendole nella memoria dello StandardScaler. `transform` applica l'equazione $Z$ su tutte le 11412 righe restituendo un NdArray numpy scalato.
- `scaler.transform(X_val)` e `scaler.transform(X_test)`: Regola d'oro del ML: mai chiamare il metodo `fit` sui dati di test. Qui l'algoritmo non ricalcola una nuova media, ma schiaccia i dati isolati del Test set **usando le identiche Medie e Varianze pre-scoperte dal Training Set**. Questo simula a livello matematico l'ingresso in un motore di dati mai visti nella realtà, salvaguardando il progetto dal *Data Leakage*.
- `.astype('int64')`: Un controllo di formattazione sui dati in uscita. TensorFlow potrebbe riscontrare crash o warning se il target binario viene interpretato come tipo Float. Castiamo imperativamente il target (0, 1) al tipo intero a 64 bit (`int64`) forzando la mano.
- `feature_names = list(X.columns)`: Conserva una copia sicura dei nomi delle stringhe (le feature). Nel passaggio da Pandas (`X_train`) a Numpy Array (`X_train_scaled`), i nomi di colonna vanno perduti strutturalmente, e ne avremo bisogno a fine codice per plottare la *Feature Importance* in Random Forest.

---

## 9. Funzioni di Noise Injection (Corruzione Manuale dei Dati)

Il cuore dello studio: una batteria di 5 metodologie scientifiche separate per degradare, viziare e distruggere intere percentuali del Dataset. Lo scopo è valutare come si comporterà la macchina quando inserita in una pipeline nel mondo reale dove i sensori si guastano, il trasferimento dati perde pacchetti o ci sono errori umani.

### Blocco 9.1: Gaussian Feature Noise (Vibrazione sui segnali)
```python
def add_gaussian_noise(X, noise_level=0.1, seed=42, cols=None):
    rng = np.random.default_rng(seed)
    if cols is None:
        return X + rng.normal(0.0, noise_level, size=X.shape)
    X_noisy = X.copy()
    cols = list(cols)
    X_noisy[:, cols] += rng.normal(0.0, noise_level, size=(X.shape[0], len(cols)))
    return X_noisy
```
#### 🌍 Analisi Macroscopica
Simula la componente spuria (l'imprecisione dell'elettronica fotomoltiplicatrice) delle antenne del MAGIC Telescope. Aggiunge a ogni singolo valore un minuscolo numero casuale, spostando il segnale e disallineandolo dal pattern originale.
#### 🔬 Analisi Microscopica
- `rng = np.random.default_rng(seed)`: Metodo moderno (scritto post-Numpy 1.17) per creare l'oggetto generatore di probabilità PCG64 garantendo replicabilità stocastica a prova di crash di threading, molto più avanzato della legacy `np.random.seed`.
- `if cols is None:`: Meccanismo di fallback per applicare un degrado di scala totale e globale se non specitifichiamo colonne.
- `rng.normal(0.0, noise_level, size=X.shape)`: L'equazione genera una mole di dati immensa, un'intera Matrice casuale Gaussiana enorme quanto la shape intera di tutto il dataset (`X.shape`). Siccome la media è `0.0`, l'incertezza del disturbo è centrata. La severity è dictata da `noise_level` che funge da parametro deviazione standard (Sigma).
- `X + rng.normal(...)`: Avviene il **Broadcasting** tensoriale di NumPy, un'operazione fulminea che fa l'addizione cella per cella (element-wise addition) e restituisce i dati istantaneamente disassati. 
- Il blocco ad isolamento (`X_noisy[:, cols] +=`): Invece di far esplodere la matrice globale intera, agisce per tranciamento vettoriale locale sulle colonne passate, sporcando esclusivamente e pesantemente alcuni sensori specifici e lasciando gli altri candidi.

### Blocco 9.2: Label Noise Random (Scombussolamento Classi Casuale)
```python
def add_label_noise_random(y, noise_level=0.1, seed=42):
    rng = np.random.default_rng(seed)
    y_noisy = np.array(y).copy()
    n_flips = int(len(y_noisy) * noise_level)
    idx = rng.choice(len(y_noisy), size=n_flips, replace=False)
    y_noisy[idx] = 1 - y_noisy[idx]
    return y_noisy

add_label_noise = add_label_noise_random 
```
#### 🌍 Analisi Macroscopica
Il peggior nemico del training supervisionato: L'oracolo che istruisce la macchina le insegna concetti intrinsecamente falsi (esempio: le dice che la foto di un gatto è un cane). Si verifica nella realtà per cause legate agli errori umani di annotazione e misclassificazione da catalogo stellare errato.
#### 🔬 Analisi Microscopica
- `y_noisy = np.array(y).copy()`: Forza la copia profonda (`.copy()`) in memoria della Series di label al fine di non sovrascrivere l'originale intonso per un difetto intrinseco del puntatore di allocazione Python.
- `n_flips = int(len(y_noisy) * noise_level)`: Calcolo algebrico. Data un'infezione del 25% su 1000 righe, estrae l'intero numerico equivalente `250` bersagli.
- `rng.choice(..., size=n_flips, replace=False)`: L'algoritmo va a campionare (scegliere a sorte) 250 indici univoci dall'elenco del set. `replace=False` garantisce che il campionamento avvenga senza re-imbussolamento, vietando allo script di selezionare due volte la stessa singola cella vanificando l'inversione.
- `y_noisy[idx] = 1 - y_noisy[idx]`: L'aritmetica di Boolean Flipper. Siccome il target è `{0, 1}`, elaborando `1 - 1 = 0`, ed elaborando `1 - 0 = 1`. Ribalta efficacemente la matrice di target in forma element-wise sfruttando array indices mask.

### Blocco 9.3: Label Noise Borderline (Etichettatura Errata Ostile-Mirata)
```python
def add_label_noise_borderline(X_scaled, y, noise_level=0.1, seed=42):
    """Inverte le etichette dei campioni piu' vicini al centroide della classe opposta."""
    y_arr = np.array(y).copy()
    n_flips = int(len(y_arr) * noise_level)
    centroids = {c: X_scaled[y_arr == c].mean(axis=0) for c in [0, 1]}
    distances = np.array([np.linalg.norm(X_scaled[i] - centroids[1 - y_arr[i]])
                          for i in range(len(y_arr))])
    borderline_idx = np.argsort(distances)[:n_flips]
    y_arr[borderline_idx] = 1 - y_arr[borderline_idx]
    return y_arr
```
#### 🌍 Analisi Macroscopica
Un rumore teorico più avanzato. A differenza dello script precedente, qui l'errore non capita a caso. Viene infettato selettivamente tutto quel blocco di dati che sta posizionato sul margine geometrico di decisione della classificazione, dove l'ambiguità astronomica tra Gamma e Hadron è massima, per spaccare letalmente le reti che provano a disegnare dei boundary iperpiani.
#### 🔬 Analisi Microscopica
- `centroids = {c: X_scaled[y_arr == c].mean(axis=0) for c in [0, 1]}`: Comprehension Avanzata di dizionario. Filtra matematicamente lo spazio dimensionale isolando tutti gli N dati di classe 0, calcolandone per essi la media (`mean(axis=0)` per riga) ovvero il Centroide logico Baricentrico del Cluster 0. Ripete in egual misura per il Cluster 1.
- Il ciclo per la formula di Geometria Analitica: `[np.linalg.norm(X_scaled[i] - centroids[1 - y_arr[i]]) ...]`: 
    1. Legge di quale classe è in verità il record $i$ (es. è uno 0).
    2. Identifica la classe antagonista (`1 - 0 = 1`).
    3. Richiama dal modulo SciPy `np.linalg.norm` che calcola la distanza Vettoriale-Euclidea nello spazio ad 11 Dimensioni, tra il nostro campione e il cuore pulsante (il centroide calcolato prima) della classe in cui **NON** appartiene.
- `borderline_idx = np.argsort(distances)[:n_flips]`: L'`argsort` non ordina gli scalari, ma restituisce la scaletta ordinata dei "Puntatori di Indice" che rappresentano quegli elementi ordinati in forma crescente dalla distanza Eudlidea Minore in assoluto. Ne "Slice-a" solo i primi `[:n_flips]` (i target geometricamente ambigui e pericolosamente insidiati nello spazio alieno).
- Sconvolge matematicamente quei soli bersagli scelti e restituisce il frame corrotto per l'aggressione ai margini operativi del classificatore.

### Blocco 9.4: Valori Mancanti e Algoritmi di Imputazione Correttiva
```python
def add_missing_values(X, missing_rate=0.1, seed=42):
    rng = np.random.default_rng(seed)
    X_missing = X.astype(float).copy()
    X_missing[rng.random(X.shape) < missing_rate] = np.nan
    return X_missing

def impute_missing(X_train_m, X_test_m, strategy='mean'):
    imp = SimpleImputer(strategy=strategy)
    return imp.fit_transform(X_train_m), imp.transform(X_test_m)

def impute_train_val_test(X_tr, X_val, X_te, strategy='mean'):
    """Un solo imputer fittato sul train, applicato a val e test (no leakage)."""
    imp = SimpleImputer(strategy=strategy)
    X_tr_i = imp.fit_transform(X_tr)
    return X_tr_i, imp.transform(X_val), imp.transform(X_te)
```
#### 🌍 Analisi Macroscopica
Un telescopio a volte perde le connesioni (Packet Loss). Quando la cella della matrice diventa vuota, la moltiplicazione della rete fallisce generando errore `NaN-Propagation`. Questo script prima fora il database casualmente distruggendo un Rateo % dei record, e poi utilizza algoritmi statistici riparatori (Mean, Median) per stuccare i fori e consentire di portare a termine l'epoch di addestramento.
#### 🔬 Analisi Microscopica
- `X_missing[rng.random(X.shape) < missing_rate] = np.nan`: Istanzia una matrice Boolean (True-False) enorme quanto i dati `X`. `rng.random` genera cifre casuali da 0 a 1 uniformemente. Ovunque questo decimale casuale è inferiore al tasso distruttivo `missing_rate` (es. `0.25`), la cella corrispettiva in NumPy viene dichiarata "morta", sostituendola brutalmente in memoria con la Costante Nativa Flottante `np.nan`.
- `imp = SimpleImputer(strategy=strategy)`: Istanzia la Classe Scikit Learn specializzata per le Toppe (Imputing). Inietta come strategia parametrica statistica un arg, comunemente definito `mean` o `median`.
- `imp.fit_transform(X_tr)` vs `imp.transform(...)`: Uguale alla questione della Standardizzazione descritta prima. Scansiona e memorizza la Mediana di una colonna affetta da Nan, *solo dal Training set*. E applica la patch sostituendo la Mediana registrata, stuccando i buchi. Non potendo estrarre le mediane dal Validation e dal Test senza spiare la densità dei dati futuri, su questi si limita passivamente al mero e cieco `transform()` colmando i loro buchi usando i valori numerici calcolati poc'anzi dal training set.

### Blocco 9.5: Generazione Outlier Maligni e Strategie di Contenimento
```python
def add_outliers(X, outlier_rate=0.05, magnitude=5.0, seed=42):
    rng = np.random.default_rng(seed)
    X_out = X.copy()
    n_out = int(len(X_out) * outlier_rate)
    idx = rng.choice(len(X_out), size=n_out, replace=False)
    sign = rng.choice([-1, 1], size=(n_out, X_out.shape[1]))
    X_out[idx] = sign * magnitude
    return X_out

def winsorize_features(X, limits=(0.05, 0.05)):
    """Tronca le code (5% per lato) di ogni feature: attenua gli outlier senza eliminarli."""
    return np.column_stack([np.asarray(winsorize(X[:, j], limits=limits))
                            for j in range(X.shape[1])])

def impute_outliers(X, thresh=4.0, strategy='median'):
    """Marca come NaN i valori |z|>thresh e li imputa (mediana)."""
    Xc = X.astype(float).copy()
    Xc[np.abs(Xc) > thresh] = np.nan
    return SimpleImputer(strategy=strategy).fit_transform(Xc)
```
#### 🌍 Analisi Macroscopica
Un sensore che invia un segnale esageratamente altissimo di voltaggio causa un Outlier, innescando esplosione dei loss. Qui modifichiamo una % dei record spostandoli in una magnitudine spaventosa (5 z-score), simulando picchi alieni. Vengono offerte due contro-strategie sperimentali: la **Winsorizzazione** (che pialla il picco ad un limite di decenza umana fissa) e l'**Imputazione ad ablazione** (che straccia ogni anomalia considerandola alla stregua di un NaN (dato mancante) sostituendolo con la calma placida della Mediana).
#### 🔬 Analisi Microscopica
- `sign = rng.choice([-1, 1], size=(n_out, X_out.shape[1]))`: Crea una matrice di segni (positivo e negativo) scelti per randomizzazione statistica. 
- `X_out[idx] = sign * magnitude`: Applica l'algoritmo di deviazione Outlier in formato broadacasting. Assegna al blocco identificato dall'`idx` (che contiene per esempio l'intera anomalia sulle 11 features) un valore schiacciante, ponendolo per esempio fisso a `+5.0` e `-5.0` deviando l'allenamento della Neural Network.
- `np.column_stack([... for j in range(X.shape[1])])`: Nella variante di attenuazione. Elabora un ciclo che scorre asincronamente tutte le colonne 1 ad 1 iterando su `j`.
- `winsorize(X[:, j], limits=(0.05, 0.05))`: Algoritmo statistico puro. Impone che tutto il materiale alieno al di fuori del 5° percentile inferiore, e fuori dal 95° percentile superiore, venga fisicamente piallato appiattendosi sul tetto del limite percentile esatto imposto in `limits`, sopprimendo le vette senza distruggere la traccia del record o riga originaria.
- `np.abs(Xc) > thresh`: Nella variante 3 (Imputazione ad Ablazione). Misura il valore Assoluto rimuovendo il segno `-`. Controlla la maschera tensoriale di NumPy. Se una qualunque cella sfora maledettamente per una threshold estrema (e.g. `> 4.0`), l'istruzione la flagga arbitrariamente ad artefatto irrecuperabile uccidendola assegnandovi il valore `np.nan`. Dopo che tutte le aberrazioni estreme divengono nan, il blocco istanzia uno SciKit Imputer passandogli di base la strategia non lineare ed ultra solida ai disturbi: La "mediana" logica `strategy=median` per arginarlo.

### Blocco 9.6: Distruzione Sensoriale Definitiva (Ablazione Feature intera)
```python
def delete_feature(X, feature_idx):
    """Azzera una o piu' feature (sensore guasto). feature_idx: int o lista."""
    X_del = X.copy()
    idxs = [feature_idx] if np.isscalar(feature_idx) else list(feature_idx)
    for fi in idxs:
        X_del[:, fi] = 0.0
    return X_del

NOISE_LEVELS = [0.10, 0.25, 0.40]
print('Funzioni di noise definite OK | livelli:', NOISE_LEVELS)
```
#### 🌍 Analisi Macroscopica
Invece di iniettare confusione ad alta o media deviazione standard, qui si verifica il crollo della logica della rete nell'ipotesi in cui per l'intero arco temporale del test una delle antenne sia disabilitata dal computer di bordo o guasta. Azzera completamente il segnale matematico (simulato al centro dello StandardScaler a 0.0) verificando se le Network o gli Alberi decisionali (Random Forest) possiedano feature ridondanti utili per dedurre i parametri che sono venuti a mancare, aggirando il guasto hardware.
#### 🔬 Analisi Microscopica
- `idxs = [feature_idx] if np.isscalar(feature_idx) else list(feature_idx)`: Un elegantissimo controllo di flow in linea Python. Gestisce l'estrazione a singolo campo (se gli viene passato un int scalare) avvolgendolo istantaneamente in una Array di grandezza unitaria `[]`, ma se riceve già di base una matrice o list (di più features), la elabora castandola puramente a list iterabile.
- `for fi in idxs: X_del[:, fi] = 0.0`: Elabora lo spegnimento forzato. Esegue masking di colonna per il blocco numpy (`[:, fi]`) che indica la scansione di "tutte le righe `:` per quell'unica colonna `.fi`". Ponendo la Costante fissa a zero. 
- Definisce un Global var List `NOISE_LEVELS = [0.10, 0.25, 0.40]`, le % di distruzione che guideranno globalmente le iterazioni ed epoch del testing pipeline nelle prossime fasi.

---

## 10. Implementazione Architetturale dei Modelli di Classificazione

In questo blocco, a livello ingegneristico-software, si procede alla modellazione dell'Architettura pura. Si definiscono tre Wrapper-Functions con Framework differenti, pronte ad essere lanciate in serie. L'estrapolazione modulare permette ad un Orchestratore esterno di re-istanziare i modelli daccapo puliti per addestramenti successivi per ogni livello di rumore sperimentale, cancellando la fallimentare persistenza di training.

### Rete Neurale: Architettura Baseline Sensibile (La rete Nuda)
```python
def train_base_network(X_tr, y_tr, X_val, y_val, epochs=75, batch_size=32, verbose=0, random_state=42):
    tf.keras.utils.set_random_seed(random_state)
    model = Sequential([
        Input(shape=(X_tr.shape[1],)),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid'),
    ])
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(1e-3), metrics=['accuracy'])
    history = model.fit(X_tr, y_tr, epochs=epochs, batch_size=batch_size,
                        validation_data=(X_val, y_val), verbose=verbose)
    return model, history
```
#### 🌍 Analisi Macroscopica
Un benchmark minimo di riferimento. Questa è una Rete Neurale Feed-Forward classica (MLP - Multi Layer Perceptron), priva di qualsiasi strategia di "difesa". È concepita di proposito "sotto-armata" per soccombere di fronte al Feature-Noise: mancando di Dropout e Regolarizzatori, imparerà matematicamente a memoria i rumori casuali che le verranno passati (andando in overfitting). Così facendo, creeremo una base-line negativa a cui paragonare la rete armata.
#### 🔬 Analisi Microscopica
- `tf.keras.utils.set_random_seed(random_state)`: Prima di cominciare l'iterazione epocale, la rete neurale viene assemblata con pesi sinaptici casuali per sfuggire ai minimi locali. Questa direttiva TensorFlow inchioda il seme generativo: ad ogni chiamata, per via del Seed fisso, il Tensor core assegnerà gli stessi pesi a monte garantendo che la varianza tra run derivi dalla diversità di Rumore Dati, non dall'estrazione del caso stocastico. 
- `Sequential([...])`: Incapsula la rete stabilendo topologia serializzata. 
- `Input(shape=(X_tr.shape[1],))`: Conforma dimensionalmente la Layer 0 per connettere tante sinapsi entranti quanto sono le Features del dataset d'ingresso, che varia al mutamento matematico generato in X.
- `Dense(32, activation='relu')`: Livello Matematico Pienamente connesso. Esegue il prodotto matriciale dei pesi aggiungendo una costante bias $z = Wx + b$. Elabora poi lo scalare ottenuto tramite Activation ReLU (Rectified Linear Unit), che annichilisce le varianti negative convertendole a zero assoluto per formare il trigger non-lineare logico e mantenere un gradiente di addestramento solido senza problemi di Vanishing Gradient (sfaldamento del gradiente tipico delle vecchie reti Sigmoidi).
- `Dense(1, activation='sigmoid')`: Ultimo Output logico, comprime le intuizioni in 1 solo neurone. Modella lo stimolo sulla curva di S Sigmoide matematica, confinando l'output a una Probabilità % compressa perfettamente all'interno dei cardini logici [0, 1].
- `compile`: Attacca le specifiche su hardware GPU/CPU. Sviluppa un Calcolatore d'Errore Loss Function `binary_crossentropy` per quantizzare in maniera logaritmica lo scostamento dal target reale 1 o 0 (molto gravosa se ti sbagli con alta confidenza matematica). Si affida all'Optimizer `Adam` con passo (Learning Rate) fisso a `1e-3`. Adam usa vettori di tracciamento e momentum matematici adattivi, imparando per direzioni storiche e non solo gradienti bruti.
- `model.fit(..., validation_data=(X_val, y_val))`: Rilascia lo start al processo di retro-propagazione iterata. Carica blocchi pacchetto memoria di 32 in 32, completando le 75 epoche. Passandogli simultaneamente e testualmente il blocco `X_val`, Keras stamperà storicamente a fine di ogni Epoch in `.history` i valori metrici derivati della validazione isolata esterna, senza che il gradiente li usi per variare i pesi (i gradient backprop non scorrono nel Validation data pass). Restituisce storici e rete serializzata.

### Rete Neurale: Architettura PRO Resistente (La rete Armata)
```python
def train_pro_network(X_tr, y_tr, X_val, y_val, epochs=150, batch_size=32, verbose=0,
                      dropout_rate=0.2, l2_lambda=1e-4, patience=15, random_state=42):
    tf.keras.utils.set_random_seed(random_state)
    model = Sequential([
        Input(shape=(X_tr.shape[1],)),
        Dense(32, activation='relu', kernel_regularizer=l2(l2_lambda)),
        Dropout(dropout_rate),
        Dense(16, activation='relu', kernel_regularizer=l2(l2_lambda)),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid'),
    ])
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(1e-3), metrics=['accuracy'])
    classes = np.unique(y_tr)
    weights = compute_class_weight('balanced', classes=classes, y=np.array(y_tr))
    cw = {int(c): w for c, w in zip(classes, weights)}
    es = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    history = model.fit(X_tr, y_tr, epochs=epochs, batch_size=batch_size,
                        validation_data=(X_val, y_val), class_weight=cw,
                        callbacks=[es], verbose=verbose)
    return model, history
```
#### 🌍 Analisi Macroscopica
Si attrezza scientificamente una Rete per affrontare Outlier matematici e Noise Gaussiano estremo. Intenzionalmente implementati tre regolarizzatori, e innescato un sistema di EarlyStopping per proteggerne il fallimento temporale epocale. Questo garantisce prestazioni stabili e logiche in ambienti caotici ostili, superando ampiamente il modello Base di cui sopra in caso di test aggressivi.
#### 🔬 Analisi Microscopica
- `kernel_regularizer=l2(1e-4)`: Aggiunge al calcolo logico del Gradient Loss una penalizzazione vettoriale L2, (Ridge Regularization) limitando la rete. Vieta formalmente (aumentando la tassazione del loss in formula quadratica) all'ottimizzatore di generare pesi sinaptici colossali. Questo serve per non permettere all'AI di sbilanciarsi enormemente e ciecamente al momento in cui si appoggia su una feature che è stata corrotta appositamente da picchi Outlier enormi. 
- `Dropout(0.2)`: Tecnica stocastica. Spegne a livello randomico il 20% della fitta connessione neuronale all'inizio di ogni micro-batch ad ogni singola iterazione di derivazione epocale. Rende impossibile la co-dipendenza (laddove un neurone si adagia ciecamente copiando i pattern logici del compagno vicino) garantendo feature estrattive generaliste ben distribuite e robuste al rumore. In Test pass/Prediction, il modulo viene disattivato e ricablato integralmente. 
- `compute_class_weight('balanced', ...)`: Affronta l'emergenza di Asimmetria del Cluster Gamma vs Hadron. Applica una costante matematica di "Peso", costringendo l'errore derivativo a tassare molto, e assai più pesantemente la derivata quando questa commette un falso su target minoritari in proporzione (Se i Raggi Gamma sono il 30% dei casi originali totali, la loss function valuterà i fallimenti e penalizzerà i loss sui Gamma molto più duramente incrementandone il bias rispetto all'errore commesso sugli abbondanti Adroni per cui sono ricolmi di dataset). Riformatta a Dictionary in `cw = {int(c): w ...}` iterabile. 
- `es = EarlyStopping(...)`: A causa dei blocchi di disturbo, una rete armata necessita di più run per imparare i trend (`epochs=150`). Ma per prevenire addestramento passivo su fallimento protratto, instanzia l'osservatore logico callback `EarlyStopping`. Esso guarda asincronamente i calcoli di `val_loss`. Se e solamente se questo valore di loss non regredisce (quindi l'algoritmo va in stallo logico) per un range fisso di `patience=15` epoche consecutive, innesca l'interruzione di processo di back-propagation chiudendo l'apprendimento pre-termine, ed eseguendo re-rollback totale logico allo stato del Checkpoint ottimale intercettato pre-stallo (`restore_best_weights=True`). Parametro integrato a livello hardware Keras tramite la List pass `callbacks=[es]`.

### Random Forest (Apprendimento ad Albero Ensemble)
```python
def train_random_forest(X_tr, y_tr, n_estimators=200, random_state=42):
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    return rf

print('Funzioni di training definite OK')
```
#### 🌍 Analisi Macroscopica
Le Reti Neurali si incollano strettamente alle scale dei valori (Standardizzate) tramite gradienti differenziabili e continui. Questo le rende vulnerabili ad aberrazioni vettoriali fortissime. Un Random Forest invece valuta logica di spartiacque condizionale geometrico e non si affida a Gradienti continui, non venendo impressionato né da un Outlier a 5 Z-score né ad uno di 500.000 Z-score (viene spaccato lo spazio e fine del discorso logico). Qui implementato per valutare la dominanza robusta statistica ad Albero. 
#### 🔬 Analisi Microscopica
- `RandomForestClassifier(...)`: Istanzia il metamodello di classe Scikit-Learn.
- `n_estimators=200`: Assegna logica logaritmica. Dichiara all'Engine che non si addestrerà 1 Albero di classificazione debole a sé stante, ma piuttosto 200 istanze indipendenti con split diversificati a Bootstrap (il dataset perlustrato ad albero viene ridotto e scosso causalmente per feature precluse). Al momento del Judgment / Test predittivo, i 200 alberi procederanno ad emettere un'unica singola Media per votazione Democratica per giungere al target. 
- `n_jobs=-1`: Il forest è per design ed accezione informatica parallelo (embarassingly parallel problem). Un albero non ha vincoli di calcolo né dipendenze sinaptiche con il suo precedente, indi per cui settare questo parametro speciale di process logging a `-1` forzerà il Python Multiprocessing Engine a dirottare istantaneamente tutti i Core hardware, reali e logici della CPU dell'utente su cui ruota il test in forma Asincrona per concludere massimamente un calcolo che potrebbe durare decine di minuti altrimenti in single-thread. 

---

## 11. Estrazioni Probabilistiche, Formattazione Console e Matrici Visive

```python
def model_proba(model, X):
    """Probabilita' della classe positiva, unificata per Keras e sklearn."""
    if hasattr(model, 'predict_proba'):          # Random Forest (sklearn)
        return model.predict_proba(X)[:, 1]
    return np.asarray(model.predict(X, verbose=0)).ravel()  # Keras
```
#### 🌍 Analisi Macroscopica
Un "Helper" di programmazione pura di Ingegneria del Software (Polimorfismo). Scikit-learn e Keras TensorFlow espongono API logiche opposte per calcolare le probabilità (Keras usa `.predict()`, Scikit usa `.predict_proba()`). L'Orchestratore non saprà quale modello gli passiamo, dunque questa funzione astrae l'estrazione dati affinché sia agnostica al framework.
#### 🔬 Analisi Microscopica
- `if hasattr(model, 'predict_proba'):`: Metodo Pythonic per l'introspezione a Runtime (Duck Typing logico). Controlla il tipo base in memory, se la logica (la funzione attributo testata) è istanziata, sa inconfutabilmente di trovarsi innanzi ad un oggetto Scikit-Learn.
- `return model.predict_proba(X)[:, 1]`: Esegue le distribuzioni ed esegue Tensor Mask Slicing isolando alla componente `1` per estrapolare unicamente il calcolo percentuale destinato alla probabilità netta del solo Target "Gamma=1", ignorando il complementare per Target Zero per logiche binarie.
- `return np.asarray(model.predict(X, verbose=0)).ravel()`: Altrimenti, calcola Keras bypassando output a terminale (verbose). Incapsula con NdArray Numpy l'istanza generata ad array bi-dimensionale forzandolo ad assottigliamento su vettore piatto lineare 1D continuo richiamando la logica formale nativa `.ravel()`.

```python
def evaluate_network(model, history, X_test, y_test, model_name='Modello'):
    """Report completo per una rete Keras: curve train/val + confusion matrix + classification report."""
    print('\n' + '=' * 60)
    print(f' RISULTATI: {model_name}')
    print('=' * 60)
    proba = np.asarray(model.predict(X_test, verbose=0)).ravel()
    y_pred = (proba > 0.5).astype('int64')
    print(f'Test loss: {log_loss(y_test, proba):.4f} - Test accuracy: {accuracy_score(y_test, y_pred):.4f}')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history.history['accuracy'], label='Train')
    axes[0].plot(history.history['val_accuracy'], label='Validation')
    axes[0].set_title(f'{model_name} - Accuracy'); axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Accuracy')
    axes[0].legend(loc='lower right'); axes[0].grid(True)
    axes[1].plot(history.history['loss'], label='Train')
    axes[1].plot(history.history['val_loss'], label='Validation')
    axes[1].set_title(f'{model_name} - Loss'); axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss')
    axes[1].legend(loc='upper right'); axes[1].grid(True)
    plt.tight_layout(); plt.show()

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted'); plt.ylabel('True'); plt.title(f'Confusion Matrix — {model_name}')
    plt.show()

    print(f'AUC: {roc_auc_score(y_test, proba):.4f}')
    print('\nReport di Classificazione:')
    print(classification_report(y_test, y_pred, zero_division=0))
    print('=' * 60)
```
#### 🌍 Analisi Macroscopica
Un estrattore totale e formattatore per output generati da TensorFlow. Stampa le probabilità di Loss incrociata incapsulando report console testuale a cui affianca tracciati Matplotlib (curve ad Epoca e matrici visuali).
#### 🔬 Analisi Microscopica
- `y_pred = (proba > 0.5).astype('int64')`: Un array broadcasting Booleano che risolve un test logico simultaneo sull'intera array di 3000 predizioni Keras (che oscillano da float 0.00 a 1.00): Tutti i valori per il quale l'ipotesi è maggiore di soglia `.5` (50%) divengono `True`, il resto `False`. Con il Cast a `.astype('int64')` converte Pythonicamente il True al numero rigido `1`, e il False a numero rigido `0`.
- `log_loss(y_test, proba)`: Utilizza la cross entropy nativa derivata fornendo scarto numerico. Stampa Console con f-string interpolate `.4f` vincolando il rounding a 4 posizioni fisse decimali al console stdout.
- `axes[0].plot(history.history['accuracy'], label='Train')`: Raggiunge le dict properties allocate in `.history` nativo dell'oggetto passato al parametro per estrapolare la lista dei punteggi calcolati. Formatta la curva grafica di Training allacciandole l'etichetta Legend. Con questo doppio tracciato l'utente può determinare con i propri occhi al frame X il punto d'origine dell'overfitting in cui la rete collassa (dove la linea di Train interseca verso un'elevazione positiva continua e lineare la curva di Validation che crolla bruscamente in distacco esponenziale). 
- `cm = confusion_matrix(y_test, y_pred)`: Produce la tabulazione 2x2. Matrice crocevia ad allocazione tra Veri Positivi/Negativi per False Allarm (Sensore deviato rumore).
- `sns.heatmap(..., fmt='d', cmap='Blues')`: Sovrappone il plot in matrice colori Azzurrati gradati (Dal bianco all'Indaco profondo blu) scrivendovi i numerali (`d`) scartando notazioni scientifiche in forma tabulata e compatta per individuazione a vista degli errori di misclassificazione del Rumore che stiamo testando. 
- `roc_auc_score(...)` e `classification_report(...)`: Dichiara parametri metrici stringati complessi separandone a vista il limite 0 e format zero-division (soppressione bug di divisori aritmetici a blocco su F1 Score).

```python
def evaluate_rf(model, X_test, y_test, model_name='RF'):
    """Report per Random Forest: confusion matrix + classification report (niente curve)."""
    print('\n' + '=' * 60)
    print(f' RISULTATI: {model_name}')
    print('=' * 60)
    proba = model.predict_proba(X_test)[:, 1]
    y_pred = (proba > 0.5).astype('int64')
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
    plt.xlabel('Predicted'); plt.ylabel('True'); plt.title(f'Confusion Matrix — {model_name}')
    plt.show()
    print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')
    print(f'AUC: {roc_auc_score(y_test, proba):.4f}')
    print('\nReport di Classificazione:')
    print(classification_report(y_test, y_pred, zero_division=0))
    print('=' * 60)

print('Funzioni di valutazione definite OK')
```
#### 🌍 Analisi Macroscopica
Un estrattore equivalente ma logico esclusivo per il framework Scikit-Learn per il Random Forest.
#### 🔬 Analisi Microscopica
Riepiloga gran parte della medesima logica di sopra applicata, ma per difetto di design omette integralmente la genesi e chiamata `plt.subplots` poiché il RF esegue addestramento parallelo su Bagging ad Alberi distribuiti e NON tramite Discesa di Gradiente derivativa iterata e ciclica ad Epoche temporali (la quale si poteva misurare per calo e decrescita step dopo step in TensorFlow tracciandone la riga cartesiana). Viene applicato croma visivo in Mappa Colore verde Forestale `Greens` per l'estrazione visiva disconnessa dalla Neural Network azzurrata precedente.

---

## 12. Funzioni Strumentali e Orchestratore Generale del Profilo Sperimentale

L'intero ciclo vitale dello script converge qui per consentire ad iterazioni parametriche estese continue di esplorare l'impatto distruttivo rumore dopo rumore, al posto che codare test scriptati manuali isolati a hard code per 50 volte. 

```python
PALETTE = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#ff7f0e', '#8c564b', '#17becf']
YLIM_CONFRONTO = (0.5, 0.9)

def compute_metrics(model, X_test, y_test):
    proba = model_proba(model, X_test)
    y_pred = (proba > 0.5).astype(int)
    return {
        'accuracy':     accuracy_score(y_test, y_pred),
        'recall_macro': recall_score(y_test, y_pred, average='macro', zero_division=0),
        'f1_macro':     f1_score(y_test, y_pred, average='macro', zero_division=0),
        'auc':          roc_auc_score(y_test, proba),
    }
```
#### 🌍 Analisi Macroscopica
Una zona per configurazioni globali (Global Configuration State). Alloca stringhe e tavolozza esadecimale universali per plot, evitando discromia ad ogni esecuzione e standardizzando la Y per comparazioni non inficiate da "Zoom Autoscale". Il pacchetto metriche incapsula per ogni calcolo la sua matrice predittiva compatta restituendo logico Dictionary per assemblamento tabellare in frame Pandas successivi ad archiviazione su file.
#### 🔬 Analisi Microscopica
- `PALETTE` e `YLIM_CONFRONTO`: Formattatori Global. L'YLim definisce una Tupla parametrica di costrizioni per l'asse delle ordinate (50% e 90%), che eviterà di "zoommare ad arte" la rete base mostrandola con calo precipitoso ad una prima occhiata per difetto dimensionale e inganno ottico a display.
- `compute_metrics`: Utilizza le primitive Scikit metriche precedentemente integrate a blocco.
- `average='macro'`: Parametro critico di valutazione formale di robustezza. La precisione standard accademica per un dataset sbilanciato distorcerebbe le somme a favore della classe abbondante, la `macro` ignora deliberatamente il peso di densità campionaria calcolando Recall 0 per gli Adroni e Recall 1 per i Raggi per conto loro indipendentemente l'uno dall'altro, poi elaborando unicamente in chiusura una Media aritmetica secca pura che castiga e impone pesantissimi divari negativi al punteggio F1 derivato qualora il modello sacrifichi e dimentichi la classe minoritaria (un colpo letale frequente inflitto dal Noise borderline sperimentato prima).

```python
def plot_three_models(df_base, df_pro, df_rf, metric='accuracy', tipo='Feature noise', title=None):
    """Confronto NN base vs NN pro vs RF su una metrica, al variare del livello di rumore."""
    plt.figure(figsize=(10, 5.5))
    for df_, color, label, mark in [
        (df_base, PALETTE[0], 'NN Base', 'o'),
        (df_pro,  PALETTE[1], 'NN Pro',  's'),
        (df_rf,   PALETTE[2], 'RF',      '^'),
    ]:
        d = df_.sort_values('livello')
        plt.plot(d['livello'], d[metric], marker=mark, lw=2.2, color=color, label=label, markersize=8)
    plt.xlabel('Livello di rumore (%)'); plt.ylabel(metric.replace('_', ' ').upper())
    plt.ylim(*YLIM_CONFRONTO)
    plt.title(title or f'Confronto modelli — {tipo} ({metric})', fontweight='bold')
    plt.grid(alpha=0.3); plt.legend(frameon=True); plt.tight_layout(); plt.show()
```
#### 🌍 Analisi Macroscopica
Un costrutto specializzato per mettere in relazione i modelli. Plotterà tre distinte Line chart cartesiane di degrado (dallo 0% al 40%) in sovrapposizione cromatica unica per stabilire visualmente alla primissima estrapolazione ottica chi ha mantenuto l'Accuracy costante orizzontale e quale rete ha accusato le peggiori perdite di prestazione piegandosi verso il basso a precipizio.
#### 🔬 Analisi Microscopica
- Il costrutto `for df_, color, label, mark in [...]`: Un loop multi-elemento ad assegnazione Tuple che scansiona istantaneamente per l'iteratore fornendo all'Engine `plt.plot()` le singole informazioni metriche formattando colore, nome in etichetta legenda per riconoscimento (NN Base, NN Pro) ed applicando `markersize` ed accenti a geometria fissa (Circle, Square, Triangle).
- `d = df_.sort_values('livello')`: Utilissima protezione architetturale Dataframe di pandas che ri-alloca temporalmente la successione logica delle ascisse tracciate al Plot, ordinandole formalmente garantendo la salita da asse-X livello Rumore `0.0` e terminando in coerenza al `0.40`.
- `plt.ylim(*YLIM_CONFRONTO)`: Scompattazione della tupla via operatore asterisco puntatore (Splat `*`). Pone i limiti definitivi come accennato e dichiarati in globale poco fa (50%-90%).
- `metric.replace('_', ' ').upper()`: Applica formattazione String. Trasforma (e.g. `f1_macro` in `F1 MACRO`) in caps-lock estendendone la coerenza visiva per i layout Asse Y Label di lettura accademica finale.

```python
def plot_aggregate_per_model(model_name, dfs_by_noise, metric='accuracy'):
    """Una figura per modello: una linea per tipo di rumore (x = livello)."""
    plt.figure(figsize=(10, 6))
    for (noise_name, df_), color in zip(dfs_by_noise.items(), PALETTE):
        d = df_.sort_values('livello')
        plt.plot(d['livello'], d[metric], marker='o', lw=2, color=color, label=noise_name)
    plt.xlabel('Livello di rumore (%)'); plt.ylabel(metric.replace('_', ' ').upper())
    plt.ylim(*YLIM_CONFRONTO)
    plt.title(f'{model_name} — robustezza per tipo di rumore ({metric})', fontweight='bold')
    plt.grid(alpha=0.3); plt.legend(frameon=True, fontsize=9); plt.tight_layout(); plt.show()
```
#### 🌍 Analisi Macroscopica
Una trasposizione concettuale derivata dalla precedente, ma con focus orientato esclusivamente ad isolare e vagliare il fallimento del singolo archetipo (e.g. solo l'NN Pro). Questa variante scansionerà tutti e 5 i rumori distruttori creati per il test (Label noise casuali e Gaussiani ecc.) per calcolare quali tra le minacce logiche al modello di quell'algoritmo infliggessero il degrado o il crollo peggiore in assoluto in base allo scollamento asse.
#### 🔬 Analisi Microscopica
- `for (noise_name, df_), color in zip(dfs_by_noise.items(), PALETTE)`: Algoritmo `zip()` fondamentale. Data Dictionary in `items()` estrae Key stringhe (es: "Rumore Gaussiano") unendole per un incastro simmetrico in coppia combinata alla lista dei colori Globali PALETTE fornendo l'iterazione asincrona abbinata in Tuple a tre oggetti in tempo lineare rapido. 

```python
def run_noise_section(noise_name, make_data, levels=NOISE_LEVELS, evaluate=True, seed=42):
    rows = {'NN base': [], 'NN pro': [], 'RF': []}
    for lvl in [0.0] + list(levels):
        Xtr, ytr, Xval, yval, Xte, yte = make_data(lvl, seed)
        livello = int(round(lvl * 100))
        key = 'Baseline' if lvl == 0 else f'Rumore {livello}%'

        mb, hb = train_base_network(Xtr, ytr, Xval, yval, random_state=seed)
        mp, hp = train_pro_network(Xtr, ytr, Xval, yval, random_state=seed)
        mr = train_random_forest(Xtr, ytr, random_state=seed)

        if evaluate:
            evaluate_network(mb, hb, Xte, yte, f'NN base — {noise_name} {key}')
            evaluate_network(mp, hp, Xte, yte, f'NN pro — {noise_name} {key}')
            evaluate_rf(mr, Xte, yte, f'RF — {noise_name} {key}')

        for name, model in [('NN base', mb), ('NN pro', mp), ('RF', mr)]:
            rows[name].append({'modello': name, 'tipo_rumore': noise_name,
                               'livello': livello, **compute_metrics(model, Xte, yte)})

    dfs = {m: pd.DataFrame(rows[m]).sort_values('livello').reset_index(drop=True) for m in rows}
    print(f'\n>>> Confronto modelli — {noise_name}')
    plot_three_models(dfs['NN base'], dfs['NN pro'], dfs['RF'], 'accuracy', noise_name)
    plot_three_models(dfs['NN base'], dfs['NN pro'], dfs['RF'], 'f1_macro', noise_name)
    return dfs

print('Driver e funzioni di plotting definiti OK')
```
#### 🌍 Analisi Macroscopica
Il vero è proprio **Motore dell'Esperimento (Orchestratore)**. Questo ciclo massivo compie tutto lo sforzo di ingegneria iterativa per noi. Riceve una delle strategie distruttive create ai blocchi 9, poi imposta una Baseline nuda (0%), avvia sequenzialmente tutti i training sui 3 archetipi e le relative predizioni grafiche di Output Console. Prosegue ciclando per step infettivi del 10, 25 e 40%, raccogliendone formalmente per ogni run tutte le deduzioni metriche accodate per ritornarle archiviate in vocabolario pandas ai plot globali di fine testing, e plottandone le preview a caldo ad interruzione di ciclo su accuracy e F1 Score.
#### 🔬 Analisi Microscopica
- `rows = {'NN base': [], 'NN pro': [], 'RF': []}`: Pre-allocazione di memoria di Dictionary di Liste per archiviazione locale. 
- `for lvl in [0.0] + list(levels):`: Modulatore di rateo rumore. Addiziona List concatenation logica per anteporre la fase Zero incontaminata (Ground truth logico) che non va sprecata a parametri successivi.
- `make_data(lvl, seed)`: Funzione generatrice polimorfica di alto livello architetturale passata come callback in invocazione. Distrugge temporaneamente a run isolate i subset di Testing Validation in base alle funzioni di Noise scritte in blocchi 9 passandogli i Ratei e l'entropia logica randomica Seed per bloccarne le fluttuazioni, scaricandone i frame nelle tuple ad incastro multiple `Xtr, ytr, ...`.
- `livello = int(round(lvl * 100))`: Elaborazione dell'etichettatrice numerale percentuale intera per chiavi.
- Esegue in forma asincrona la chiamata ai macro framework `train_base_network` eccetera riaddestrando formalmente macchine nuove ex-novo prive di inquinamento di back-progression persistente causato dai cicli precedenti a livelli più lievi, scongiurando il catastrofico data-leakage e bias epocale continuo. 
- Istruzione Unpacking Dict Logico Python `{**compute_metrics(...)}`: Estrae ed appiattisce le chiavi dizionarie ritornate dal wrapper metrico, saldandone gli indicatori `accuracy: x, auc: y` istantaneamente allo schema righe del Dictionary rows. 
- `dfs = {m: pd.DataFrame(rows[m]).sort_values... reset_index(drop=True)}`: Sconvolge la memoria allocata del loop formativo, gettando via i dictionary grezzi tramutandone con dictionary comprehension l'identificativo in una Dataframe solido tabularizzato Pandas per ogni famiglia logica, ri-ordinato coerentemente dalla gravità e sbiancato dell'indice di tracciato casuale per azzeramento ad `reset_index(drop=True)`. Infine stampa a video le due varianti (Accuracy contro F1). 

---

## 13. Esecuzione del Test e Loop degli Esperimenti Noise

In questa macro sequenza il computer dà via ufficialmente allo Start alle sperimentazioni in cui esegue i test e fa crollare formalmente la stabilità delle reti per quantizzarne a livello empirico la solidità a bordo di un computer astronomico reale. 

### Primo Passo: Valutazione Baseline
```python
print('>>> BASELINE su dati puliti')
m_base_clean, h_base_clean = train_base_network(X_train_scaled, y_train, X_val_scaled, y_val, random_state=seed)
evaluate_network(m_base_clean, h_base_clean, X_test_scaled, y_test, 'NN base — Baseline pulita')

m_pro_clean, h_pro_clean = train_pro_network(X_train_scaled, y_train, X_val_scaled, y_val, random_state=seed)
evaluate_network(m_pro_clean, h_pro_clean, X_test_scaled, y_test, 'NN pro — Baseline pulita')

rf_clean = train_random_forest(X_train_scaled, y_train, random_state=seed)
evaluate_rf(rf_clean, X_test_scaled, y_test, 'RF — Baseline pulita')

fi = pd.Series(rf_clean.feature_importances_, index=feature_names).sort_values(ascending=False)
plt.figure(figsize=(9, 5))
sns.barplot(x=fi.values, y=fi.index, palette='viridis')
plt.title('RF — Feature Importance (dati puliti)', fontweight='bold')
plt.xlabel('Importanza media'); plt.tight_layout(); plt.show()

# Dizionario che raccoglie il df canonico di ogni rumore, per il confronto aggregato finale
AGG = {'NN base': {}, 'NN pro': {}, 'RF': {}}
```
#### 🌍 Analisi Macroscopica
Esegue una run a livello teorico per saggiare le reti sul Dataset standard perfettamente intonso ed idealizzato (che nella realtà astronomica di un telescopio non esiste per interferenza cosmica) come punto di riferimento di accuratezza da poter scalfire nei prossimi blocchi. Innesca dal Random Forest un'operazione analitica fondamentale per il Data Scientist: la misurazione ed estrazione di "Feature Importance" (che indica al team quale sensore domini di più per decidere l'identità tra Raggio o particella) estrapolandone il gradiente di valore matematico per poi graficarlo in ordine di peso.
#### 🔬 Analisi Microscopica
- Lancia i 3 Training standard fornendogli gli `X_train_scaled` originali, con parametri invariati bloccati sui seed.
- `fi = pd.Series(rf_clean.feature_importances_, index=feature_names)`: Accede alle variabili statistiche dell'albero di addestramento Random Forest. Tramite calcolo della "Gini Impurity" calata di nodo in nodo durante la genesi casuale degli alberi, SciKit-Learn espone questo logico attributo pre-calcolato `.feature_importances_`. Traccia in forma lineare e unidirezionale il vettore a colonna Series, indicizzandovi ai lati con il list nomi (str) memorizzato nelle fasi 8 del preprocessing originario, incrociandone i frame e formattando con ordinamento dal valore di purezza scaturito più grande al meno influente (`sort_values(ascending=False)`).
- Modella un plot Matplotlib per visualizzare in grafico a barre (`sns.barplot(...)`) cromatico, il gerarchico predominante tra le varianti. Alloca poi ufficialmente a Memoria il Macro Dizionario Totale di Accumulo Progetto (AGG), atto a contenere e incapsulare tutti gli esiti dei 5 test condotti, salvandoli per i macro-comparatori delle heat-map previste per il capitolo finale. 

### Serie dei Rumori e Loop degli Stress Test
```python
def make_fn(lvl, s):
    Xtr = X_train_scaled if lvl == 0 else add_gaussian_noise(X_train_scaled, noise_level=lvl, seed=s)
    return Xtr, np.array(y_train), X_val_scaled, np.array(y_val), X_test_scaled, np.array(y_test)

dfs_fn = run_noise_section('Feature noise', make_fn, seed=seed)
for m in AGG:
    AGG[m]['Feature noise'] = dfs_fn[m]
```
- Esegue formalmente Run (Gaussiano Totale) e stocca il ritorno formattato al loop generico di stoccaggio `AGG` sfruttando iterazione a Dictionary Keys (`for m in AGG`).

```python
SIGMA_FN_LOC = 0.35
top_feats_fnloc = list(fi.index)                 
FNLOC_LEVEL_TO_K = {0: 0, 10: 1, 25: 3, 40: 5}   

def make_fn_local(lvl, s):
    k = FNLOC_LEVEL_TO_K[int(round(lvl * 100))]
    idxs = [feature_names.index(f) for f in top_feats_fnloc[:k]]
    Xtr = X_train_scaled if not idxs else add_gaussian_noise(X_train_scaled, noise_level=SIGMA_FN_LOC, seed=s, cols=idxs)
    print(f'  livello {int(round(lvl*100))}% -> rumore sigma={SIGMA_FN_LOC} su {k} feature: {top_feats_fnloc[:k]}')
    return Xtr, np.array(y_train), X_val_scaled, np.array(y_val), X_test_scaled, np.array(y_test)

dfs_fn_local = run_noise_section('Feature noise localizzato', make_fn_local, seed=seed)
for m in AGG:
    AGG[m]['Feature noise localizzato'] = dfs_fn_local[m] 
```
#### 🌍 Analisi Macroscopica
Un'intrusione logica di alto livello. Piuttosto che danneggiare il rumore per l'intero set (in maniera aleatoria), il computer estrae l'elenco stilato dall'albero di Random Forest (il "Ranking" di importanza estratto 1 blocco prima), e va selettivamente ad azzoppare e mandare in avaria (Gaussiana di magnitudo fissa sigma 0.35) unicamente la PRIMA ed unica colonna sensore considerata la più affidabile dalla macchina se la magnitudo passata dal Driver è del 10%. Se invece il Driver comanda tasso di Distruzione Estrema 40%, il virus infetterà e crittograferà contemporaneamente le PRIMISSIME 5 Antenne Sensore di vitale importanza distorcendone la matrice ad alta intensità per far collassare la previsione di classe.
#### 🔬 Analisi Microscopica
- Genera dizionario di Hash `FNLOC_LEVEL_TO_K` in grado di traslare rapidamente con match perfetto le percentuali fornite in intero `10`, alla quota di estrapolazione per slice `k=1`. 
- `idxs = [feature_names.index(f) for f in top_feats_fnloc[:k]]`: Scompone con list comprehension le key string di nome (es: `fSize`) ed incrocia nell'elenco array statico l'indice numerico intero che il Tensor compiler Keras necessita per mappare le allocazioni vettoriali durante la distorsione in Noise function. Evoca orchestrazione ed archivia nel Macro.

```python
def make_ln(lvl, s):
    ytr = np.array(y_train) if lvl == 0 else add_label_noise_random(y_train, noise_level=lvl, seed=s)
    return X_train_scaled, ytr, X_val_scaled, np.array(y_val), X_test_scaled, np.array(y_test)

dfs_ln = run_noise_section('Label noise random', make_ln, seed=seed)
for m in AGG:
    AGG[m]['Label noise random'] = dfs_ln[m]
```
- Esegue formalmente Run (Rumore Inversione Label puramente stocastico Causale a bersagli non selettivi). 

```python
def make_lnb(lvl, s):
    ytr = np.array(y_train) if lvl == 0 else add_label_noise_borderline(X_train_scaled, y_train, noise_level=lvl, seed=s)
    return X_train_scaled, ytr, X_val_scaled, np.array(y_val), X_test_scaled, np.array(y_test)

dfs_lnb = run_noise_section('Label noise borderline', make_lnb, seed=seed)
for m in AGG:
    AGG[m]['Label noise borderline'] = dfs_lnb[m]
```
- Esegue formalmente Run (Etichette Ostili a Inversione Borderline dei target calcolati geometricamente per far sbandare la superficie delimitativa d'azione logica separante delle macchine logaritmiche NN).

```python
def make_missing(strategy):
    def _f(lvl, s):
        if lvl == 0:
            return X_train_scaled, np.array(y_train), X_val_scaled, np.array(y_val), X_test_scaled, np.array(y_test)
        Xtr_m = add_missing_values(X_train_scaled, missing_rate=lvl, seed=s)
        Xtr_i = SimpleImputer(strategy=strategy).fit_transform(Xtr_m)
        return Xtr_i, np.array(y_train), X_val_scaled, np.array(y_val), X_test_scaled, np.array(y_test)
    return _f

print('### Missing Values — imputazione MEDIA ###')
dfs_miss_mean   = run_noise_section('Missing values (media)',   make_missing('mean'),   seed=seed)
print('### Missing Values — imputazione MEDIANA ###')
dfs_miss_median = run_noise_section('Missing values (mediana)', make_missing('median'), seed=seed)

for m in AGG:
    AGG[m]['Missing values'] = dfs_miss_median[m]  # variante canonica = mediana
```
- Il test missing value viene attuato due volte separate tramite la Closure generatrice iterabile Python (`def _f`) per intercettarne un parametro astratto differenziatore della strategia per i Missing: Un frame di calcolo usa Media classica aritmetica, il successivo applica logicamente sostituzione a ripiego in Mediana posizionale, la quale diventerà formalmente per l'analista la canonica stoccata stabilmente alle finali nel contenitore `AGG`.

```python
for m in ['NN base', 'NN pro', 'RF']:
    plt.figure(figsize=(8, 5))
    plt.plot(dfs_miss_mean[m]['livello'],   dfs_miss_mean[m]['accuracy'],   marker='o', lw=2, label='Imputazione media')
    plt.plot(dfs_miss_median[m]['livello'], dfs_miss_median[m]['accuracy'], marker='s', lw=2, label='Imputazione mediana')
    plt.xlabel('Livello di rumore (%)'); plt.ylabel('Accuracy')
    plt.ylim(*YLIM_CONFRONTO)
    plt.title(f'Missing values: media vs mediana — {m}', fontweight='bold')
    plt.grid(alpha=0.3); plt.legend(); plt.tight_layout(); plt.show()
```
- Plotta, per scopi di analisi puramente tecnica ed intra-model, per visualizzare quale tra `Media` e `Mediana` ha permesso ai classificatori di subire scossoni limitati. Mostrando graficamente il cedimento per Accuracy con Linea marcata da forme `Square e Circle` a sovrapposizione cromatica differenziale.

```python
def make_outlier(strategy):
    def _f(lvl, s):
        if lvl == 0:
            return X_train_scaled, np.array(y_train), X_val_scaled, np.array(y_val), X_test_scaled, np.array(y_test)
        Xtr = add_outliers(X_train_scaled, outlier_rate=lvl, magnitude=5.0, seed=s)
        if strategy == 'winsorize':
            Xtr = winsorize_features(Xtr)
        elif strategy == 'impute':
            Xtr = impute_outliers(Xtr)
        # 'leave' -> nessuna correzione
        return Xtr, np.array(y_train), X_val_scaled, np.array(y_val), X_test_scaled, np.array(y_test)
    return _f

print('### Outlier — LASCIATI ###')
dfs_out_leave = run_noise_section('Outlier (lasciati)',        make_outlier('leave'),     seed=seed)
print('### Outlier — WINSORIZZAZIONE ###')
dfs_out_wins  = run_noise_section('Outlier (winsorizzazione)', make_outlier('winsorize'), seed=seed)
print('### Outlier — IMPUTAZIONE ###')
dfs_out_imp   = run_noise_section('Outlier (imputazione)',     make_outlier('impute'),    seed=seed)

for m in AGG:
    AGG[m]['Outlier'] = dfs_out_wins[m]  # variante canonica = winsorizzazione

for m in ['NN base', 'NN pro', 'RF']:
    plt.figure(figsize=(8, 5))
    plt.plot(dfs_out_leave[m]['livello'], dfs_out_leave[m]['accuracy'], marker='o', lw=2, label='Lasciati')
    plt.plot(dfs_out_wins[m]['livello'],  dfs_out_wins[m]['accuracy'],  marker='s', lw=2, label='Winsorizzazione')
    plt.plot(dfs_out_imp[m]['livello'],   dfs_out_imp[m]['accuracy'],   marker='^', lw=2, label='Imputazione')
    plt.xlabel('Livello di rumore (%)'); plt.ylabel('Accuracy')
    plt.ylim(*YLIM_CONFRONTO)
    plt.title(f'Outlier: strategie a confronto — {m}', fontweight='bold')
    plt.grid(alpha=0.3); plt.legend(); plt.tight_layout(); plt.show()
```
- Valuta e cicla le logiche difensive verso i picchi (Outlier). Crea variante ad ignoranza totale `leave` misurando l'abbattimento estremo del classificatore a rete Neurale, che senza tecniche prenderà a memoria il picco alieno collassando ai test. Poi applica mitigazione statica con `winsorize` per appiattirlo ai margini del 95 percentile, ed infine per Ablazione ed `impute` rimpiazzando integralmente e a priori il calore. Formalizza Winsorizzazione a parametro di canone Archiviandolo globalmente per l'AGG e graficando il cedimento con `Triangoli, Quadri, e Cerchi`.

```python
top_feats = list(fi.index)                  # feature ordinate per importanza (desc)
LEVEL_TO_K = {0: 0, 10: 1, 25: 3, 40: 5}    # livello% -> n. feature rimosse

def make_missfeat(lvl, s):
    k = LEVEL_TO_K[int(round(lvl * 100))]
    idxs = [feature_names.index(f) for f in top_feats[:k]]
    Xtr  = delete_feature(X_train_scaled, idxs) if idxs else X_train_scaled
    Xval = delete_feature(X_val_scaled,   idxs) if idxs else X_val_scaled
    Xte  = delete_feature(X_test_scaled,  idxs) if idxs else X_test_scaled
    print(f'  livello {int(round(lvl*100))}% -> rimosse {k} feature: {top_feats[:k]}')
    return Xtr, np.array(y_train), Xval, np.array(y_val), Xte, np.array(y_test)

dfs_mf = run_noise_section('Missing feature', make_missfeat, seed=seed)
for m in AGG:
    AGG[m]['Missing feature'] = dfs_mf[m]
```
- Esegue Formalmente Test di privazione di Colonna Assoluta (Sensore fulminato offline), agganciandovi ancora il sistema astuto di bersagliamento che devia e scollega fisicamente il Top-Performing sensor individuato tramite ranking Gini Impurity (L'attributo RandomForest ad importanza calcolato 5 blocchi addietro). Registra metriche globali e logga l'esito.

---

## 14. Reportistica Aggregata Finale

Ora che i 52 training di rete neurale (più centinaia di random forest) sono conclusi e valutati internamente, i dataframe locali collassano in aggregazioni visive per estrarre la statistica scientifica comparata in modo esplicativo per il Deployment Finale del prodotto da immettere per il Telescopio.

```python
for m in ['NN base', 'NN pro', 'RF']:
    plot_aggregate_per_model(m, AGG[m], 'accuracy')
    plot_aggregate_per_model(m, AGG[m], 'f1_macro')
```
- Utilizza il Plottatore configurato prima che richiama l'architettura per ogni Modello estrapolando la Mappa globale dei 5 test eseguiti mostrando una tela con tutte le corruzioni e cedimenti generati tracciandole comparativamente a vari colori sul calo della purezza Accuracy e sull'F1 Score (molto più veritiero siccome punisce fortemente i classificatori che diventano ciechi dimenticandosi della presenza della Classe dei Raggi Gamma solo perché scemata).

```python
def get_val(df, livello, metric):
    r = df[df['livello'] == livello]
    return float(r[metric].iloc[0]) if len(r) else float('nan')

summary_rows = []
for m in ['NN base', 'NN pro', 'RF']:
    for noise, df_ in AGG[m].items():
        for metric in ['accuracy', 'f1_macro', 'auc']:
            v0, v40 = get_val(df_, 0, metric), get_val(df_, 40, metric)
            summary_rows.append({'Modello': m, 'Rumore': noise, 'Metrica': metric,
                                 '@0%': f'{v0:.4f}', '@40%': f'{v40:.4f}', 'Delta': f'{v40 - v0:+.4f}'})

df_summary = pd.DataFrame(summary_rows)
print('=== TABELLA RIASSUNTIVA — Baseline (0%) vs Caso Peggiore (40%) ===')
print(df_summary.to_string(index=False))
df_summary.to_csv('risultati_summary.csv', index=False)
print('\nSalvata in risultati_summary.csv')
```
#### 🌍 Analisi Macroscopica
Un loggatore per conservazione di massa e revisione numerica non distorta da grafici visivi ad occhio. Mappa in una griglia per righe e colonne la differenza precisa tra le prestazioni perfette di baseline 0 e quelle spaventose sotto la pressione del virus dati massimo applicato nella suite di calcolo (il fatidico livello limite 40% di avversità per cui abbiamo misurato l'innesco catastrofico dei modelli, per cui molti classificatori cadono rovinosamente nel range random del 0.50 ad una predizione).
#### 🔬 Analisi Microscopica
- Scorre il frame per prelevare formalmente per incrocio tabellare `df_` al level `livello=0` che contiene parametrizzazione pura ideale registrando e salvando lo scorporo logico `v0`, ripetendone l'azione per estrapolazione per il blocco al limite `v40`.
- Crea ed impacchetta List di dizionari associati contenenti le stringhe logiche `Modello e Nome test` seguite dai calcoli Floating formattati a cifre con offset decimale e apponendovi la sottrazione aritmetica tra Baseline e Disastro (Esistita come `Delta = v40-v0` formattato a simbolo numerico + o - per `+.4f`). Ed esportando nativamente il file locale al computer in CSV aperto formale e privo di Index id irrilevante.

```python
noises = list(AGG['NN base'].keys())
models = ['NN base', 'NN pro', 'RF']

# 1) Heatmap accuracy @40% (modello x rumore)
mat = np.array([[get_val(AGG[m][n], 40, 'accuracy') for n in noises] for m in models])
plt.figure(figsize=(11, 4))
sns.heatmap(mat, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0.5, vmax=1.0,
            xticklabels=noises, yticklabels=models)
plt.title('Accuracy nel caso peggiore (40%) — modello x rumore', fontweight='bold')
plt.xticks(rotation=20, ha='right'); plt.tight_layout(); plt.show()

# 2) Degradazione (acc@0 - acc@40)
x = np.arange(len(noises)); width = 0.25
fig, ax = plt.subplots(figsize=(12, 5))
for i, m in enumerate(models):
    deg = [get_val(AGG[m][n], 0, 'accuracy') - get_val(AGG[m][n], 40, 'accuracy') for n in noises]
    ax.bar(x + i * width, deg, width, label=m, color=PALETTE[i])
ax.set_xticks(x + width); ax.set_xticklabels(noises, rotation=20, ha='right')
ax.set_ylabel('Calo di accuracy (0% -> 40%)')
ax.set_title('Robustezza: degradazione per tipo di rumore', fontweight='bold')
ax.legend(); ax.grid(axis='y', alpha=0.3); plt.tight_layout(); plt.show()

# 3) Ranking robustezza complessiva (accuracy media @40% sui rumori)
rank = {m: float(np.mean([get_val(AGG[m][n], 40, 'accuracy') for n in noises])) for m in models}
plt.figure(figsize=(7, 4))
plt.bar(list(rank.keys()), list(rank.values()), color=PALETTE[:3])
plt.ylabel('Accuracy media @40%'); plt.ylim(0.5, 1.0)
plt.title('Ranking robustezza complessiva (media sui 7 esperimenti)', fontweight='bold')
plt.tight_layout(); plt.show()
```
#### 🌍 Analisi Macroscopica
Questi tre tracciati sono le pietre miliari per il management decisionale. Piuttosto che far vedere 300 righe di csv, traducono il dolore della degradazione per visual analysis istantaneo (Es. la prima Heapmap renderà le righe Rosse se il Classificatore Base esplode, la Rete Pro ingiallisce per flessione e il Random forest permane in luce Verde solida confermando solidità strutturale da Albero al crollo dei segnali). L'ultimo plot dichiara tramite Media di calcolo globale chi ha vinto di diritto la Corona formale della robustezza totale su scala aggregata.
#### 🔬 Analisi Microscopica
- L'Heapmap estrapola list comprehension List multidimensionale a base List `[[get_val...]]` e la inserisce in array NumPy originario per tracciato matrice di Heatmap con min 50 max 1.0 (RdYlGn).
- Per il Bar Chart di degradazione, Matplotlib non ha layout formali per grafici affiancati "Dodged" per cui bisogna istruire la logica del piano cartesiano Numpy e sfasare il posizionamento col fattore iterativo di addizione laterale `(x + i * width)` che decostruisce il cluster di bar plottando i 3 archetipi per ogni tipologia di test logico.
- Traccia in bar singolo (Ranking Bar Chart) la chiusura estraendo dizionari di purezze per via asincrona `.mean()` che impone un rateo finale per determinare quanto una Random Forest sopravviva solidamente in formati multi-test al discapito di una rete feed-forward indifesa per calcolo statistico del limite del peggior test `40%`.

```python
X_fn40 = add_gaussian_noise(X_train_scaled, noise_level=0.40, seed=seed)
mb40, _ = train_base_network(X_fn40, y_train, X_val_scaled, y_val, random_state=seed)
mp40, _ = train_pro_network(X_fn40, y_train, X_val_scaled, y_val, random_state=seed)
rf40    = train_random_forest(X_fn40, y_train, random_state=seed)

pairs = [
    ('NN base pulito', m_base_clean, 'blue',  '-'),
    ('NN pro pulito',  m_pro_clean,  'green', '-'),
    ('RF pulito',      rf_clean,     'red',   '-'),
    ('NN base 40%FN',  mb40,         'blue',  '--'),
    ('NN pro 40%FN',   mp40,         'green', '--'),
    ('RF 40%FN',       rf40,         'red',   '--'),
]
plt.figure(figsize=(10, 7))
for name, model, c, ls in pairs:
    sc = model_proba(model, X_test_scaled)
    fpr, tpr, _ = roc_curve(y_test, sc)
    plt.plot(fpr, tpr, color=c, ls=ls, lw=2, label=f'{name} (AUC={roc_auc_score(y_test, sc):.3f})')
plt.plot([0, 1], [0, 1], 'k--', alpha=0.4)
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('ROC — Dati puliti vs 40% Feature Noise', fontweight='bold')
plt.legend(loc='lower right', fontsize=8); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()
```
#### 🌍 Analisi Macroscopica
Nel machine learning la Curva **ROC** (Receiver Operating Characteristic) è forse il grafico più autorevole per certificare la qualità intrinseca del modello (indipendente dalla soglia >50% imposta manualmente). Traccia l'equilibrio tra il richiamo positivo esatto (Veri Positivi) e gli Allarmi falsi scattati in eccesso (Falsi Positivi). Più la linea abbraccia l'angolo a sbalzo estremo sinistro alto e si innalza distaccando, migliore è il classificatore. In questa griglia finale tracciamo contemporaneamente le versioni allenate sui Dati Perfetti e quelle allenate sui Dati al 40% di Guasto per vedere otticamente la "schiacciata" involutiva della qualità intrinseca che si adagia contro la diagonale neutra per l'area 0.5 (Tiro di dadi cieco e fallimentare casuale).
#### 🔬 Analisi Microscopica
- Crea artificialmente un ambiente infettato al 40% `X_fn40` eseguendovi daccapo le chiamate al generatore `train_random_forest` e `train_base_network` per salvarle provvisoriamente a blocchi isolati locali (`mb40, mp40, ecc`).
- Crea matrice relazionale `pairs` in cui abbina Testi identificativi, Oggetto Memory puntatore Modello Keras (e.g. `m_base_clean` generato mille righe addietro all'innesco di run 0.0), e imposta stili visivi assegnandovi la riga tracciata compatta `-` (Per le reti immacolate perfette) e `--` a tratteggio (per la degradazione corrotta ad infezione 40).
- Nel Loop: Estrae le probabilità pure `model_proba()`, la cui formula viene elaborata dalla funzione Scikit Learn `roc_curve` generando le Array Vettoriali metriche corrispondenti a tasso True-Positive (`tpr`) e False-Positive (`fpr`). 
- Stampa linea Cartesiana con `plt.plot([0,1],[0,1])` (la diagonale tratteggiata cieca neutrale del tiro randomico a moneta) ad Alpha sfuocato. Ed unisce in calcolo d'Area Geometrica il numero AUC (Area Under the Curve) allacciato ai formattatori Legend. 

---

## 15. Controlli Multi-Seed: L'Invarianza alla Genesi Stocastica

Questo macro blocco applica rigorosamente un metodo scientifico. Se il classificatore ha ceduto a quel modo ai test, è stato un puro accidente dettato dalla fortuna/sfortuna dell'inizializzazione casuale dei pesi sinaptici Keras Neurali per quell'avvio del codice, o è davvero un sintomo strutturale della rete inabile al Telescopio? 

```python
# --- ESTENSIONE: controllo multi-seed (feature noise) -------------------------
def run_multiseed(noise_name, make_data, seeds=(42, 7, 123, 2024, 99), levels=NOISE_LEVELS):
    rec = []
    for sd in seeds:
        for lvl in [0.0] + list(levels):
            Xtr, ytr, Xval, yval, Xte, yte = make_data(lvl, sd)
            mb, _ = train_base_network(Xtr, ytr, Xval, yval, random_state=sd)
            mp, _ = train_pro_network(Xtr, ytr, Xval, yval, random_state=sd)
            mr    = train_random_forest(Xtr, ytr, random_state=sd)
            for name, model in [('NN base', mb), ('NN pro', mp), ('RF', mr)]:
                rec.append({'seed': sd, 'modello': name, 'livello': int(round(lvl * 100)),
                            **compute_metrics(model, Xte, yte)})
    return pd.DataFrame(rec)

def plot_multiseed(df, metric='accuracy', noise_name=''):
    plt.figure(figsize=(10, 5.5))
    for name, color, mark in [('NN base', PALETTE[0], 'o'), ('NN pro', PALETTE[1], 's'), ('RF', PALETTE[2], '^')]:
        g = df[df['modello'] == name].groupby('livello')[metric]
        plt.errorbar(g.mean().index, g.mean().values, yerr=g.std().values,
                     marker=mark, lw=2, capsize=4, color=color, label=name)
    plt.xlabel('Livello di rumore (%)'); plt.ylabel(metric.replace('_', ' ').upper())
    plt.ylim(*YLIM_CONFRONTO)
    plt.title(f'Controllo multi-seed — {noise_name} ({metric})', fontweight='bold')
    plt.grid(alpha=0.3); plt.legend(); plt.tight_layout(); plt.show()

SEEDS_CTRL = (42, 7, 123, 2024, 99)
print(f'### Controllo multi-seed su Feature noise | seed = {SEEDS_CTRL} ###')
df_ms = run_multiseed('Feature noise', make_fn, seeds=SEEDS_CTRL)
plot_multiseed(df_ms, 'accuracy', 'Feature noise')
plot_multiseed(df_ms, 'f1_macro', 'Feature noise')

tab_ms = (df_ms.groupby(['modello', 'livello'])[['accuracy', 'f1_macro', 'auc']]
                .agg(['mean', 'std']).round(4))
display(tab_ms)
```
#### 🌍 Analisi Macroscopica
Ripetiamo ciecamente il test per N volte separate usando Inneschi Casuali Seed differenti (Invece che l'esclusivo seed fisso n=42, iteriamo per numeri alieni tra cui n=7, n=123 e così via). Ogni volta i Pesi delle reti e l'estrazione Gaussiana varieranno. Al termine non viene riportata una linea di Accuracy piatta, bensì una Linea che contiene delle "Barre di Errore" verticali, indicanti matematicamente l'Intervallo di Confidenza (o Deviazione Standard) dettato e causato dal fluttuare del caso stocastico. Più la barra di incertezza verticale è ristretta, più l'algoritmo analizzato è scientificamente stabile al guasto e affidabile per il progetto.
#### 🔬 Analisi Microscopica
- Crea un Ciclo Avvolgente `for sd in seeds:` attorno al loop originale, spingendo la pipeline a rieseguire asincronamente un ammasso incalcolabile logico per train reiterati a framework (Addestrerà per noi nel silenzio del background dozzine di reti passandogli la `random_state=sd` preclusiva e differente bloccata istante per istante).
- Costruisce list Pandas e ritorna DF compatto senza interpellare print metrici. 
- L'estrattore Grafico instanzia raggruppamento per Livelli d'impatto sfruttando l'engine SQL-like di pandas `.groupby('livello')[metric]`. Esegue poi su di esso il blocco costrutto visivo di Matplotlib `plt.errorbar(..)` a cui assegna per i centroidi asse y la media aritmetica dell'array multi-test `.mean().values`, a cui affianca la variabile d'ingombro Error (L'asta) invocando la Varianza Standard Dev tramite attributo `.std().values` per indicarne lo spike d'incertezza al generatore aleatorio causato col variare dal rumore `sd` bloccando le teste orizzontali ad ampiezza fissa `capsize=4`.
- Per concludere, riassume a terminale Python la Dataframe estrattiva incrociando l'allocazione multipla SQL `.agg(['mean', 'std'])` per evidenziarne tabulati algebrici.

---

## 16. K-Means: Classificatore al Buio (Senza Etichette)

In questo macro-blocco il Machine Learning smette di fare training (Supervised ML, Rete Neurale e Alberi). Cambia branchia, affrontando la disciplina definita **Apprendimento Non Supervisionato (Unsupervised)** in cui gli passiamo un'intera matrice di feature senza target e gli chiediamo alla cieca: "Raggruppa le misurazioni per similarità logica vettoriale matematica che riscontri da te, scoprendo eventuali differenze da solo". E testiamo questo approccio contro lo sporco del rumore gaussiano. 

### Elaborazione K-Optimum e Silhouette
```python
ks = list(range(2, 7))
inertias, silhouettes = [], []
for kk in ks:
    km = KMeans(n_clusters=kk, n_init=10, random_state=seed).fit(X_train_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_train_scaled, km.labels_, sample_size=2000, random_state=seed))

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].plot(ks, inertias, marker='o', color=PALETTE[0]); ax[0].set_title('Elbow — inertia vs k')
ax[0].set_xlabel('k'); ax[0].set_ylabel('Inertia')
ax[1].plot(ks, silhouettes, marker='s', color=PALETTE[1]); ax[1].set_title('Silhouette vs k')
ax[1].set_xlabel('k'); ax[1].set_ylabel('Silhouette')
for a in ax:
    a.axvline(2, color='red', ls='--', alpha=0.5); a.grid(alpha=0.3)
plt.suptitle('Scelta del numero di cluster k', fontweight='bold')
plt.tight_layout(); plt.show()

best_k = ks[int(np.argmax(silhouettes))]
print(f'k con silhouette massima: {best_k}  ->  procediamo con k=2, coerente con le due classi note.')
```
#### 🌍 Analisi Macroscopica
Un algoritmo senza etichette (K-Means) deve ricevere un indizio dal Data Scientist: in quanti gruppi K deve smistare il set fornito per trovare densità sensate? Questa griglia elabora la Silhouette e la WCSS (Within-Cluster Sum of Square - l'Inertia). Rintraccia e valida formalmente che il gruppo naturale coeso dei dati del Telescopio (Non conoscendone i nomi Raggio o Adrone) è naturalmente frammentato in modo ottimale in esattamente due comparti, che in seguito l'autore fisserà per il proseguimento (k=2).
#### 🔬 Analisi Microscopica
- Genera List iterabile `ks` valutando compartimenti di 2 fino a 6. 
- `KMeans(n_clusters=kk, n_init=10...).fit(X_train_scaled)`: Per ogni ipotesi elabora in 10 tentativi (`n_init=10`) lanciando randomicamente e rimescolando i 10 punti di Centroide originari dello Spazio, per scendere e ottimizzare la miglior compattezza possibile senza bloccarsi localmente, salvando ai list `inertias` l'attributo calcolatore di coesione interna `km.inertia_` per poi aggiungervi `silhouette_score`. L'algoritmo per calcolare la Silhouette logica è molto dispendioso geometricamente (N * N operazioni), per cui si passa il fix costrutto `sample_size=2000` per non far esplodere il Python Ram process ma approssimarne una traccia sufficientemente valida stocastica. 
- Esegue Plot (Elbow Method / Gomito). Quando il calo di Inerzia si ferma in appiattimento orizzontale a forma di gomito flesso, e simultaneamente il diagramma a destra di Silhouette (Più alto è, meglio è la purezza densa ed isolata di separazione) traccia ed estrapola un picco altissimo a 2, si aggiunge la retta tracciata con `axvline` in rosso scuro a `x=2`, ed indicizza estraendo via `np.argmax(silhouettes)` (La cella col maggior punteggio Silhouette List) definendo esplicitamente il cluster `best_k` ottimale nativo e preimpostato in cui incastonare le logiche seguenti.

### Dimostrazione del Rumore sui Cluster in Geometria Analitica
```python
def cluster_quality(X_scaled, y_true, k=2, random_state=42):
    km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    labels = km.fit_predict(X_scaled)
    y_arr = np.array(y_true)
    purity = sum(np.bincount(y_arr[labels == c]).max() for c in np.unique(labels)) / len(y_arr)
    metrics = {
        'ARI':        adjusted_rand_score(y_arr, labels),
        'AMI':        adjusted_mutual_info_score(y_arr, labels),
        'NMI':        normalized_mutual_info_score(y_arr, labels),
        'purezza':    purity,
        'silhouette': silhouette_score(X_scaled, labels, sample_size=2000, random_state=random_state),
    }
    return metrics, labels, km

q_clean, labels_clean, _ = cluster_quality(X_train_scaled, y_train)

pc = PCA(n_components=2, random_state=seed).fit_transform(X_train_scaled)
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
ax[0].scatter(pc[:, 0], pc[:, 1], c=labels_clean, cmap='coolwarm', s=6, alpha=0.4)
ax[0].set_title('Cluster K-Means (k=2)')
ax[1].scatter(pc[:, 0], pc[:, 1], c=np.array(y_train), cmap='coolwarm', s=6, alpha=0.4)
ax[1].set_title('Classi reali (gamma / hadron)')
for a in ax:
    a.set_xlabel('PC1'); a.set_ylabel('PC2')
plt.suptitle('PCA — geometria dei cluster (dati puliti)', fontweight='bold')
plt.tight_layout(); plt.show()
```
#### 🌍 Analisi Macroscopica
Le misurazioni vettoriali ad 11 Dimensioni del set originario sono indecifrabili da disegnare per l'occhio umano su un monitor. Prima di valutare l'attacco del virus Noise Gaussiano sui K-Means, bisogna applicare la "Dimensionality Reduction" (PCA, Principal Component Analysis), che comprimerà brutalmente tutti i dati in un'ascia di 2 assi X e Y senza distruggere i confini di spaccatura Cluster. Dopo il plot avremo modo di osservare se i Cluster previsti a ignoranza dall'AI cieca assomigliano visivamente in geometria cromatica alle risposte vere e purificate del dataset.
#### 🔬 Analisi Microscopica
- Crea logica estrapolativa per metriche complesse non supervisionate tra cui ARI e NMI (che a differenza del Recall misurano quanto overlap o incastro scambievole hanno le etichette dell'AI coi target umani mascherati). 
- `purity = sum(np.bincount(y_arr[labels == c]).max()...`: Un list logico iterativo che preleva a cluster isolato quanto la classe dominante in quel settore (Raggio o Adrone nativo) costituisca densamente quasi tutto il gruppo per poi dividerlo per `len(y_arr)` del set in un indicatore di Purezza Percentuale statica 0-1.
- `PCA(n_components=2).fit_transform(X)`: Applica calcolo di rotazione asse con estrazione di due autovettori dominanti per la massima varianza per comprimerne gli 11 valori e ricavare il grafico. 
- Utilizza Matplotlib costrutto visivo per grafici a dispersione di pallini spaziali `.scatter(...)`. Nel lato sx utilizza `c=labels_clean` (dando i colori ai pallini che la K-Means ha dedotto essere isolati di classe tra loro) e al dx in `c=np.array(y_train)` forzando la stampa di colore dettata dall'oracolo Y etichetta fisica del telescopio a formattare croma `coolwarm`.

### L'assalto Gaussiano distruttivo dei Cluster finali
```python
pca_clean = PCA(n_components=2, random_state=seed).fit(X_train_scaled)
noise_show = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for ax, lvl in zip(axes.ravel(), noise_show):
    Xn = X_train_scaled if lvl == 0 else add_gaussian_noise(X_train_scaled, noise_level=lvl, seed=seed)
    labels = KMeans(n_clusters=2, n_init=10, random_state=seed).fit_predict(Xn)
    pc = pca_clean.transform(Xn)
    ax.scatter(pc[:, 0], pc[:, 1], c=labels, cmap='coolwarm', s=6, alpha=0.4)
    sil = silhouette_score(Xn, labels, sample_size=2000, random_state=seed)
    ax.set_title(f'sigma = {lvl}  |  silhouette = {sil:.3f}')
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
plt.suptitle('Cluster K-Means (k=2) al crescere del feature noise', fontweight='bold')
plt.tight_layout(); plt.show()
```
#### 🌍 Analisi Macroscopica
Un ciclo massivo. Più ci si alza a deviazione rumore gaussiano estremo (sfociante al pazzesco sigma 2.0 per rottura sensoristica profonda del test), i dati vettoriali subiscono disordine casuale, allontanandosi dai centroidi geometrici. La macchina perderà colpi non sapendo più raggruppare i Raggi dalle particele Gamma, impastando una nebulosa insensata e vedendo decadere drasticamente il punteggio Silhouette di logica densità ad indicare fallimento strutturale.
#### 🔬 Analisi Microscopica
- Fissa uno Schedulatore test su lista Python `[0.0, 0.25, 0.5, 1.0, 1.5, 2.0]`. Instanzia griglia per 6 layout visivi subplots tramite la formula `2, 3` riga/colonna.
- L'astuta ed elegante espressione `pca_clean.transform(Xn)` utilizza il `transform` per assottigliare le componenti ai Sub-plot 2D usando ciecamente l'ispezione della PCA fissata sui dati Puri ed immacolati al momento dello spawn al rigo 1 dello script per fare il fitting, forzando la base su medesima scala cartesiana senza ricalibrarsi adattivamente (cosa che sfalserebbe la distruzione, allineandola all'incertezza). 
- Plot tracciato Scatter loop a estrapolazione di string titolatrice `ax.set_title(..)` riportante in F-string diretta per console il log decrementato del frame di Silhouette calcolato a blocco incapsulante.

### Linea del Crollo Definitivo (ARI vs Silhouette)
```python
feature_levels = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]
print('--- Effetto del Feature Noise su K-Means ---')
print(f'{"sigma":>6}{"ARI":>8}{"AMI":>8}{"silhouette":>12}{"purezza":>10}')
ari_list, sil_list = [], []
for lvl in feature_levels:
    X_noisy = add_gaussian_noise(X_train_scaled, noise_level=lvl, seed=seed)
    q, _, _ = cluster_quality(X_noisy, y_train)
    ari_list.append(q['ARI']); sil_list.append(q['silhouette'])
    print(f'{lvl:>6.2f}{q["ARI"]:>8.3f}{q["AMI"]:>8.3f}{q["silhouette"]:>12.3f}{q["purezza"]:>10.3f}')

fig, ax1 = plt.subplots(figsize=(10, 5.5))
ax1.plot(feature_levels, ari_list, marker='o', color=PALETTE[0], label='ARI (vs classi reali)')
ax1.set_xlabel('Sigma del feature noise'); ax1.set_ylabel('ARI', color=PALETTE[0])
ax1.tick_params(axis='y', labelcolor=PALETTE[0]); ax1.grid(alpha=0.3)
ax2 = ax1.twinx()
ax2.plot(feature_levels, sil_list, marker='s', color=PALETTE[1], label='Silhouette (coesione)')
ax2.set_ylabel('Silhouette', color=PALETTE[1]); ax2.tick_params(axis='y', labelcolor=PALETTE[1])
plt.title('K-Means — degrado dei cluster al crescere del feature noise', fontweight='bold')
fig.tight_layout(); plt.show()

y_train_flip = add_label_noise_random(y_train, noise_level=0.40, seed=seed)
_, labels_lab, _ = cluster_quality(X_train_scaled, y_train_flip)
print(f'\nARI(clustering pulito, clustering label-noise 40%) = {adjusted_rand_score(labels_clean, labels_lab):.3f}')
print('K-Means non riceve y in fit_predict -> il Label Noise non sposta i cluster (ARI=1.0).')
```
#### 🌍 Analisi Macroscopica
Un finale riepilogativo di dismissione al K-Means. Applica Line Chart e formatta le stringhe in modo Console Ascii-Table-Like. Poi traccia il calo inesorabile. Conclude formalmente spiegando con un ultimo test per il Label Noise, che se si sporca col 40% il Target etichettatura `y`, al KMeans logico questo test non intaccherà NULLA e restituirà ARI=1.0 (Stabilità immacolata a perfezione), poiché essendo un Apprendimento Non Supervisionato (Cieco) non calcola in via nativa e non usa derivate relative alla label per modellare le figure ad array, risultando impassibile (Immune) ad inquinamento che agisce solo ed unicamente sul target Y. 
#### 🔬 Analisi Microscopica
- Dichiara per Python logica stringhe interpolata spaziatore esplicita `{:>8.3f}` imponendovi allineamento Right formale per console Output per le varie metriche AMI (Adjusted Mutual Information, quantizzatore non lineare entropico logaritmico). 
- `ax2 = ax1.twinx()`: Costrutto avanzato di Plot per formare ed ancorare un tracciato Bi-assiale con Y sfasato duale. ARI va da range di 0 a 1 mentre Silhouette potrebbe avere altre scale. Disegnandole in unica scala y si schiaccerebbe la preview, questo comando modella una tela Trasparente 2 per l'asse di Destra per Silhouette, e Asse di Sinistra base ad ascisse per il log decostruttivo di ARI, colorando i marker laterali asincronamente per una visualizzazione ottica leggibilissima.

---

✅ **Fine del Documento.**
Tutta l'architettura logica e la pipeline esplorata nel Notebook originale sono stati decostruiti fino alla loro logica computazionale e di algebra lineare. Speriamo di aver chiarito esattamente come avvengano le computazioni vettoriali in TensorFlow, Pandas e SciKit Learn!
