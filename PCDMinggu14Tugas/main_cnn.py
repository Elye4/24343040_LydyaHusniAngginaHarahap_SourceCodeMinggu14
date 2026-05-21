import os
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)

from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import pandas as pd

np.random.seed(42)

IMG_SIZE = 32

BASE_DIR = 'dataset'
RESULTS_DIR = 'results'

CLASSES = ['bird', 'cat', 'dog', 'car', 'motorcycle', 'truck']

CLASS_MAP = {
    'animals/bird': 'bird',
    'animals/cat': 'cat',
    'animals/dog': 'dog',
    'vehicles/car': 'car',
    'vehicles/motorcycle': 'motorcycle',
    'vehicles/truck': 'truck'
}

NUM_CLASSES = 6

os.makedirs(f'{RESULTS_DIR}/plots', exist_ok=True)

# ================= LOAD DATA =================

X = []
y = []

for folder, label in CLASS_MAP.items():

    path = os.path.join(BASE_DIR, folder)

    idx = CLASSES.index(label)

    for f in sorted(os.listdir(path)):

        if f.endswith('.png'):

            img = Image.open(os.path.join(path, f))

            img = img.convert('RGB')

            img = img.resize((IMG_SIZE, IMG_SIZE))

            img = np.array(img).astype(np.float32) / 255.0

            X.append(img.flatten())

            y.append(idx)

X = np.array(X)
y = np.array(y)

print("Dataset loaded:", X.shape)

# ================= SPLIT =================

X_tr, X_te, y_tr, y_te = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ================= MODELS =================

models = {

    'MLP_Small':
        MLPClassifier(
            hidden_layer_sizes=(128,),
            max_iter=50,
            random_state=42
        ),

    'MLP_Medium':
        MLPClassifier(
            hidden_layer_sizes=(256,128),
            max_iter=50,
            random_state=42
        ),

    'MLP_Deep':
        MLPClassifier(
            hidden_layer_sizes=(512,256,128),
            max_iter=50,
            random_state=42
        )
}

results = {}

# ================= TRAIN =================

for name, model in models.items():

    print(f"\nTraining {name}")

    t0 = time.time()

    model.fit(X_tr, y_tr)

    train_time = time.time() - t0

    pred = model.predict(X_te)

    acc = accuracy_score(y_te, pred)

    print(f"Accuracy: {acc:.4f}")

    results[name] = {
        'model': model,
        'acc': acc,
        'time': train_time
    }

# ================= BEST MODEL =================

best_name = max(results, key=lambda k: results[k]['acc'])

best_model = results[best_name]['model']

print(f"\nBest Model: {best_name}")

# ================= PREDICTION =================

y_pred = best_model.predict(X_te)

y_prob = best_model.predict_proba(X_te)

# ================= CONFUSION MATRIX =================

cm = confusion_matrix(y_te, y_pred)

plt.figure(figsize=(8,6))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=CLASSES,
    yticklabels=CLASSES
)

plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')

plt.tight_layout()

plt.savefig(f'{RESULTS_DIR}/plots/confusion_matrix.png')

plt.close()

print("Confusion matrix saved")

# ================= ROC =================

y_bin = label_binarize(y_te, classes=range(NUM_CLASSES))

plt.figure(figsize=(8,6))

colors = plt.cm.tab10(np.linspace(0,1,NUM_CLASSES))

for i, cls in enumerate(CLASSES):

    fpr, tpr, _ = roc_curve(y_bin[:,i], y_prob[:,i])

    roc_auc = auc(fpr, tpr)

    plt.plot(
        fpr,
        tpr,
        lw=2,
        color=colors[i],
        label=f'{cls} AUC={roc_auc:.2f}'
    )

plt.plot([0,1],[0,1],'k--')

plt.legend()

plt.xlabel('FPR')
plt.ylabel('TPR')

plt.title('ROC Curves')

plt.tight_layout()

plt.savefig(f'{RESULTS_DIR}/plots/roc_curves.png')

plt.close()

print("ROC curves saved")

# ================= t-SNE =================

pca = PCA(n_components=50)

X_pca = pca.fit_transform(X_te)

tsne = TSNE(
    n_components=2,
    random_state=42,
    perplexity=20
)

emb = tsne.fit_transform(X_pca)

plt.figure(figsize=(9,7))

colors = plt.cm.tab10(np.linspace(0,1,NUM_CLASSES))

for i, cls in enumerate(CLASSES):

    mask = y_te == i

    plt.scatter(
        emb[mask,0],
        emb[mask,1],
        label=cls,
        color=colors[i],
        alpha=0.7
    )

plt.legend()

plt.title('t-SNE Embeddings')

plt.tight_layout()

plt.savefig(f'{RESULTS_DIR}/plots/tsne_embeddings.png')

plt.close()

print("t-SNE saved")

# ================= COMPARISON =================

comp_data = []

for name, res in results.items():

    model = res['model']

    pred = model.predict(X_te)

    comp_data.append({

        'Model': name,

        'Accuracy': f"{accuracy_score(y_te,pred):.4f}",

        'F1': f"{f1_score(y_te,pred,average='macro'):.4f}",

        'Precision': f"{precision_score(y_te,pred,average='macro'):.4f}",

        'Recall': f"{recall_score(y_te,pred,average='macro'):.4f}",

        'Train Time': f"{res['time']:.2f}s"

    })

df = pd.DataFrame(comp_data)

print(df)

df.to_csv(f'{RESULTS_DIR}/comparison_table.csv', index=False)

# ================= BAR CHART =================

names = [d['Model'] for d in comp_data]

accs = [float(d['Accuracy']) for d in comp_data]

plt.figure(figsize=(8,5))

bars = plt.bar(names, accs)

plt.ylim(0,1)

for b,v in zip(bars, accs):

    plt.text(
        b.get_x() + b.get_width()/2,
        v + 0.01,
        f'{v:.3f}',
        ha='center'
    )

plt.title('Model Accuracy Comparison')

plt.ylabel('Accuracy')

plt.tight_layout()

plt.savefig(f'{RESULTS_DIR}/plots/comparison_chart.png')

plt.close()

print("Comparison chart saved")

# ================= REPORT =================

print("\n=== Classification Report ===")

print(classification_report(
    y_te,
    y_pred,
    target_names=CLASSES
))