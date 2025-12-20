#%%

# Import libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
import copy
import random
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
#%%

# Random seed
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
#%% 1. Load data

# Root dataset directory
ROOT = Path("Amazon_products")

# Corpus paths
TRAIN_CORPUS_PATH = ROOT / "train" / "train_corpus.txt"
TEST_CORPUS_PATH = ROOT / "test" / "test_corpus.txt"

# Class-related information
CLASS_HIERARCHY_PATH = ROOT / "class_hierarchy.txt"
CLASS_KEYWORDS_PATH = ROOT / "class_related_keywords.txt"
CLASS_NAMES_PATH = ROOT / "classes.txt"

# Pre-trained embeddings
# (In "generate_embeddings.ipynb" / No need to reproduce that file)
LABEL_EMB_PATH = ROOT / "label_bert_mean_dapt.pt"
TRAIN_EMB_PATH = ROOT / "train_bert_mean_dapt.pt"
TEST_EMB_PATH  = ROOT / "test_bert_mean_dapt.pt"

#%%

# Data loading function
def load_corpus(path):
    """
    Load corpus file (train/test).
    Each line: `<int_id> <space> <review text...>`
    Returns: {doc_id: text}
    """
    corpus = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            doc_id_str, text = line.split(maxsplit=1)
            doc_id = int(doc_id_str)
            corpus[doc_id] = text.strip()
    return corpus

def load_class_names(path):
    """
    Load class name and id.
    Each line: `<id> <class_name>`
    Returns:
      - class_names: index == class_id (list)
      - name_to_id : class_name -> class_id (dictionary)
    """
    class_names = []
    name_to_id = {}

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            cls_id = int(parts[0])
            cls_name = parts[1]

            class_names.append(cls_name)
            name_to_id[cls_name] = cls_id

    return class_names, name_to_id

def load_class_hierarchy(path):
    """
    Load class hierarchy edges.
    Each line: `<parent_id> <child_id>`
    Returns:
      - parent_to_children: {parent_id: [child_id, ...]}
      - child_to_parents: {child_id: [parent_id, ...]}
    """
    parent_to_children = defaultdict(list)
    child_to_parents = defaultdict(list)

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parent_str, child_str = line.split()
            parent = int(parent_str)
            child = int(child_str)
            parent_to_children[parent].append(child)
            child_to_parents[child].append(parent)

    return dict(parent_to_children), dict(child_to_parents)

def load_class_keywords(path, class_names):
    """
    Load class-related keywords.
    Each line: `<class_name>:kw1,kw2,...`
    Returns: {class_id: [kw1, kw2, ...]}
    """
    name_to_id = {name: idx for idx, name in enumerate(class_names)}
    class_keywords = {}

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cls_name, kws_str = line.split(":", maxsplit=1)
            cls_name = cls_name.strip()
            kws = [k.strip() for k in kws_str.split(",") if k.strip()]
            cls_id = name_to_id[cls_name]
            class_keywords[cls_id] = kws

    return class_keywords

#%%

# Load data
train_corpus = load_corpus(TRAIN_CORPUS_PATH)
test_corpus  = load_corpus(TEST_CORPUS_PATH)
class_names, name_to_id = load_class_names(CLASS_NAMES_PATH)
parent2children, child2parents = load_class_hierarchy(CLASS_HIERARCHY_PATH)
class_keywords = load_class_keywords(CLASS_KEYWORDS_PATH, class_names)

# Load pre-trained embeddings
label_data = torch.load(LABEL_EMB_PATH)
label_init_emb = label_data["embeddings"]

train_data = torch.load(TRAIN_EMB_PATH)
train_ids = train_data["ids"]
train_doc_embs = train_data["embeddings"]

test_data = torch.load(TEST_EMB_PATH)
test_ids = test_data["ids"]
test_doc_embs = test_data["embeddings"]

#%%

# Train data + test data
# ("You are allowed to use the test corpus information during training.")
train_texts = [train_corpus[pid] for pid in train_ids]
test_texts = [test_corpus[pid] for pid in test_ids]
all_doc_texts = train_texts + test_texts

all_doc_embs = torch.cat([train_doc_embs, test_doc_embs], dim=0)
all_ids = list(range(len(all_doc_embs)))

#%% 2. Generate silver labels

# Label texts
def build_label_texts(class_names, class_keywords):
    """
    Args: class_names, class_keywords
    Returns: label_texts (List with length C)
    ex) 'grocery gourmet food snacks condiments ...'
    """
    label_texts = []

    for cid, name in enumerate(class_names):
        pretty_name = name.replace("_", " ")
        keywords = class_keywords.get(cid, [])
        text = " ".join([pretty_name] + keywords)
        label_texts.append(text)

    return label_texts

label_texts = build_label_texts(class_names, class_keywords)

#%%

# Label TF-IDF vectorizer
def build_label_tfidf(label_texts):
    """
    Args: label_texts (list with length C)
    Returns: vectorizer, label_tfidf (C x V sparse matrix)
    """
    vectorizer = TfidfVectorizer()
    label_tfidf = vectorizer.fit_transform(label_texts)
    return vectorizer, label_tfidf

# Semantic similarity (BERT embedding)
def compute_embedding_similarity_matrix(doc_embs, label_emb):
    """
    Args:
      - doc_embs: (N, D) torch.Tensor
      - label_emb: (C, D) torch.Tensor
    Returns: S_emb (N x C) numpy array (cosine similarity)
    """
    # L2 normalization
    doc_norm = torch.nn.functional.normalize(doc_embs, p=2, dim=1)
    label_norm = torch.nn.functional.normalize(label_emb, p=2, dim=1)
    S_emb = doc_norm @ label_norm.T

    return S_emb.cpu().numpy()

# Lexical similarity (TF-IDF)
def compute_lexical_similarity_matrix(doc_texts, vectorizer, label_tfidf, batch_size=2000):
    """
    Args:
      - doc_texts (list with length C)
      - vectorizer, label_tfidf
    Returns: S_lex (N x C) numpy array
    """
    sims_list = []
    for i in tqdm(range(0, len(doc_texts), batch_size), desc="Computing lexical similarity"):
        batch = doc_texts[i:i+batch_size]
        doc_vec = vectorizer.transform(batch)
        sims = cosine_similarity(doc_vec, label_tfidf)
        sims_list.append(sims)
    S_lex = np.vstack(sims_list)

    return S_lex

# Per-doc normalization
def normalize_per_doc(S, eps=1e-8):
    """
    Args: (N, C) numpy array
    Returns: normalized array
    """
    S_min = S.min(axis=1, keepdims=True)
    S_max = S.max(axis=1, keepdims=True)
    S_norm = (S - S_min) / (S_max - S_min + eps)
    return S_norm

#%%

# Make weighted sum of scores S_total
def build_doc_label_scores(doc_embs, doc_texts, label_emb, label_texts, alpha=0.7):

    # Semantic similarity
    S_emb = compute_embedding_similarity_matrix(doc_embs, label_emb)
    S_emb_norm = normalize_per_doc(S_emb)

    # Lexical similarity
    vectorizer, label_tfidf = build_label_tfidf(label_texts)
    S_lex = compute_lexical_similarity_matrix(doc_texts, vectorizer, label_tfidf)
    S_lex_norm = normalize_per_doc(S_lex)

    # Weighted sum
    S_total = alpha * S_emb_norm + (1.0 - alpha) * S_lex_norm

    return S_total, S_emb_norm, S_lex_norm

#%%

S_total, S_emb_norm, S_lex_norm = build_doc_label_scores(all_doc_embs, all_doc_texts,
                                                         label_init_emb, label_texts)

#%%

# Get all ancestors
def get_ancestors(label_id, child2parents):
    """
    Args: label_id, child2parents
    Returns: label ids of all ancestors (set)
    """
    ancestors = set()
    stack = [label_id]

    while stack:
        child = stack.pop()
        for parent in child2parents.get(child, []):
            if parent not in ancestors:
                ancestors.add(parent)
                stack.append(parent)

    return ancestors

# Generate silver labels from S_total
def silver_labels_from_scores(S_total, all_ids, child2parents, top_k_base=3,
                                min_labels=2, max_labels=3, score_threshold=0.7):
    """
    Args: S_total, all_ids, child2parents
    Returns: silver_labels (dictionary)
    ex) {pid: [label_id1, label_id2, ...]}  (multi-label in ascending order)
    """
    num_docs, num_classes = S_total.shape
    silver_labels = {}

    for i in tqdm(range(num_docs), desc="Generating silver labels"):
        scores = S_total[i]
        pid = all_ids[i]

        # Sort all labels in descending order based on scores
        sorted_indices = np.argsort(-scores)

        # Base candidates (top three labels)
        base_candidates = sorted_indices[:top_k_base]

        label_set = set()

        for cid in base_candidates:
            cid = int(cid)
            if score_threshold is not None and scores[cid] < score_threshold:
                continue
            label_set.add(cid)

            # Add Ancestors reflecting hierarchy
            label_set.update(get_ancestors(cid, child2parents))

        if not label_set:
            best = int(sorted_indices[0])
            label_set.add(best)
            label_set.update(get_ancestors(best, child2parents))

        # At most three labels
        if len(label_set) > max_labels:
            sorted_by_score = sorted(label_set, key=lambda cid: scores[cid], reverse=True)
            label_set = set(sorted_by_score[:max_labels])

        # At least two labels
        if len(label_set) < min_labels:
            for cid in sorted_indices:
                cid = int(cid)
                if cid not in label_set:
                    label_set.add(cid)
                    if len(label_set) >= min_labels:
                        break

        # Sort all label ids in ascending order
        silver_labels[pid] = sorted(label_set)

    return silver_labels

#%%

silver_labels = silver_labels_from_scores(S_total, all_ids, child2parents)

#%%

# Generate multi-hot matrix from silver_labels
def multi_hot_silver(silver_labels, num_docs, num_classes):
    """
    Args: silver_labels, num_docs, num_classes
    Returns: (num_docs, num_classes) multi-hot tensor (0/1)
    """
    y_silver = torch.zeros((num_docs, num_classes), dtype=torch.float32)

    for pid, label_list in silver_labels.items():
        if pid < 0 or pid >= num_docs:
            continue
        for cid in label_list:
            if 0 <= cid < num_classes:
                y_silver[pid, cid] = 1.0

    return y_silver

#%%

y_silver = multi_hot_silver(silver_labels, 49145, 531)

#%% 3. Set GCN classifier

# Make adjacency matrix
def build_label_adjacency(num_classes, parent2children, undirected=True):
    """
    Args: num_classes, parent2children, undirected=True
    Returns: A -> (num_classes, num_classes) 0/1 numpy array (float32)
    """
    A = np.zeros((num_classes, num_classes), dtype=np.float32)

    for p, children in parent2children.items():
        for c in children:
            if 0 <= p < num_classes and 0 <= c < num_classes:
                A[p, c] = 1.0
                if undirected:
                    A[c, p] = 1.0

    return A

# Normalize adjacency matrix
def normalize_adjacency(A):
    """
    Args: (C, C) numpy array (0/1 adjacency)
    Returns: A_hat -> (C, C) torch.FloatTensor
    """
    assert A.ndim == 2 and A.shape[0] == A.shape[1], "A must be square"

    C = A.shape[0]
    A_tilde = A + np.eye(C, dtype=np.float32)
    deg = A_tilde.sum(axis=1)   # shape (C,)
    deg_inv_sqrt = 1.0 / np.sqrt(deg + 1e-8)
    D_inv_sqrt = np.diag(deg_inv_sqrt)   # (C, C)
    A_hat = D_inv_sqrt @ A_tilde @ D_inv_sqrt   # (C, C)

    return torch.from_numpy(A_hat.astype(np.float32))

#%%

NUM_CLASSES = len(class_names)

A = build_label_adjacency(NUM_CLASSES, parent2children, undirected=True)
A_hat = normalize_adjacency(A)

#%%

class LabelGCN(nn.Module):
    """
    Multi-layer Graph Convolutional Network (GCN) encoder for label embeddings.
    Each layer applies:
        H <- torch.matmul(A_hat, H)
        H <- torch.matmul(H, W)
    followed by ReLU + Dropout (except the last layer).
    """
    def __init__(self, emb_dim, num_layers=2, dropout=0.5):
        super().__init__()

        # Learnable weight matrices for each GCN layer (square: emb_dim x emb_dim)
        self.weights = nn.ParameterList([nn.Parameter(torch.empty(emb_dim, emb_dim)) for _ in range(num_layers)])
        for W in self.weights:
            nn.init.xavier_uniform_(W)  # Xavier init for stability

        self.num_layers = num_layers
        self.dropout = dropout

    def forward(self, H, A_hat):
        """
        Args:
            H: Initial label embeddings (num_labels x emb_dim)
            A_hat: Normalized adjacency matrix (num_labels x num_labels)

        Returns:
            Updated label embeddings (num_labels x emb_dim)
        """
        for i, W in enumerate(self.weights):
            # Message passing: aggregate neighbor embeddings
            H = torch.matmul(A_hat, H)     # (num_labels x num_labels) * (num_labels x emb_dim)

            # Linear transformation with learnable weights
            H = torch.matmul(H, W)         # (num_labels x emb_dim) * (emb_dim x emb_dim)

            # Apply non-linearity + dropout (except last layer)
            if i < self.num_layers - 1:
                H = F.relu(H)
                H = F.dropout(H, p=self.dropout, training=self.training)
        return H
    
#%%

class GCNEnhancedClassifier(nn.Module):
    """
    Classifier that combines:
      - Document representation (x) projected into label embedding space
      - Label embeddings refined by a GCN over the label hierarchy
    """
    def __init__(self, input_dim, label_init_emb, A_hat, num_layers=1, dropout=0.5):
        super().__init__()
        emb_dim = label_init_emb.size(1)  # dimension of label embeddings

        # Project document embeddings to the same space as labels
        self.proj = nn.Linear(input_dim, emb_dim)

        # GCN to propagate information between related labels
        self.gcn = LabelGCN(emb_dim=emb_dim, num_layers=num_layers, dropout=dropout)

        # Trainable initial label embeddings
        self.label_init_emb = nn.Parameter(label_init_emb.clone())

        # Store adjacency matrix (not trainable, fixed as buffer)
        self.register_buffer("A_hat", A_hat)
        self.dropout = dropout

    def forward(self, x):
        """
        Args:
            x: Input embeddings for documents (batch_size x input_dim)

        Returns:
            logits: Prediction scores (batch_size x num_labels)
        """
        # Update label embeddings with GCN
        label_emb = self.gcn(self.label_init_emb, self.A_hat)   # (num_labels x emb_dim)

        # Project input to label space
        x_proj = self.proj(x)                                  # (batch_size x emb_dim)
        x_proj = F.dropout(x_proj, p=self.dropout, training=self.training)

        # Compute similarity between inputs and labels
        logits = torch.matmul(x_proj, label_emb.T)             # (batch_size x num_labels)

        return logits
    
#%%

# Random seed
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

#%%

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GCNEnhancedClassifier(all_doc_embs.size(1), label_init_emb, A_hat.to(device), num_layers=1).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

#%% 4. Prepare dataset

# Confidence values from S_total
S_total_t = torch.from_numpy(S_total)
with torch.no_grad():
    conf_values, _ = S_total_t.max(dim=1)

# Threshold to trust silver labels
t_silver = 0.7

# Divide into Labeled/Unlabeled indices using confidence values
l_mask = conf_values >= t_silver
u_mask = ~l_mask

l_indices = l_mask.nonzero(as_tuple=True)[0]
u_indices = u_mask.nonzero(as_tuple=True)[0]


#%%

# Select samples among U to send to LLM
u_conf = conf_values[u_indices]
sorted_conf, sorted_idx = torch.sort(u_conf)

n_llm = 1000
u_llm_indices = u_indices[sorted_idx[:n_llm]]

#%%

# Load LLM result from jsonl
import json

ROOT = Path("Amazon_products")
save_path = ROOT / "llm_labels_250.jsonl"

llm_labels_dict = {}   # {doc_index: [label_ids]}

with save_path.open("r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        obj = json.loads(line)
        doc_idx = int(obj["doc_index"])
        labels = [int(x) for x in obj.get("labels", [])]
        llm_labels_dict[doc_idx] = labels

u_llm_indices_final = [doc_idx for doc_idx, labels in llm_labels_dict.items() if labels]

print(f"#U_LLM with valid labels: {len(u_llm_indices_final)}")

#%%

# y_llm Make multi-hot matrix (y_llm)
num_llm = len(u_llm_indices_final)
num_labels = y_silver.shape[1]

y_llm = torch.zeros(num_llm, num_labels, dtype=torch.float32)

for row_idx, doc_idx in enumerate(u_llm_indices_final):
    for lbl in llm_labels_dict[doc_idx]:
        if 0 <= lbl < num_labels:
            y_llm[row_idx, lbl] = 1.0
            
#%%

# Make new L / U indices
l_indices_tensor = l_indices.clone().to(torch.long)
u_llm_tensor = torch.tensor(u_llm_indices_final, dtype=torch.long)
l_new_indices = torch.cat([l_indices_tensor, u_llm_tensor], dim=0)

u_set = set(u_indices.tolist())
for doc_idx in u_llm_indices_final:
    if doc_idx in u_set:
        u_set.remove(doc_idx)
u_new_list = sorted(list(u_set))
u_new_indices = torch.tensor(u_new_list, dtype=torch.long)

#%%

# Embedding dataset for multi-label
class MultiLabelEmbeddingDataset(Dataset):
    def __init__(self, doc_embs, y_multi_hot=None):
        """
        Args:
          - doc_embs : [N, emb_dim] tensor
          - y_multi_hot : [N, num_labels] tensor or None
        """
        assert y_multi_hot is None or doc_embs.size(0) == y_multi_hot.size(0)
        self.doc_embs = doc_embs
        self.labels = y_multi_hot
        self.has_labels = y_multi_hot is not None

    def __len__(self):
        return self.doc_embs.size(0)

    def __getitem__(self, idx):
        X = self.doc_embs[idx]
        if self.has_labels:
            y = self.labels[idx]
            return {"X": X, "y": y}
        else:   # for unlabeled
            return {"X": X}
        
#%%

# Labeled dataset
labeled_doc_embs = all_doc_embs[l_new_indices]
labeled_y_old = y_silver[l_indices_tensor]
labeled_y_new = torch.cat([labeled_y_old, y_llm.to(labeled_y_old.device)], dim=0)
labeled_dataset = MultiLabelEmbeddingDataset(labeled_doc_embs, labeled_y_new)

# Unlabeled dataset: provide embeddings of unlabeled products for pseudo-labeling
unlabeled_doc_embs = all_doc_embs[u_new_indices]
unlabeled_dataset = MultiLabelEmbeddingDataset(unlabeled_doc_embs, None)

# 10% for validation
val_ratio = 0.1
val_size = int(len(labeled_dataset) * val_ratio)
train_size = len(labeled_dataset) - val_size

# Fix random seed and split the labeled_dataset
g = torch.Generator()
g.manual_seed(42)

train_dataset, val_dataset = random_split(labeled_dataset, [train_size, val_size], generator=g)

# Prepare the DataLoader
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, generator=g)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, generator=g)
unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=256, shuffle=False, generator=g)

#%% 5. Train the model

# Evaluation function for multi-label
def evaluate_multi_label(model, dataloader, device="cpu", threshold=0.5):
    model.eval()
    all_true, all_pred = [], []

    with torch.no_grad():
        for batch in dataloader:
            X = batch["X"].to(device)
            y = batch["y"].to(device)
            logits = model(X)
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()
            all_true.append(y.cpu())
            all_pred.append(preds.cpu())

    y_true = torch.vstack(all_true).numpy().astype(int)
    y_pred = torch.vstack(all_pred).numpy().astype(int)

    acc = accuracy_score(y_true, y_pred)
    f1_samples = f1_score(y_true, y_pred, average="samples", zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    return {"accuracy": acc, "f1_samples": f1_samples,
            "f1_micro": f1_micro, "f1_macro": f1_macro}

# Print evaluation result
def print_eval_result(metrics, stage="val", is_improved=False):
    star = " *" if is_improved else ""
    print(
        f"[{stage.upper():4}] "
        f"Acc: {metrics['accuracy']:.4f} | "
        f"F1-samples: {metrics['f1_samples']:.4f} | "
        f"F1-micro: {metrics['f1_micro']:.4f} | "
        f"F1-macro: {metrics['f1_macro']:.4f}{star}"
    )
    
#%%

# Generate pseudo_dataset for unlabeled data using model predictions above a confidence threshold
def generate_pseudo_dataset_from_unlabeled(model, unlabeled_loader, device="cpu", threshold=0.85):
    # Set the model to evaluation mode (disable dropout, batchnorm updates, etc.)
    model.eval()
    pseudo_X, pseudo_y = [], []

    # Disable gradient calculation for efficiency
    with torch.no_grad():
        for batch in unlabeled_loader:
            # Move input batch to the given device (CPU/GPU)
            X_batch = batch["X"].to(device)

            # Forward pass: compute logits
            logits = model(X_batch)

            # Convert logits to probability distribution (multi-label)
            probs = torch.sigmoid(logits)

            # Select labels for each sample (probs >= threshold)
            batch_mask = probs >= threshold

            # Iterate over each sample in the batch
            for i in range(X_batch.size(0)):
                label_mask = batch_mask[i]
                if not label_mask.any():
                    continue

                x_i = X_batch[i].detach().cpu()
                y_i = label_mask.float().detach().cpu()

                pseudo_X.append(x_i)
                pseudo_y.append(y_i)

    if len(pseudo_X) == 0:
        return None

    pseudo_X_tensor = torch.stack(pseudo_X)
    pseudo_y_tensor = torch.stack(pseudo_y)

    pseudo_dataset = MultiLabelEmbeddingDataset(pseudo_X_tensor, pseudo_y_tensor)
    return pseudo_dataset

#%%

criterion = nn.BCEWithLogitsLoss()

#%%

best_val_f1 = -1
best_model_state = None
patience = 5
patience_counter = 0

train_loss_list = []
val_f1_list = []

EPOCHS = 200

base_train_dataset = train_dataset
pseudo_dataset = None

for epoch in range(1, EPOCHS + 1):
    # === Check train dataset update every epoch ===
    if pseudo_dataset is not None:
        train_dataset = ConcatDataset([base_train_dataset, pseudo_dataset])
    else:
        train_dataset = base_train_dataset
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    # === Train (Consistency regularization) ===
    lambda_cons = 0.1   # consistency loss weight
    noise_std   = 0.1   # noise size added to embedding

    model.train()
    total_loss = 0.0
    total_batches = 0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
        X = batch["X"].to(device)
        y = batch["y"].to(device)

        # Make a slightly modified input
        noise = torch.randn_like(X) * noise_std
        X_aug = X + noise

        optimizer.zero_grad()

        # Forward twice
        logits1 = model(X)
        logits2 = model(X_aug)

        # supervised loss (BCEWithLogitsLoss)
        supervised_loss = criterion(logits1, y)

        # consistency loss
        probs1 = torch.sigmoid(logits1)
        probs2 = torch.sigmoid(logits2)
        consistency_loss = F.mse_loss(probs1, probs2)

        # Final loss = supervised + lambda * consistency
        loss = supervised_loss + lambda_cons * consistency_loss

        # backward & update
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

    avg_train_loss = total_loss / max(1, total_batches)
    train_loss_list.append(avg_train_loss)
    print(f"[Epoch {epoch}] Train Loss: {avg_train_loss:.4f}")

    # === Validation ===
    val_result = evaluate_multi_label(model, val_loader, device=device, threshold=0.5)
    val_f1 = val_result["f1_samples"]
    val_f1_list.append(val_f1)

    is_improved = val_f1 > best_val_f1
    print_eval_result(val_result, stage="val", is_improved=is_improved)

    # === Update best model ===
    if is_improved:
        best_val_f1 = val_f1
        best_model_state = copy.deepcopy(model.state_dict())
        patience_counter = 0
    else:
        patience_counter += 1

    # === Self-training: pseudo-label generation every 3 epochs ===
    if epoch % 3 == 0:
        pseudo_dataset = generate_pseudo_dataset_from_unlabeled(model, unlabeled_loader,
                                                                device=device, threshold=0.85)

        if pseudo_dataset is not None:
            print(f"[Self-training] Epoch {epoch}: Pseudo-labeled {len(pseudo_dataset)} examples added.")
        else:
            print(f"[Self-training] Epoch {epoch}: No pseudo-labeled data above threshold.")

    # === Early stopping ===
    if patience_counter >= patience:
        print(f"[Early Stopping] No improvement for {patience} consecutive epochs.")
        break

#%%

model.load_state_dict(best_model_state)
print(f"Best val F1-samples: {best_val_f1:.4f}")

#%% 6. Kaggle submission

import csv

# --- Paths ---
ROOT = Path("Amazon_products")
TEST_EMB_PATH    = ROOT / "test_bert_mean_dapt.pt"
SUBMISSION_PATH  = ROOT / "2021250031_final.csv"   # output file

# --- Constants ---
NUM_CLASSES = 531   # total number of classes (0–530)
MIN_LABELS = 2   # minimum number of labels per sample
MAX_LABELS = 3   # maximum number of labels per sample

# --- Load test embeddings ---
test_data = torch.load(TEST_EMB_PATH, map_location=device)
test_ids = test_data["ids"]
test_doc_embs = test_data["embeddings"]

# === Custom Dataset ===
class TestEmbeddingDataset(Dataset):
    def __init__(self, ids, embeddings):
        self.ids = ids
        self.embeddings = embeddings

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        pid = int(self.ids[idx])
        x = self.embeddings[idx]
        return {"id": pid, "X": x}

# === Build dataset and loader ===
test_dataset = TestEmbeddingDataset(test_ids, test_doc_embs)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# === Run predictions ===
model.to(device)
model.eval()

all_pred_ids = []
all_pred_labels = []

threshold = 0.5

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Predicting for Kaggle test"):
        X = batch["X"].to(device)
        ids = batch["id"].to(torch.long)

        logits = model(X)
        scores = torch.sigmoid(logits)

        # Top three labels per sample
        topk_scores, topk_indices = scores.topk(MAX_LABELS, dim=1)
        ids = ids.cpu().tolist()
        topk_scores = topk_scores.cpu()
        topk_indices = topk_indices.cpu()

        for pid, score_row, idx_row in zip(ids, topk_scores, topk_indices):
            # Leave only labels above threshold
            keep_mask = score_row >= threshold
            labels = idx_row[keep_mask].tolist()

            # At least two labels
            if len(labels) < MIN_LABELS:
                labels = idx_row[:MIN_LABELS].tolist()

            labels = sorted(labels)
            all_pred_ids.append(pid)
            all_pred_labels.append(labels)

# Sort by id to prevent possible order twist
pairs = sorted(zip(all_pred_ids, all_pred_labels), key=lambda x: x[0])

# === Build submission file ===
with open(SUBMISSION_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "label"])
    for pid, labels in pairs:
        writer.writerow([pid, ",".join(map(str, labels))])

print(f"Submission file saved to: {SUBMISSION_PATH}")
print("First 5 rows:")
for pid, labels in pairs[:5]:
    print(pid, "->", ",".join(map(str, labels)))




