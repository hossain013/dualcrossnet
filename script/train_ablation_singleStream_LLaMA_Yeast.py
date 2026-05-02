"""
Ablation Study: Single-Stream LLaMA (Concatenated Embeddings)
==============================================================
Change vs original dual-stream model:
  - REMOVED : separate per-protein stems, two LLaMA blocks, cross-attention,
               dual residuals, dual hybrid pooling
  - REPLACED: single shared stem → concatenate protein_A + protein_B along
               sequence axis → one LLaMA self-attention block over the
               joint sequence → one hybrid pooling → MLP head
  - KEPT    : stem projection dim (384), LLaMA config, hybrid pooling,
               MLP head dims (256→256→1), all dropout values, BN momentum
  - KEPT    : all training settings (focal loss, AdamW, warmup-cosine LR)

Design:
  inp1  (B, L, 1024)  ──stem──┐
                               ├─ concat along L → (B, 2L, 384) ──LLaMA──HybridPool──MLP──out
  inp2  (B, L, 1024)  ──stem──┘

  Padding masks are also concatenated so LLaMA sees the correct
  real/pad positions across the joint sequence.

Purpose: measure whether explicit dual-stream + cross-attention is
         needed, vs letting a single LLaMA block model both proteins
         jointly from the start.

SOTA (RLEAAI) reference metrics (threshold=0.50):
  Precision=0.887  Recall=0.911  Specificity=0.884
  F1=0.899  AUC=0.948  AUPR=0.939
"""

# ──────────────────────────────────────────────────────────────
# Environment
# ──────────────────────────────────────────────────────────────
import os, random, time, warnings
warnings.filterwarnings("ignore")
os.environ["KERAS_BACKEND"]         = "tensorflow"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "2"

import numpy as np
import pandas as pd
import h5py
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
import keras
from keras import layers, Model, Input
from keras_hub.src.models.llama.llama_decoder import LlamaTransformerDecoder
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    matthews_corrcoef, roc_curve
)
from tqdm.auto import tqdm

# ──────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ──────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────
SCRIPT_DIR = '../DeepInterAware/data/Yeast'
H5_PATH    = os.path.join(SCRIPT_DIR, "Yeast-ProtT5-Full.h5")
CSV_PATH   = "https://raw.githubusercontent.com/Fengithub/symLMF-PPI/refs/heads/master/datasets/S.cerevisiae-benchmark/pros_AB.txt"

# ──────────────────────────────────────────────────────────────
# CONFIG  (identical to original — only architecture changes)
# ──────────────────────────────────────────────────────────────
TIMESTAMP = time.strftime("%Y%m%d-%H%M%S")

CONFIG = {
    # ── data
    "batch_size"      : 64,
    "max_len"         : 512,
    "n_splits"        : 5,
    "random_state"    : SEED,
    # ── training schedule
    "epochs"          : 15,
    "peak_lr"         : 3e-4,
    "min_lr"          : 1e-6,
    "warmup_ratio"    : 0.10,
    # ── optimiser
    "optimizer"       : "AdamW",
    "weight_decay"    : 1e-2,
    "gradient_clip"   : 1.0,
    # ── loss
    "loss"            : "focal",
    "focal_gamma"     : 2.0,
    "focal_alpha"     : 0.5,
    "label_smoothing" : 0.05,
    # ── model head (unchanged)
    "drop_pool"       : 0.30,
    "drop_linear"     : 0.20,
    "bn_momentum"     : 0.90,
    # ── meta
    "architecture"    : "ProtT5",
    "dataset"         : "Yeast",
    "task"            : "Prot-Prot Classification",
    "ablation"        : "single_stream_llama",   # ← ablation tag
}

PROJECT_NAME = f"{CONFIG['dataset']}-ABLATION-SingleStreamLLaMA-{CONFIG['architecture']}-{TIMESTAMP}"
OUT_PATH = os.path.join(SCRIPT_DIR, "weights", PROJECT_NAME)
os.makedirs(os.path.join(OUT_PATH, "logs"),    exist_ok=True)
os.makedirs(os.path.join(OUT_PATH, "weights"), exist_ok=True)

print(f"[INFO] ABLATION: Single-Stream LLaMA (concatenated embeddings)")
print(f"[INFO] Output path: {OUT_PATH}")
print(f"[INFO] Config:\n{CONFIG}\n")

# ──────────────────────────────────────────────────────────────
# 1. Load embeddings
# ──────────────────────────────────────────────────────────────
print("[INFO] Loading embeddings from HDF5 …")
load_seq = {}
with h5py.File(H5_PATH, "r") as hf:
    for seq in tqdm(hf.keys(), desc="HDF5 keys"):
        load_seq[seq] = hf[seq][:]
print(f"[INFO] Loaded {len(load_seq)} sequences.")

# ──────────────────────────────────────────────────────────────
# 2. Load CSV
# ──────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH, sep='\t')
print(f"[INFO] Dataset shape: {df.shape}  |  label counts:\n{df['Interaction'].value_counts().to_dict()}")

# ──────────────────────────────────────────────────────────────
# 3. LR schedule: linear warmup + cosine decay
# ──────────────────────────────────────────────────────────────
class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Linear warmup for `warmup_steps`, then cosine decay to `min_lr`."""

    def __init__(self, peak_lr, total_steps, warmup_ratio=0.10, min_lr=1e-6):
        super().__init__()
        self.peak_lr      = float(peak_lr)
        self.min_lr       = float(min_lr)
        self.total_steps  = int(total_steps)
        self.warmup_steps = int(total_steps * warmup_ratio)

    def __call__(self, step):
        step      = tf.cast(step, tf.float32)
        w_steps   = tf.cast(self.warmup_steps, tf.float32)
        t_steps   = tf.cast(self.total_steps,  tf.float32)
        peak      = self.peak_lr
        mn        = self.min_lr
        warmup_lr = peak * step / tf.maximum(w_steps, 1.0)
        progress  = tf.maximum(step - w_steps, 0.0) / tf.maximum(t_steps - w_steps, 1.0)
        cosine_lr = mn + 0.5 * (peak - mn) * (1.0 + tf.cos(np.pi * progress))
        return tf.where(step < w_steps, warmup_lr, cosine_lr)

    def get_config(self):
        return dict(peak_lr=self.peak_lr, total_steps=self.total_steps,
                    warmup_steps=self.warmup_steps, min_lr=self.min_lr)


# ──────────────────────────────────────────────────────────────
# 4. Focal Loss
# ──────────────────────────────────────────────────────────────
def binary_focal_loss(gamma=2.0, alpha=0.5, label_smoothing=0.05):
    def loss_fn(y_true, y_pred):
        y_true  = tf.cast(y_true, tf.float32)
        y_pred  = tf.cast(y_pred, tf.float32)
        if label_smoothing > 0:
            y_true = y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing
        y_pred  = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        bce     = -(y_true * tf.math.log(y_pred)
                    + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
        p_t     = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        focal_w = tf.pow(1.0 - p_t, gamma)
        alpha_t = y_true * alpha + (1.0 - y_true) * (1.0 - alpha)
        return tf.reduce_mean(alpha_t * focal_w * bce)
    loss_fn.__name__ = f"focal_g{gamma}_a{alpha}_ls{label_smoothing}"
    return loss_fn


# ──────────────────────────────────────────────────────────────
# 5. Data loader
# ──────────────────────────────────────────────────────────────
def process_sequence_tf(x_emb, max_len=512, pad_value=0.0):
    seq_len = x_emb.shape[0]
    if seq_len > max_len:
        x_emb = tf.convert_to_tensor(x_emb[:max_len])
        mask  = tf.ones([max_len], dtype=tf.float32)
    else:
        pad_len = max_len - seq_len
        x_emb   = tf.pad(x_emb, [[0, pad_len], [0, 0]], constant_values=pad_value)
        mask    = tf.pad(tf.ones([seq_len], dtype=tf.float32),
                         [[0, pad_len]], constant_values=0.0)
    return x_emb, mask


class DataSequenceLoader(Sequence):
    def __init__(self, df, batch_size=32, shuffle=True, max_len=512, pad_value=0.0):
        self.x1_emb     = df["Protein_A_sequence"].values
        self.x2_emb     = df["Protein_B_sequence"].values
        self.labels     = df["Interaction"].values
        self.batch_size = batch_size
        self.shuffle    = shuffle
        self.max_len    = max_len
        self.pad_value  = pad_value
        self.indices    = np.arange(len(df))
        if shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        x1_list, x2_list, m1_list, m2_list, labs = [], [], [], [], []
        for i in batch_idx:
            e1 = load_seq[self.x1_emb[i]].reshape(-1, 1024)
            e2 = load_seq[self.x2_emb[i]].reshape(-1, 1024)
            x1, m1 = process_sequence_tf(e1, self.max_len, self.pad_value)
            x2, m2 = process_sequence_tf(e2, self.max_len, self.pad_value)
            x1_list.append(x1.numpy()); x2_list.append(x2.numpy())
            m1_list.append(m1.numpy()); m2_list.append(m2.numpy())
            labs.append(self.labels[i])
        return (np.stack(x1_list), np.stack(x2_list),
                np.stack(m1_list), np.stack(m2_list)), np.array(labs)


# ──────────────────────────────────────────────────────────────
# 6. Model — ABLATION: Single-Stream LLaMA
# ──────────────────────────────────────────────────────────────
class HybridPooling(layers.Layer):
    """Max + mean pool concatenated along feature axis. Unchanged."""
    def call(self, x):
        return keras.ops.concatenate(
            [keras.ops.max(x, axis=1), keras.ops.mean(x, axis=1)], axis=-1)


def llama_self_attention(x, mask, hidden_dim=100, num_heads=4):
    """Identical LLaMA config to original."""
    llama = LlamaTransformerDecoder(
        intermediate_dim    = hidden_dim * 4,
        num_query_heads     = 8,
        num_key_value_heads = 2,
        dropout             = 0.0,
        layer_norm_epsilon  = 1e-5,
        activation          = "silu",
    )
    return llama(x, decoder_padding_mask=mask)


def build_model(drop_pool=0.30, drop_linear=0.20, bn_momentum=0.90,
                heads=4, d_dim=32, conv_out=100):
    """
    Single-stream architecture:

    inp1 (B, L, 1024) ──stem──┐
                               ├─ Concatenate(axis=1) → (B, 2L, 384)
    inp2 (B, L, 1024) ──stem──┘
                                    │
                               LLaMA self-attention over joint (B, 2L, 384)
                                    │
                               HybridPooling → (B, 768)
                                    │
                               Dense(256) → Dropout → Dense(256) → Dropout
                                    │
                               Dense(1, sigmoid)

    Shared stem weights: both proteins pass through the SAME Dense+BN
    before concatenation, enforcing symmetric treatment.

    Joint padding mask: concat(mask1, mask2) along L so LLaMA correctly
    attends to real tokens and ignores padding in both proteins.
    """
    inp1  = Input((None, 1024), name="inp1")
    inp2  = Input((None, 1024), name="inp2")
    mask1 = Input((None,),      name="mask1")
    mask2 = Input((None,),      name="mask2")

    x_dim = 384

    # ── Shared stem (same Dense + BN for both proteins) ────────
    stem_dense = layers.Dense(x_dim, use_bias=False, name="stem_shared")
    stem_bn    = layers.BatchNormalization(momentum=bn_momentum, name="bn_shared")

    p1 = stem_bn(stem_dense(inp1))   # (B, L, 384)
    p2 = stem_bn(stem_dense(inp2))   # (B, L, 384)

    # ── Concatenate along sequence axis → single joint sequence ─
    # Shape: (B, 2L, 384)
    joint_seq  = layers.Concatenate(axis=1, name="joint_seq")([p1, p2])

    # ── Concatenate padding masks → joint mask (B, 2L) ──────────
    joint_mask = layers.Concatenate(axis=1, name="joint_mask")([mask1, mask2])

    # ── Single LLaMA block over the joint sequence ──────────────
    # LLaMA config identical to original (hidden_dim=384, heads=8, kv=2)
    s = llama_self_attention(joint_seq, joint_mask, hidden_dim=x_dim,
                             num_heads=heads)   # (B, 2L, 384)

    # ── Dropout (replaces per-stream dropout after Add) ─────────
    s = layers.Dropout(drop_pool)(s)

    # ── Hybrid pooling over joint sequence → (B, 768) ───────────
    pooled = HybridPooling()(s)

    # ── MLP head (unchanged dims) ───────────────────────────────
    x = layers.Dense(256, activation="relu")(pooled)
    x = layers.Dropout(drop_linear)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(drop_linear)(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    return Model(inputs=[inp1, inp2, mask1, mask2], outputs=out)


# ──────────────────────────────────────────────────────────────
# 7. Threshold optimisation (Youden's J)
# ──────────────────────────────────────────────────────────────
def find_optimal_threshold(y_true, y_prob):
    """Return threshold that maximises Youden's J = sensitivity + specificity − 1."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    best_idx = np.argmax(tpr - fpr)
    return float(thresholds[best_idx])


# ──────────────────────────────────────────────────────────────
# 8. K-Fold Training Loop
# ──────────────────────────────────────────────────────────────
skf         = StratifiedKFold(n_splits=CONFIG["n_splits"], shuffle=True,
                               random_state=CONFIG["random_state"])
all_metrics = []

for fold, (train_idx, valid_idx) in enumerate(skf.split(df, df["Interaction"]), 1):
    print(f"\n{'='*40}")
    print(f" FOLD {fold} / {CONFIG['n_splits']}")
    print(f"{'='*40}")

    train_df = df.iloc[train_idx].reset_index(drop=True)
    valid_df = df.iloc[valid_idx].reset_index(drop=True)
    print(f"  train={len(train_df)}  val={len(valid_df)}")

    train_loader = DataSequenceLoader(
        train_df, batch_size=CONFIG["batch_size"],
        max_len=CONFIG["max_len"], shuffle=True)
    valid_loader = DataSequenceLoader(
        valid_df, batch_size=CONFIG["batch_size"],
        max_len=CONFIG["max_len"], shuffle=False)

    # ── LR schedule
    steps_per_epoch = len(train_loader)
    total_steps     = steps_per_epoch * CONFIG["epochs"]
    lr_schedule     = WarmupCosineDecay(
        peak_lr      = CONFIG["peak_lr"],
        total_steps  = total_steps,
        warmup_ratio = CONFIG["warmup_ratio"],
        min_lr       = CONFIG["min_lr"],
    )
    print(f"  steps/epoch={steps_per_epoch}  total_steps={total_steps}  "
          f"warmup={int(total_steps*CONFIG['warmup_ratio'])}")

    # ── Optimiser
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate = lr_schedule,
        weight_decay  = CONFIG["weight_decay"],
        clipnorm      = CONFIG["gradient_clip"],
    )

    # ── Loss
    loss_fn = binary_focal_loss(
        gamma           = CONFIG["focal_gamma"],
        alpha           = CONFIG["focal_alpha"],
        label_smoothing = CONFIG["label_smoothing"],
    )

    # ── Model
    model = build_model(
        drop_pool   = CONFIG["drop_pool"],
        drop_linear = CONFIG["drop_linear"],
        bn_momentum = CONFIG["bn_momentum"],
    )
    model.compile(
        optimizer = optimizer,
        loss      = loss_fn,
        metrics   = ["accuracy", tf.keras.metrics.AUC(name="auc")],
    )

    # ── Callbacks
    ckpt_path = os.path.join(OUT_PATH, "weights", f"weights_fold{fold}-best.weights.h5")
    callbacks = [
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(OUT_PATH, "logs", f"fold_{fold}")),
        tf.keras.callbacks.ModelCheckpoint(
            filepath          = ckpt_path,
            save_weights_only = True,
            save_best_only    = True,
            monitor           = "val_auc",
            mode              = "max",
        ),
        tf.keras.callbacks.TerminateOnNaN(),
    ]

    # Optional W&B
    try:
        import wandb
        from wandb.integration.keras import WandbMetricsLogger
        run = wandb.init(
            project = PROJECT_NAME, name=f"fold_{fold}",
            group   = "KFold-CV-ABLATION-SingleStreamLLaMA",
            config  = {**CONFIG, "fold": fold}, reinit=True)
        callbacks.append(WandbMetricsLogger(log_freq="epoch"))
        use_wandb = True
    except Exception:
        use_wandb = False

    # ── Train
    history = model.fit(
        train_loader,
        validation_data = valid_loader,
        epochs          = CONFIG["epochs"],
        callbacks       = callbacks,
    )

    # Save last weights
    model.save_weights(
        os.path.join(OUT_PATH, "weights", f"weights_fold{fold}-last.weights.h5"))

    # ── Load best checkpoint for evaluation
    model.load_weights(ckpt_path)

    # ── Predict
    y_prob = model.predict(valid_loader, verbose=0).ravel()
    y_true = valid_df["Interaction"].values

    # ── Evaluation A: threshold = 0.5 (SOTA-comparable)
    y_pred_05 = (y_prob >= 0.5).astype(int)
    tn05, fp05, fn05, tp05 = confusion_matrix(y_true, y_pred_05).ravel()
    spec05 = tn05 / (tn05 + fp05) if (tn05 + fp05) > 0 else 0.0

    m_sota = {
        "fold"        : fold,
        "threshold"   : 0.5,
        "accuracy"    : round(accuracy_score(y_true, y_pred_05),       4),
        "precision"   : round(precision_score(y_true, y_pred_05),      4),
        "recall"      : round(recall_score(y_true, y_pred_05),         4),
        "specificity" : round(spec05,                                   4),
        "f1"          : round(f1_score(y_true, y_pred_05),             4),
        "auc"         : round(roc_auc_score(y_true, y_prob),           4),
        "aupr"        : round(average_precision_score(y_true, y_prob), 4),
        "mcc"         : round(matthews_corrcoef(y_true, y_pred_05),    4),
    }
    print(f"\n  Fold {fold} @ threshold=0.50 (SOTA-comparable):")
    for k, v in m_sota.items():
        print(f"    {k:14s}: {v}")

    # ── Evaluation B: Youden's J optimal threshold
    opt_thresh = find_optimal_threshold(y_true, y_prob)
    y_pred_opt = (y_prob >= opt_thresh).astype(int)
    tn_o, fp_o, fn_o, tp_o = confusion_matrix(y_true, y_pred_opt).ravel()
    spec_o = tn_o / (tn_o + fp_o) if (tn_o + fp_o) > 0 else 0.0

    m_opt = {
        "fold"        : fold,
        "threshold"   : round(opt_thresh, 4),
        "accuracy"    : round(accuracy_score(y_true, y_pred_opt),      4),
        "precision"   : round(precision_score(y_true, y_pred_opt),     4),
        "recall"      : round(recall_score(y_true, y_pred_opt),        4),
        "specificity" : round(spec_o,                                   4),
        "f1"          : round(f1_score(y_true, y_pred_opt),            4),
        "auc"         : round(roc_auc_score(y_true, y_prob),           4),
        "aupr"        : round(average_precision_score(y_true, y_prob), 4),
        "mcc"         : round(matthews_corrcoef(y_true, y_pred_opt),   4),
    }
    print(f"\n  Fold {fold} @ threshold={opt_thresh:.4f} (Youden's J optimal):")
    for k, v in m_opt.items():
        print(f"    {k:14s}: {v}")

    all_metrics.append({"eval": "threshold_0.5",  **m_sota})
    all_metrics.append({"eval": "youden_optimal",  **m_opt})

    if use_wandb:
        wandb.log({f"sota_{k}": v for k, v in m_sota.items()})
        wandb.log({f"opt_{k}":  v for k, v in m_opt.items()})
        run.finish()

# ──────────────────────────────────────────────────────────────
# 9. Summary
# ──────────────────────────────────────────────────────────────
metrics_df = pd.DataFrame(all_metrics)

SOTA = {"precision": 0.887, "recall": 0.911, "specificity": 0.884,
        "f1": 0.899, "auc": 0.948, "aupr": 0.939}

numeric_cols = ["accuracy", "precision", "recall", "specificity",
                "f1", "auc", "aupr", "mcc"]

def summarise_group(df_group, label):
    avg = df_group[numeric_cols].mean()
    std = df_group[numeric_cols].std()
    print(f"\n{'='*62}")
    print(f"  {label}")
    print(f"{'='*62}")
    print(f"  {'Metric':14s}  {'Avg':>7}  {'±Std':>7}  {'vs SOTA':>12}")
    print(f"  {'-'*50}")
    for m in ["precision", "recall", "specificity", "f1", "auc", "aupr"]:
        val  = float(avg[m])
        sd   = float(std[m])
        sota = SOTA.get(m, None)
        diff_str = f"{'✓ +' if val-sota>=0 else '✗ '}{val-sota:+.4f}" if sota else ""
        print(f"  {m:14s}  {val:7.4f}  {sd:7.4f}  {diff_str:>12}")
    print(f"\n  MCC : {float(avg['mcc']):.4f} ± {float(std['mcc']):.4f}")
    return avg, std

df_05  = metrics_df[metrics_df["eval"] == "threshold_0.5"].copy()
df_opt = metrics_df[metrics_df["eval"] == "youden_optimal"].copy()

avg_05,  std_05  = summarise_group(df_05,  "5-FOLD AVG @ threshold=0.50  (SOTA RLEAAI uses 0.50)")
avg_opt, std_opt = summarise_group(df_opt, "5-FOLD AVG @ Youden's J threshold  (optimised)")

# Per-fold CSV
rows = []
for fold_id in sorted(df_05["fold"].unique()):
    r05  = df_05[df_05["fold"]==fold_id].iloc[0]
    ropt = df_opt[df_opt["fold"]==fold_id].iloc[0]
    row  = {"fold": fold_id}
    for m in numeric_cols:
        row[f"{m}_@0.5"]     = round(float(r05[m]),  4)
        row[f"{m}_@youdenJ"] = round(float(ropt[m]), 4)
    rows.append(row)

avg_row = {"fold": "Average"}
for m in numeric_cols:
    avg_row[f"{m}_@0.5"]     = round(float(avg_05[m]),  4)
    avg_row[f"{m}_@youdenJ"] = round(float(avg_opt[m]), 4)
rows.append(avg_row)

std_row = {"fold": "Std"}
for m in numeric_cols:
    std_row[f"{m}_@0.5"]     = round(float(std_05[m]),  4)
    std_row[f"{m}_@youdenJ"] = round(float(std_opt[m]), 4)
rows.append(std_row)

summary_df = pd.DataFrame(rows)
csv_out  = os.path.join(OUT_PATH, f"{PROJECT_NAME}-kfold_metrics.csv")
csv_out2 = os.path.join(OUT_PATH, f"{PROJECT_NAME}-Eval-kfold_metrics.csv")
summary_df.to_csv(csv_out,  index=False)
metrics_df.to_csv(csv_out2, index=False)
print(f"\n\n  Full results saved to:\n  {csv_out}")
