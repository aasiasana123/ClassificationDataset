"""
Toxic Comment Classification - Streamlit app

Uses the LSTM model from the exercise (Embedding -> LSTM -> Dropout -> Dense -> Softmax).

Files expected next to this app.py in the GitHub repo:
    model.keras                      trained model            (exported from the notebook)
    word_index.json                  tokenizer vocabulary     (exported from the notebook)
    meta.json                        classes, max_len, history (exported from the notebook)
    toxic_comments_dataset.csv       dataset (for the Dataset / Performance tabs)

If the three exported files are missing but the CSV is present, the app trains
the model itself on first start-up and saves the files (takes a minute or two).
"""

import json
import os
import re
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Toxic Comment Classifier", page_icon="🛡️", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.keras")
WORD_INDEX_PATH = os.path.join(BASE_DIR, "word_index.json")
META_PATH = os.path.join(BASE_DIR, "meta.json")
DATA_PATH = os.path.join(BASE_DIR, "toxic_comments_dataset.csv")

OOV_TOKEN = "<OOV>"
NUM_WORDS = 5000
EPOCHS = 15
BATCH_SIZE = 64

SAMPLE_COMMENTS = [
    "Thank you for sharing this perspective with the community.",
    "This comment is rude and disrespectful.",
    "This reply contains a threat-like statement and is unsafe.",
    "The message targets a group unfairly and violates community rules.",
    "This message contains severe abusive language and should be removed.",
]


# ---------------------------------------------------------------------------
# Text preprocessing (same steps as Task 2 / Task 3)
# ---------------------------------------------------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)              # HTML tags
    text = re.sub(r"http\S+|www\.\S+", " ", text)   # URLs
    text = re.sub(r"[^\x00-\x7F]+", " ", text)      # emojis / non-ASCII
    text = re.sub(r"\d+", " ", text)                # numbers
    text = re.sub(r"[^a-z\s]", " ", text)           # punctuation / special characters
    text = re.sub(r"\s+", " ", text).strip()        # extra whitespace
    return text


def encode_texts(texts, word_index, max_len, num_words=NUM_WORDS):
    """Tokenize, convert to integer sequences, pad/truncate at the end ('post')."""
    oov_id = word_index.get(OOV_TOKEN, 1)
    rows = []
    for t in texts:
        ids = []
        for w in clean_text(t).split():
            i = word_index.get(w)
            if i is None or i >= num_words:
                i = oov_id
            ids.append(i)
        ids = ids[:max_len]
        ids += [0] * (max_len - len(ids))
        rows.append(ids)
    return np.array(rows, dtype="int32")


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_raw_dataset():
    if not os.path.exists(DATA_PATH):
        return None
    return pd.read_csv(DATA_PATH)


def prepare_split(classes=None):
    """Clean data and make the same stratified 80/20 split used in the notebook."""
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["Comment_Text", "Toxicity_Label"])
    df = df.drop_duplicates().reset_index(drop=True)
    df["Clean_Comment"] = df["Comment_Text"].apply(clean_text)

    if classes is None:
        classes = sorted(df["Toxicity_Label"].unique())  # same order as LabelEncoder
    label_to_id = {c: i for i, c in enumerate(classes)}
    y = df["Toxicity_Label"].map(label_to_id)

    X_train, X_test, y_train, y_test = train_test_split(
        df["Clean_Comment"], y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train.to_numpy(), y_test.to_numpy(), list(classes)


# ---------------------------------------------------------------------------
# Model: load saved files, or train once as a fallback
# ---------------------------------------------------------------------------
def train_fallback():
    import tensorflow as tf
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding, Input
    from tensorflow.keras.models import Sequential

    tf.random.set_seed(42)
    np.random.seed(42)

    X_train, X_test, y_train, y_test, classes = prepare_split()

    counts = Counter(w for t in X_train for w in t.split())
    word_index = {OOV_TOKEN: 1}
    for i, (w, _) in enumerate(counts.most_common(), start=2):
        word_index[w] = i

    lengths = [len(t.split()) for t in X_train]
    max_len = int(np.percentile(lengths, 99))

    X_train_pad = encode_texts(X_train, word_index, max_len)
    input_dim = min(NUM_WORDS, len(word_index) + 1)

    model = Sequential([
        Input(shape=(max_len,)),
        Embedding(input_dim=input_dim, output_dim=64),
        LSTM(64),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dropout(0.3),
        Dense(len(classes), activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(
        X_train_pad, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
        validation_split=0.2, verbose=0,
    )

    meta = {
        "classes": classes,
        "max_len": max_len,
        "num_words": NUM_WORDS,
        "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
    }
    try:  # saving is best-effort (the host file system may be read-only)
        model.save(MODEL_PATH)
        with open(WORD_INDEX_PATH, "w") as f:
            json.dump(word_index, f)
        with open(META_PATH, "w") as f:
            json.dump(meta, f)
    except Exception:
        pass
    return model, word_index, meta


@st.cache_resource(show_spinner="Loading model... (first start-up may take a minute)")
def load_artifacts():
    import tensorflow as tf

    if all(os.path.exists(p) for p in (MODEL_PATH, WORD_INDEX_PATH, META_PATH)):
        model = tf.keras.models.load_model(MODEL_PATH)
        with open(WORD_INDEX_PATH) as f:
            word_index = json.load(f)
        with open(META_PATH) as f:
            meta = json.load(f)
        return model, word_index, meta

    if os.path.exists(DATA_PATH):
        return train_fallback()

    raise FileNotFoundError(
        "No model files found. Add model.keras, word_index.json and meta.json "
        "(or toxic_comments_dataset.csv so the app can train itself) to the repo."
    )


def predict_one(text, model, word_index, meta):
    seq = encode_texts([text], word_index, meta["max_len"], meta.get("num_words", NUM_WORDS))
    return model.predict(seq, verbose=0)[0]


@st.cache_resource(show_spinner="Evaluating on the test set...")
def compute_metrics():
    from sklearn.metrics import classification_report, confusion_matrix

    model, word_index, meta = load_artifacts()
    _, X_test, _, y_test, classes = prepare_split(meta["classes"])
    X_test_pad = encode_texts(X_test, word_index, meta["max_len"], meta.get("num_words", NUM_WORDS))

    loss, acc = model.evaluate(X_test_pad, y_test, verbose=0)
    y_pred = np.argmax(model.predict(X_test_pad, verbose=0), axis=1)
    report = classification_report(y_test, y_pred, target_names=classes, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    return loss, acc, pd.DataFrame(report).T, cm, classes


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("🛡️ Toxic Comment Classifier")
st.caption("Deep Learning (Embedding + LSTM) multi-class toxicity classification built with TensorFlow/Keras.")

try:
    model, word_index, meta = load_artifacts()
except Exception as e:
    st.error(f"Could not load the model: {e}")
    st.stop()

classes = meta["classes"]

with st.sidebar:
    st.header("Model info")
    st.write(f"**Classes ({len(classes)}):**")
    for c in classes:
        st.write(f"- {c}")
    st.write(f"**Max sequence length:** {meta['max_len']}")
    st.write(f"**Vocabulary size:** {len(word_index)}")
    st.info(
        "The model only knows the words it saw during training, so comments with "
        "mostly unfamiliar words will give unreliable predictions."
    )

tab_predict, tab_data, tab_perf = st.tabs(["🔍 Predict", "📊 Dataset", "📈 Model Performance"])

# ----- Predict tab -----
with tab_predict:
    st.subheader("Classify a comment")
    sample = st.selectbox("Pick a sample comment (optional)", ["(write my own)"] + SAMPLE_COMMENTS)
    default_text = "" if sample == "(write my own)" else sample
    text = st.text_area("Comment text", value=default_text, height=120,
                        placeholder="Type or paste a comment here...")

    if st.button("Classify", type="primary"):
        if not text.strip():
            st.warning("Please enter a comment first.")
        else:
            probs = predict_one(text, model, word_index, meta)
            idx = int(np.argmax(probs))
            label, conf = classes[idx], float(probs[idx])

            if label == "Clean":
                st.success(f"Prediction: **{label}**  ({conf:.1%} confidence)")
            else:
                st.error(f"Prediction: **{label}**  ({conf:.1%} confidence)")

            words = clean_text(text).split()
            if words:
                unknown = sum(1 for w in words if w not in word_index)
                if unknown / len(words) >= 0.5:
                    st.warning(
                        f"{unknown} of {len(words)} words are not in the model's vocabulary, "
                        "so this prediction may be unreliable."
                    )

            st.write("**Class probabilities**")
            st.bar_chart(pd.Series(probs, index=classes, name="Probability"))
            with st.expander("Show cleaned text"):
                st.code(clean_text(text) or "(empty after cleaning)")

# ----- Dataset tab -----
with tab_data:
    raw = load_raw_dataset()
    if raw is None:
        st.info("Add toxic_comments_dataset.csv to the repo to see the dataset explorer.")
    else:
        st.subheader("Dataset overview")
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{raw.shape[0]:,}")
        c2.metric("Columns", raw.shape[1])
        c3.metric("Missing values", int(raw.isnull().sum().sum()))

        st.write("**Sample records**")
        st.dataframe(raw.head(10), use_container_width=True)

        st.write("**Toxicity label distribution**")
        st.bar_chart(raw["Toxicity_Label"].value_counts())

        st.write("**Statistical summary (numerical columns)**")
        st.dataframe(raw.describe(), use_container_width=True)

        n_unique = raw["Comment_Text"].nunique()
        st.caption(f"Note: the dataset has only {n_unique} distinct comment texts, "
                   "so very high accuracy is expected.")

# ----- Performance tab -----
with tab_perf:
    st.subheader("Training history")
    hist = meta.get("history")
    if hist:
        h1, h2 = st.columns(2)
        h1.write("**Accuracy**")
        h1.line_chart(pd.DataFrame({"Training": hist["accuracy"], "Validation": hist["val_accuracy"]}))
        h2.write("**Loss**")
        h2.line_chart(pd.DataFrame({"Training": hist["loss"], "Validation": hist["val_loss"]}))
    else:
        st.info("No training history saved. Include 'history' in meta.json to show the curves.")

    st.subheader("Test set evaluation")
    if not os.path.exists(DATA_PATH):
        st.info("Add toxic_comments_dataset.csv to the repo to evaluate on the test set.")
    elif st.button("Run evaluation"):
        import matplotlib.pyplot as plt

        loss, acc, report_df, cm, cm_classes = compute_metrics()
        m1, m2 = st.columns(2)
        m1.metric("Test accuracy", f"{acc:.2%}")
        m2.metric("Test loss", f"{loss:.4f}")

        st.write("**Classification report**")
        st.dataframe(report_df.round(4), use_container_width=True)

        st.write("**Confusion matrix**")
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(cm_classes)))
        ax.set_yticks(range(len(cm_classes)))
        ax.set_xticklabels(cm_classes, rotation=30, ha="right")
        ax.set_yticklabels(cm_classes)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        thresh = cm.max() / 2 if cm.max() else 0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.colorbar(im)
        fig.tight_layout()
        st.pyplot(fig)
