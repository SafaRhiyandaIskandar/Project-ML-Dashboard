import io
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit_shadcn_ui as ui
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


st.set_page_config(
    page_title="Churn Intelligence",
    page_icon="CI",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_style():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
        :root { --ink:#14213d; --muted:#65708a; --line:#e5e9f1; --blue:#2457f5; --mint:#28c7a6; --canvas:#f5f7fb; }
        .stApp { background:var(--canvas); color:var(--ink); font-family:'Manrope',sans-serif; }
        [data-testid='stHeader'] { background:rgba(245,247,251,.86); }
        [data-testid='stSidebar'] { background:#101a32; }
        [data-testid='stSidebar'] * { color:#eef3ff !important; }
        [data-testid='stSidebar'] [data-testid='stSidebarContent'] { padding-top:2rem; }
        [data-testid='stSidebar'] .stButton { margin:1px 0; }
        [data-testid='stSidebar'] .stButton > button { background:transparent !important; border:1px solid transparent !important; border-radius:9px !important; box-shadow:none !important; color:#cbd7f2 !important; font-size:.86rem !important; font-weight:700 !important; justify-content:flex-start !important; min-height:38px !important; padding:7px 11px !important; text-align:left !important; transition:background .15s ease,color .15s ease !important; }
        [data-testid='stSidebar'] .stButton > button:hover { background:rgba(118,148,225,.14) !important; color:#fff !important; }
        [data-testid='stSidebar'] .stButton > button[kind='primary'] { background:#2858ed !important; box-shadow:0 6px 18px rgba(31,78,218,.32) !important; color:#fff !important; }
        [data-testid='stSidebar'] hr { border-color:rgba(194,211,250,.15) !important; margin:1.4rem 0 !important; }
        h1,h2,h3 { color:var(--ink); font-family:'Manrope',sans-serif !important; letter-spacing:-.035em; }
        h1 { font-size:2.25rem !important; font-weight:800 !important; }
        h2 { font-size:1.35rem !important; font-weight:800 !important; margin-top:1.25rem !important; }
        .eyebrow { color:#4d5d7e; font-family:'DM Mono',monospace; font-size:.75rem; font-weight:500; letter-spacing:.11em; text-transform:uppercase; }
        .hero { background:linear-gradient(122deg,#111d39 0%,#172b57 63%,#2457f5 145%); border-radius:20px; color:#fff; padding:34px 38px; margin:0 0 24px; overflow:hidden; }
        .hero h1 { color:#fff !important; margin:7px 0 9px !important; }
        .hero p { color:#c7d4f9; font-size:1rem; max-width:720px; line-height:1.7; }
        .metric-card { background:#fff; border:1px solid var(--line); border-radius:15px; padding:17px 18px; min-height:102px; }
        .metric-label { color:var(--muted); font-size:.74rem; font-weight:700; letter-spacing:.07em; text-transform:uppercase; }
        .metric-value { color:var(--ink); font-size:1.8rem; font-weight:800; letter-spacing:-.06em; margin-top:7px; }
        .metric-note { color:#74809a; font-size:.76rem; margin-top:4px; }
        .section-card { background:#fff; border:1px solid var(--line); border-radius:16px; padding:21px 23px; margin:12px 0; }
        .section-card h3 { margin:0 0 8px !important; }
        .model-stack { display:grid; gap:10px; }
        .model-recommendation { background:#fff; border:1px solid var(--line); border-radius:13px; padding:16px 17px; }
        .model-topline { align-items:center; display:flex; justify-content:space-between; margin-bottom:7px; }
        .model-topline strong { color:var(--ink); font-size:.96rem; }
        .model-tag { border-radius:99px; font-size:.68rem; font-weight:800; letter-spacing:.04em; padding:4px 7px; }
        .model-tag.blue { background:#eaf0ff; color:#2457f5; }
        .model-tag.green { background:#e7f8f2; color:#168465; }
        .model-recommendation p { color:var(--muted); font-size:.81rem; line-height:1.55; margin:0; }
        .pipeline { align-items:center; background:#fff; border:1px solid var(--line); border-radius:14px; display:flex; gap:0; min-height:184px; padding:18px 12px; }
        .pipeline-step { flex:1; min-width:0; position:relative; text-align:center; }
        .pipeline-step:not(:last-child):after { background:#d6deee; content:''; height:2px; left:61%; position:absolute; right:-39%; top:14px; z-index:0; }
        .pipeline-node { align-items:center; background:#eef3ff; border:2px solid #cbd8ff; border-radius:50%; color:#2457f5; display:flex; font-family:'DM Mono',monospace; font-size:.67rem; font-weight:700; height:29px; justify-content:center; margin:0 auto 9px; position:relative; width:29px; z-index:1; }
        .pipeline-step:last-child .pipeline-node { background:#e8f8f2; border-color:#b9ebdc; color:#168465; }
        .pipeline-label { color:#4c5871; font-size:.7rem; line-height:1.35; padding:0 3px; }
        .project-intro { background:linear-gradient(115deg,#ffffff 0%,#f2f6ff 72%,#e8f0ff 100%); border:1px solid #dbe4f7; border-radius:16px; margin:0 0 22px; overflow:hidden; padding:27px 30px; position:relative; }
        .project-intro:before { background:#315efb; border-radius:99px; content:''; height:58px; left:0; position:absolute; top:29px; width:5px; }
        .project-intro:after { background:radial-gradient(circle,rgba(49,94,251,.17) 0 2px,transparent 3px); background-size:16px 16px; content:''; height:126px; opacity:.75; position:absolute; right:-10px; top:-6px; width:176px; }
        .project-intro h1 { font-size:2rem !important; margin:5px 0 8px !important; }
        .project-intro p { color:var(--muted); font-size:.94rem; line-height:1.65; margin:0; max-width:760px; }
        .project-kicker { color:#315efb; font-family:'DM Mono',monospace; font-size:.72rem; font-weight:700; letter-spacing:.1em; text-transform:uppercase; }
        .project-meta { align-items:center; display:flex; gap:8px; margin-top:17px; position:relative; z-index:1; }
        .project-pill { background:#fff; border:1px solid #d7e1f7; border-radius:99px; color:#4d5c79; font-size:.72rem; font-weight:700; padding:6px 9px; }
        .project-pill.active { background:#e9f8f2; border-color:#c8ecdf; color:#168465; }
        .info-grid { display:grid; gap:12px; grid-template-columns:repeat(2, minmax(0,1fr)); margin:22px 0 4px; }
        .info-panel { background:#fff; border:1px solid var(--line); border-radius:14px; padding:19px; }
        .info-panel h3 { font-size:1rem !important; margin:0 0 13px !important; }
        .model-list, .feature-list { display:grid; gap:8px; }
        .model-item, .feature-item { align-items:flex-start; display:flex; gap:9px; }
        .model-code { align-items:center; background:#edf2ff; border-radius:6px; color:#315efb; display:flex; flex:0 0 29px; font-family:'DM Mono',monospace; font-size:.67rem; font-weight:700; height:25px; justify-content:center; }
        .feature-dot { background:#20a886; border-radius:50%; flex:0 0 6px; height:6px; margin-top:7px; }
        .model-item strong, .feature-item strong { color:var(--ink); display:block; font-size:.84rem; margin-bottom:2px; }
        .model-item span, .feature-item span { color:var(--muted); font-size:.78rem; line-height:1.45; }
        @media (max-width: 760px) { .info-grid { grid-template-columns:1fr; } .project-intro { padding:21px; } }
        @media (max-width: 760px) { .pipeline { align-items:stretch; flex-direction:column; gap:0; min-height:0; padding:10px 15px; } .pipeline-step { align-items:center; display:flex; gap:10px; text-align:left; } .pipeline-step:not(:last-child):after { height:20px; left:14px; right:auto; top:28px; width:2px; } .pipeline-node { flex:0 0 29px; margin:4px 0; } .pipeline-label { padding:0; } }
        .stButton > button, .stDownloadButton > button { background:var(--blue); border:0; border-radius:9px; color:#fff; font-family:'Manrope',sans-serif; font-weight:700; padding:.6rem 1rem; }
        .stButton > button:hover, .stDownloadButton > button:hover { background:#173fca; color:#fff; }
        [data-testid='stDataFrame'] { border:1px solid var(--line); border-radius:12px; overflow:hidden; }
        .stAlert { border-radius:10px; margin-top:14px; }
        .small-note { color:var(--muted); font-size:.86rem; line-height:1.6; }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_style()


TELCO_DATASET_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"


@st.cache_data(show_spinner=False)
def load_default_dataset():
    try:
        return pd.read_csv(TELCO_DATASET_URL)
    except Exception:
        return pd.DataFrame()


def read_uploaded_data(uploaded):
    if uploaded.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded)
    return pd.read_excel(uploaded)


def init_state():
    defaults = {
        "raw_data": load_default_dataset(),
        "dataset_name": "IBM Telco Customer Churn (default)",
        "target": "Churn",
        "prepared": None,
        "results": None,
        "models": {},
        "training_signature": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()

# Session lama bisa belum memiliki key navigasi setelah aplikasi diperbarui.
if "nav_page" not in st.session_state:
    st.session_state.nav_page = "Ringkasan Proyek"


def use_default_dataset():
    """Reset workspace to bundled Telco dataset."""
    data = load_default_dataset()
    if data.empty:
        raise FileNotFoundError("File telco_customer_churn.csv tidak ditemukan atau tidak dapat dibaca.")
    st.session_state.raw_data = data
    st.session_state.dataset_name = "IBM Telco Customer Churn (default)"
    st.session_state.target = "Churn"
    st.session_state.prepared = None
    st.session_state.results = None
    st.session_state.models = {}
    st.session_state.pop("importance", None)
    st.session_state.dataset_notice = "Dataset Telco default aktif: 7.043 pelanggan, 21 kolom, target Churn."


def metric_card(label, value, note=""):
    st.markdown(
        f"<div class='metric-card'><div class='metric-label'>{label}</div>"
        f"<div class='metric-value'>{value}</div><div class='metric-note'>{note}</div></div>",
        unsafe_allow_html=True,
    )


def target_options(df):
    return [column for column in df.columns if df[column].nunique(dropna=True) >= 2]


def infer_id_columns(df):
    result = []
    for column in df.columns:
        name_hint = any(token in column.lower() for token in ("id", "code", "number"))
        near_unique = df[column].nunique(dropna=True) >= len(df) * 0.95
        if name_hint or near_unique:
            result.append(column)
    return result


def binary_target(y):
    classes = list(pd.Series(y).dropna().unique())
    if len(classes) != 2:
        raise ValueError("Kolom target harus memiliki tepat dua kelas.")
    positive_candidates = {"yes", "y", "1", "true", "churn", "positive"}
    positive = next((x for x in classes if str(x).strip().lower() in positive_candidates), classes[-1])
    negative = next(x for x in classes if x != positive)
    encoded = (pd.Series(y) == positive).astype(int)
    return encoded, {0: str(negative), 1: str(positive)}


def make_preprocessor(X):
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = [col for col in X.columns if col not in numeric]
    transformers = []
    if numeric:
        transformers.append(
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric)
        )
    if categorical:
        transformers.append(
            ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))]), categorical)
        )
    return ColumnTransformer(transformers=transformers, remainder="drop"), numeric, categorical


def prepare_data(df, target, drop_cols, test_size, random_state):
    data = df.copy()
    data = data.drop_duplicates().replace(r"^\s*$", np.nan, regex=True)
    data = data.dropna(subset=[target])
    y, labels = binary_target(data[target])
    X = data.drop(columns=list(set(drop_cols + [target])), errors="ignore")
    if X.empty:
        raise ValueError("Tidak ada fitur tersisa setelah kolom target dan ID dibuang.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return {
        "X": X,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "labels": labels,
        "numeric": X.select_dtypes(include=np.number).columns.tolist(),
        "categorical": X.select_dtypes(exclude=np.number).columns.tolist(),
        "clean_rows": len(data),
        "removed_duplicates": len(df) - len(data),
    }


def get_models(random_state, tree_depth, forest_trees, forest_depth, balanced):
    weight = "balanced" if balanced else None
    return {
        "Dummy Majority": DummyClassifier(strategy="most_frequent"),
        "SVM Linear": SVC(kernel="linear", C=1.0, probability=True, random_state=random_state, class_weight=weight),
        "SVM RBF": SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=random_state, class_weight=weight),
        "Decision Tree": DecisionTreeClassifier(criterion="gini", max_depth=tree_depth, random_state=random_state, class_weight=weight),
        "Random Forest": RandomForestClassifier(n_estimators=forest_trees, max_depth=forest_depth, random_state=random_state, class_weight=weight, n_jobs=-1),
    }


def train_models(prepared, model_names, settings):
    stored, rows = {}, []
    for name, estimator in get_models(**settings).items():
        if name not in model_names:
            continue
        preprocessor, _, _ = make_preprocessor(prepared["X"])
        pipe = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
        started = time.perf_counter()
        pipe.fit(prepared["X_train"], prepared["y_train"])
        elapsed = time.perf_counter() - started
        test_prob = pipe.predict_proba(prepared["X_test"])[:, 1]
        train_prob = pipe.predict_proba(prepared["X_train"])[:, 1]
        pred = (test_prob >= 0.5).astype(int)
        test_auc = roc_auc_score(prepared["y_test"], test_prob) if len(np.unique(prepared["y_test"])) == 2 else np.nan
        train_auc = roc_auc_score(prepared["y_train"], train_prob)
        stored[name] = pipe
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(prepared["y_test"], pred),
            "Precision": precision_score(prepared["y_test"], pred, zero_division=0),
            "Recall": recall_score(prepared["y_test"], pred, zero_division=0),
            "F1-Score": f1_score(prepared["y_test"], pred, zero_division=0),
            "ROC-AUC": test_auc,
            "Train ROC-AUC": train_auc,
            "Gap ROC-AUC": train_auc - test_auc,
            "Waktu Training (s)": elapsed,
        })
    return stored, pd.DataFrame(rows)


def page_overview():
    df = st.session_state.raw_data
    target = st.session_state.target
    if df.empty:
        st.error("Dataset default gagal dimuat. Unggah dataset pada menu Dataset.")
        return
    st.markdown("<div class='project-intro'><div class='project-kicker'>Customer Churn Classification</div><h1>Machine Learning Project</h1><p>Dashboard untuk menganalisis risiko customer churn dari data pelanggan. Mulai dari eksplorasi data, pelatihan model, evaluasi hasil, sampai prediksi dan export data.</p><div class='project-meta'><span class='project-pill active'>DATASET READY</span><span class='project-pill'>BINARY CLASSIFICATION</span></div></div>", unsafe_allow_html=True)
    churn_rate = df[target].value_counts(normalize=True).min() * 100 if target in df else 0
    cols = st.columns(4)
    with cols[0]: metric_card("Records", f"{len(df):,}", st.session_state.dataset_name)
    with cols[1]: metric_card("Features", str(max(len(df.columns) - 1, 0)), "Kolom selain target")
    with cols[2]: metric_card("Target", target, f"{df[target].nunique()} kelas")
    with cols[3]: metric_card("Minority share", f"{churn_rate:.1f}%", "Perlu lihat recall, bukan accuracy saja")
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("## Keputusan model")
        st.markdown("<div class='model-stack'><div class='model-recommendation primary'><div class='model-topline'><strong>Random Forest</strong><span class='model-tag blue'>RANKING RISIKO</span></div><p>Accuracy 0.8001 dan ROC-AUC 0.8458 tertinggi pada eksperimen Telco. Pilih untuk mengurutkan pelanggan berdasarkan risiko churn.</p></div><div class='model-recommendation recall'><div class='model-topline'><strong>SVM Linear</strong><span class='model-tag green'>TANGKAP CHURN</span></div><p>Recall 0.5268 dan F1-score 0.5701 tertinggi. Pilih saat lebih banyak pelanggan churn perlu terdeteksi.</p></div></div>", unsafe_allow_html=True)
    with right:
        st.markdown("## Alur kerja")
        st.markdown("<div class='pipeline'><div class='pipeline-step'><div class='pipeline-node'>01</div><div class='pipeline-label'>Data &amp; target</div></div><div class='pipeline-step'><div class='pipeline-node'>02</div><div class='pipeline-label'>Bersihkan data</div></div><div class='pipeline-step'><div class='pipeline-node'>03</div><div class='pipeline-label'>Latih model</div></div><div class='pipeline-step'><div class='pipeline-node'>04</div><div class='pipeline-label'>Evaluasi</div></div><div class='pipeline-step'><div class='pipeline-node'>05</div><div class='pipeline-label'>Prediksi</div></div></div>", unsafe_allow_html=True)
    st.markdown("<div class='info-grid'><div class='info-panel'><h3>Model yang digunakan</h3><div class='model-list'><div class='model-item'><div class='model-code'>01</div><div><strong>Dummy Majority</strong><span>Baseline untuk membandingkan hasil model utama.</span></div></div><div class='model-item'><div class='model-code'>02</div><div><strong>SVM Linear &amp; RBF</strong><span>Menguji pola linear dan non-linear pada data pelanggan.</span></div></div><div class='model-item'><div class='model-code'>03</div><div><strong>Decision Tree</strong><span>Model sederhana yang mudah dijelaskan.</span></div></div><div class='model-item'><div class='model-code'>04</div><div><strong>Random Forest</strong><span>Ensemble untuk prediksi risiko yang lebih stabil.</span></div></div></div></div><div class='info-panel'><h3>Fitur website</h3><div class='feature-list'><div class='feature-item'><span class='feature-dot'></span><div><strong>Dataset fleksibel</strong><span>Upload CSV atau Excel, lalu pilih kolom target.</span></div></div><div class='feature-item'><span class='feature-dot'></span><div><strong>Eksplorasi data</strong><span>Lihat distribusi target, missing value, dan pola fitur.</span></div></div><div class='feature-item'><span class='feature-dot'></span><div><strong>Evaluasi model</strong><span>Bandingkan accuracy, precision, recall, F1-score, dan ROC-AUC.</span></div></div><div class='feature-item'><span class='feature-dot'></span><div><strong>Prediksi dan export</strong><span>Prediksi satu data atau batch, lalu unduh hasil CSV.</span></div></div></div></div></div>", unsafe_allow_html=True)
    st.info("Model bukan keputusan bisnis final. Pakai threshold sesuai biaya promo, nilai pelanggan, dan target retensi.")


def page_dataset():
    st.markdown("<div class='eyebrow'>01 / Data source</div>", unsafe_allow_html=True)
    st.title("Dataset & target")
    st.caption("Pilih data rujukan penelitian atau unggah data klasifikasi biner milikmu.")
    if st.session_state.get("dataset_notice"):
        st.success(st.session_state.pop("dataset_notice"))
    default_col, upload_col = st.columns(2, gap="large")
    with default_col:
        st.markdown("<div class='dataset-card'><span class='tag'>DATASET RUJUKAN</span><h3>Telco Customer Churn</h3><p>Dataset lokal IBM Telco. Berisi 7.043 pelanggan, 21 kolom, dan target <b>Churn</b>. Dataset sama seperti dokumen penelitian, tersedia tanpa koneksi internet.</p></div>", unsafe_allow_html=True)
        if st.button("Gunakan dataset Telco default", type="primary", use_container_width=True):
            try:
                use_default_dataset()
            except FileNotFoundError as error:
                st.error(str(error))
                return
            st.session_state.nav_page = "Data dan Eksplorasi"
            st.rerun()
    with upload_col:
        st.markdown("<div class='upload-card'><span class='tag'>DATASET SENDIRI</span><h3>Unggah dataset</h3><p>Gunakan CSV, XLSX, atau XLS. Pilih kolom target setelah file dimuat. Sistem akan membangun form, visualisasi, dan pipeline mengikuti struktur data.</p></div>", unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload file", type=["csv", "xlsx", "xls"], label_visibility="collapsed", help="CSV atau Excel. Baris adalah record, kolom adalah fitur.")
        if uploaded:
            try:
                data = read_uploaded_data(uploaded)
                if st.button("Gunakan dataset upload", use_container_width=True):
                    options = target_options(data)
                    if not options:
                        st.error("Dataset perlu memiliki minimal satu kolom dengan dua kelas atau lebih.")
                    else:
                        st.session_state.raw_data = data
                        st.session_state.dataset_name = uploaded.name
                        st.session_state.target = "Churn" if "Churn" in options else options[-1]
                        st.session_state.prepared = None
                        st.session_state.results = None
                        st.session_state.models = {}
                        st.session_state.pop("importance", None)
                        st.rerun()
            except Exception as error:
                st.error(f"File tidak dapat dibaca: {error}")
    df = st.session_state.raw_data
    if df.empty:
        return
    options = target_options(df)
    index = options.index(st.session_state.target) if st.session_state.target in options else 0
    selected = st.selectbox("Kolom target klasifikasi", options, index=index)
    if selected != st.session_state.target:
        st.session_state.target = selected
        st.session_state.prepared = None
        st.session_state.results = None
        st.session_state.models = {}
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Baris", f"{len(df):,}", st.session_state.dataset_name)
    with c2: metric_card("Kolom", str(len(df.columns)), "Termasuk target")
    with c3: metric_card("Missing", f"{int(df.isna().sum().sum()):,}", "Sel kosong awal")
    with c4: metric_card("Duplikat", f"{int(df.duplicated().sum()):,}", "Baris sama persis")
    tabs = st.tabs(["Preview", "Skema", "Kualitas"])
    with tabs[0]: st.dataframe(df, use_container_width=True, height=380)
    with tabs[1]:
        schema = pd.DataFrame({"Kolom": df.columns, "Tipe": df.dtypes.astype(str).values, "Unique": [df[c].nunique(dropna=True) for c in df.columns], "Missing": df.isna().sum().values})
        st.dataframe(schema, use_container_width=True, hide_index=True)
    with tabs[2]:
        missing = df.isna().sum().sort_values(ascending=False).reset_index()
        missing.columns = ["Kolom", "Missing"]
        st.plotly_chart(px.bar(missing[missing["Missing"] > 0], x="Missing", y="Kolom", orientation="h", title="Missing values"), use_container_width=True)


def page_preprocess():
    st.markdown("<div class='eyebrow'>02 / Reliable inputs</div>", unsafe_allow_html=True)
    st.title("Preprocessing")
    df, target = st.session_state.raw_data, st.session_state.target
    if df.empty:
        st.warning("Muat dataset dulu.")
        return
    suggested = [x for x in infer_id_columns(df) if x != target]
    dropped = st.multiselect("Kolom identitas atau kolom yang dibuang", [x for x in df.columns if x != target], default=suggested)
    a, b, c = st.columns(3)
    with a: test_size = st.slider("Porsi data uji", .15, .40, .25, .05)
    with b: random_state = st.number_input("Random state", min_value=0, value=42, step=1)
    with c: st.metric("Metode split", "Stratified 75:25", "Distribusi kelas dijaga")
    st.markdown("<div class='section-card'><h3>Pipeline aman</h3><p class='small-note'>Duplikat dibuang. Nilai kosong numerik diisi median. Kategorikal diisi modus lalu one-hot encoded. StandardScaler diterapkan pada fitur numerik di dalam pipeline training, sehingga test data tidak bocor ke proses fit.</p></div>", unsafe_allow_html=True)
    if st.button("Siapkan data", type="primary"):
        try:
            prepared = prepare_data(df, target, dropped, test_size, int(random_state))
            st.session_state.prepared = prepared
            st.session_state.results = None
            st.session_state.models = {}
            st.session_state.training_signature = None
            st.success("Data siap untuk pelatihan.")
        except ValueError as error:
            st.error(str(error))
    prepared = st.session_state.prepared
    if prepared:
        x, y, z, q = st.columns(4)
        with x: metric_card("Fitur awal", str(prepared["X"].shape[1]), "Sebelum encoding")
        with y: metric_card("Numerik", str(len(prepared["numeric"])), "Di-standardisasi")
        with z: metric_card("Kategorikal", str(len(prepared["categorical"])), "One-hot encoding")
        with q: metric_card("Train / Test", f"{len(prepared['X_train']):,} / {len(prepared['X_test']):,}", "Split stratified")
        distribution = pd.DataFrame({"Kelas": [prepared["labels"][0], prepared["labels"][1]], "Train": [int((prepared["y_train"] == 0).sum()), int((prepared["y_train"] == 1).sum())], "Test": [int((prepared["y_test"] == 0).sum()), int((prepared["y_test"] == 1).sum())]})
        st.plotly_chart(px.bar(distribution, x="Kelas", y=["Train", "Test"], barmode="group", title="Distribusi target sesudah split"), use_container_width=True)


def page_eda():
    st.markdown("<div class='eyebrow'>03 / Pattern scan</div>", unsafe_allow_html=True)
    st.title("Exploratory analysis")
    df, target = st.session_state.raw_data.copy(), st.session_state.target
    if df.empty:
        return
    left, right = st.columns(2, gap="large")
    with left:
        counts = df[target].fillna("Missing").astype(str).value_counts().reset_index()
        counts.columns = [target, "Jumlah"]
        st.plotly_chart(px.pie(counts, names=target, values="Jumlah", hole=.63, title="Distribusi target", color_discrete_sequence=["#2457f5", "#28c7a6", "#ffb86b"]), use_container_width=True)
    with right:
        missing = df.isna().sum().sort_values(ascending=False).head(12).reset_index()
        missing.columns = ["Kolom", "Missing"]
        st.plotly_chart(px.bar(missing, x="Missing", y="Kolom", orientation="h", title="Data quality: missing per kolom", color_discrete_sequence=["#ff8c69"]), use_container_width=True)
    numeric = df.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in df.columns if c not in numeric and c != target]
    c1, c2 = st.columns(2, gap="large")
    with c1:
        if numeric:
            col = st.selectbox("Fitur numerik", numeric)
            st.plotly_chart(px.histogram(df, x=col, color=target, barmode="overlay", marginal="box", title=f"Distribusi {col} menurut {target}", color_discrete_sequence=["#2457f5", "#28c7a6"]), use_container_width=True)
    with c2:
        if categorical:
            col = st.selectbox("Fitur kategorikal", categorical)
            rate = pd.crosstab(df[col].fillna("Missing"), df[target], normalize="index") * 100
            st.plotly_chart(px.bar(rate.reset_index().melt(id_vars=col, var_name=target, value_name="Persen"), x=col, y="Persen", color=target, barmode="group", title=f"Komposisi {target} menurut {col}", color_discrete_sequence=["#2457f5", "#28c7a6"]), use_container_width=True)
    if len(numeric) >= 2:
        st.plotly_chart(px.imshow(df[numeric].corr(numeric_only=True), text_auto=".2f", color_continuous_scale="Blues", title="Korelasi fitur numerik"), use_container_width=True)


def page_training():
    st.markdown("<div class='eyebrow'>04 / Model lab</div>", unsafe_allow_html=True)
    st.title("Latih model")
    if not st.session_state.prepared:
        st.warning("Siapkan data pada menu Preprocessing dulu.")
        return
    selected = st.multiselect("Model pembanding", list(get_models(42, 5, 200, 8, False).keys()), default=["Dummy Majority", "SVM Linear", "SVM RBF", "Decision Tree", "Random Forest"])
    a, b, c, d = st.columns(4)
    with a: tree_depth = st.slider("Tree depth", 2, 16, 5)
    with b: forest_trees = st.slider("Forest trees", 50, 500, 200, 50)
    with c: forest_depth = st.slider("Forest depth", 2, 20, 8)
    with d: balanced = st.toggle("Class weight balanced", value=False)
    st.caption("Default sesuai dokumen: Decision Tree depth 5, Random Forest 200 trees depth 8, SVM C=1.0.")
    if st.button("Mulai pelatihan", type="primary", disabled=not selected):
        settings = {"random_state": 42, "tree_depth": tree_depth, "forest_trees": forest_trees, "forest_depth": forest_depth, "balanced": balanced}
        with st.spinner("Melatih model dan menghitung metrik..."):
            models, results = train_models(st.session_state.prepared, selected, settings)
        st.session_state.models = models
        st.session_state.results = results
        st.success("Pelatihan selesai.")
    if st.session_state.results is not None:
        metric_formats = {
            column: ".4f"
            for column in st.session_state.results.columns
            if column not in ["Model", "Waktu Training (s)"]
        }
        styled_results = st.session_state.results.style.format(metric_formats).format({"Waktu Training (s)": ".3f"})
        st.dataframe(styled_results, use_container_width=True, hide_index=True)


def trained():
    return st.session_state.results is not None and bool(st.session_state.models)


def page_evaluation():
    st.markdown("<div class='eyebrow'>05 / Model scorecard</div>", unsafe_allow_html=True)
    st.title("Evaluasi model")
    if not trained():
        st.warning("Latih setidaknya satu model dulu.")
        return
    results = st.session_state.results
    cards = st.columns(4)
    for holder, metric in zip(cards, ["Accuracy", "Recall", "F1-Score", "ROC-AUC"]):
        row = results.loc[results[metric].idxmax()]
        with holder: metric_card(f"Best {metric}", row["Model"], f"{row[metric]:.4f}")
    shown = results.copy()
    for col in shown.columns:
        if col != "Model": shown[col] = shown[col].map(lambda x: f"{x:.4f}")
    st.dataframe(shown, use_container_width=True, hide_index=True)
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    chart_data = results.melt(id_vars="Model", value_vars=metrics, var_name="Metrik", value_name="Nilai")
    st.plotly_chart(px.bar(chart_data, x="Model", y="Nilai", color="Metrik", barmode="group", title="Perbandingan performa", color_discrete_sequence=px.colors.qualitative.Safe), use_container_width=True)
    gap = results.loc[results["Gap ROC-AUC"].idxmax()]
    st.info(f"Gap ROC-AUC terbesar: {gap['Model']} ({gap['Gap ROC-AUC']:.4f}). Gap besar perlu dicek dengan cross-validation sebelum dipakai sebagai keputusan.")


def model_probability(name):
    prepared = st.session_state.prepared
    return st.session_state.models[name].predict_proba(prepared["X_test"])[:, 1]


def page_threshold():
    st.markdown("<div class='eyebrow'>06 / Decision threshold</div>", unsafe_allow_html=True)
    st.title("Confusion matrix & threshold")
    if not trained():
        st.warning("Latih model dulu.")
        return
    prepared = st.session_state.prepared
    name = st.selectbox("Model", list(st.session_state.models))
    threshold = st.slider("Threshold churn", .05, .95, .50, .05)
    probs, actual = model_probability(name), prepared["y_test"].to_numpy()
    pred = (probs >= threshold).astype(int)
    cm = confusion_matrix(actual, pred, labels=[0, 1])
    a, b = st.columns([.8, 1.2], gap="large")
    with a:
        fig = px.imshow(cm, text_auto=True, x=[f"Prediksi {prepared['labels'][0]}", f"Prediksi {prepared['labels'][1]}"], y=[f"Aktual {prepared['labels'][0]}", f"Aktual {prepared['labels'][1]}"], color_continuous_scale="Blues", title="Confusion matrix")
        st.plotly_chart(fig, use_container_width=True)
    with b:
        k1, k2, k3, k4 = st.columns(4)
        for box, label, value in zip([k1,k2,k3,k4], ["TN", "FP", "FN", "TP"], cm.ravel()):
            with box: metric_card(label, str(value))
        st.markdown(f"<div class='section-card'><h3>At threshold {threshold:.2f}</h3><p class='small-note'>Precision {precision_score(actual, pred, zero_division=0):.3f} | Recall {recall_score(actual, pred, zero_division=0):.3f} | F1 {f1_score(actual, pred, zero_division=0):.3f}</p><p class='small-note'>False negative berarti pelanggan churn tidak masuk prioritas retensi. Turunkan threshold untuk menaikkan recall, dengan konsekuensi promo lebih banyak ke non-churn.</p></div>", unsafe_allow_html=True)
    thresholds = np.arange(.10, .91, .05)
    rows = []
    for t in thresholds:
        p = (probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(actual, p, labels=[0, 1]).ravel()
        rows.append({"Threshold": t, "Accuracy": accuracy_score(actual,p), "Precision": precision_score(actual,p,zero_division=0), "Recall": recall_score(actual,p,zero_division=0), "F1-Score": f1_score(actual,p,zero_division=0), "FP":fp, "FN":fn, "TP":tp, "TN":tn})
    table = pd.DataFrame(rows)
    lines = table.melt(id_vars="Threshold", value_vars=["Precision", "Recall", "F1-Score"], var_name="Metrik", value_name="Nilai")
    st.plotly_chart(px.line(lines, x="Threshold", y="Nilai", color="Metrik", markers=True, title="Trade-off threshold"), use_container_width=True)
    st.dataframe(table.style.format({x:".4f" for x in ["Accuracy","Precision","Recall","F1-Score"]}), use_container_width=True, hide_index=True)


def page_curves():
    st.markdown("<div class='eyebrow'>07 / Separation quality</div>", unsafe_allow_html=True)
    st.title("ROC & precision-recall")
    if not trained():
        st.warning("Latih model dulu.")
        return
    actual = st.session_state.prepared["y_test"]
    roc_fig, pr_fig = go.Figure(), go.Figure()
    for name in st.session_state.models:
        probability = model_probability(name)
        fpr, tpr, _ = roc_curve(actual, probability)
        precision, recall, _ = precision_recall_curve(actual, probability)
        roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} ({auc(fpr,tpr):.3f})"))
        pr_fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name=name))
    roc_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", line=dict(dash="dash", color="#9ba4b8"), name="Random"))
    roc_fig.update_layout(title="ROC curve", xaxis_title="False positive rate", yaxis_title="True positive rate", legend_title="Model (AUC)")
    pr_fig.update_layout(title="Precision-recall curve", xaxis_title="Recall", yaxis_title="Precision")
    a, b = st.columns(2)
    with a: st.plotly_chart(roc_fig, use_container_width=True)
    with b: st.plotly_chart(pr_fig, use_container_width=True)


def page_importance():
    st.markdown("<div class='eyebrow'>08 / Explainability</div>", unsafe_allow_html=True)
    st.title("Fitur berpengaruh")
    if not trained():
        st.warning("Latih model dulu.")
        return
    available = [name for name in st.session_state.models if name not in ["Dummy Majority"]]
    name = st.selectbox("Model untuk permutation importance", available)
    if st.button("Hitung permutation importance", type="primary"):
        with st.spinner("Mengacak fitur satu per satu pada data uji..."):
            result = permutation_importance(st.session_state.models[name], st.session_state.prepared["X_test"], st.session_state.prepared["y_test"], n_repeats=8, random_state=42, scoring="roc_auc", n_jobs=-1)
        importance = pd.DataFrame({"Fitur": st.session_state.prepared["X_test"].columns, "Importance": result.importances_mean}).sort_values("Importance", ascending=False).head(15)
        st.session_state.importance = {"model": name, "data": importance}
    if "importance" in st.session_state and st.session_state.importance["model"] == name:
        importance = st.session_state.importance["data"]
        st.plotly_chart(px.bar(importance.sort_values("Importance"), x="Importance", y="Fitur", orientation="h", title=f"Top feature: {name}", color="Importance", color_continuous_scale="Blues"), use_container_width=True)
        st.caption("Permutation importance mengukur penurunan ROC-AUC saat satu fitur diacak. Nilai tinggi berarti fitur penting untuk prediksi.")


def page_predict_export():
    st.markdown("<div class='eyebrow'>09 / Operational output</div>", unsafe_allow_html=True)
    st.title("Prediksi & ekspor")
    if not trained():
        st.warning("Latih model dulu.")
        return
    prepared = st.session_state.prepared
    name = st.selectbox("Model prediksi", list(st.session_state.models))
    threshold = st.slider("Threshold untuk output", .05, .95, .50, .05, key="export_threshold")
    tab1, tab2 = st.tabs(["Satu record", "Batch upload"])
    with tab1:
        values = {}
        with st.form("single_prediction"):
            columns = st.columns(2)
            for i, feature in enumerate(prepared["X"].columns):
                with columns[i % 2]:
                    series = prepared["X"][feature]
                    if feature in prepared["numeric"]:
                        default = float(series.median()) if not series.dropna().empty else 0.0
                        values[feature] = st.number_input(feature, value=default)
                    else:
                        options = sorted(series.dropna().astype(str).unique().tolist())
                        values[feature] = st.selectbox(feature, options or ["Unknown"])
            submitted = st.form_submit_button("Prediksi risiko")
        if submitted:
            row = pd.DataFrame([values])
            probability = st.session_state.models[name].predict_proba(row)[0, 1]
            label = prepared["labels"][1] if probability >= threshold else prepared["labels"][0]
            risk = "Tinggi" if probability >= .60 else "Sedang" if probability >= .30 else "Rendah"
            st.markdown(f"<div class='section-card'><h3>{label}</h3><p class='small-note'>Probabilitas kelas positif: <b>{probability:.1%}</b> | Risiko: <b>{risk}</b> | Threshold: {threshold:.2f}</p></div>", unsafe_allow_html=True)
    with tab2:
        uploaded = st.file_uploader("Unggah data untuk prediksi", type=["csv", "xlsx", "xls"], key="batch")
        if uploaded:
            try:
                batch = read_uploaded_data(uploaded)
                expected = prepared["X"].columns.tolist()
                missing = [c for c in expected if c not in batch]
                if missing:
                    st.error("Kolom wajib tidak ada: " + ", ".join(missing))
                else:
                    input_data = batch[expected]
                    probability = st.session_state.models[name].predict_proba(input_data)[:, 1]
                    output = batch.copy()
                    output["prediction_probability"] = probability
                    output["prediction"] = np.where(probability >= threshold, prepared["labels"][1], prepared["labels"][0])
                    output["risk_level"] = pd.cut(probability, [-.01,.30,.60,1], labels=["Rendah","Sedang","Tinggi"])
                    st.dataframe(output.sort_values("prediction_probability", ascending=False), use_container_width=True)
                    st.download_button("Unduh hasil CSV", output.to_csv(index=False).encode("utf-8"), "hasil_prediksi_churn.csv", "text/csv")
            except Exception as error:
                st.error(f"Prediksi batch gagal: {error}")


def page_data_explore():
    dataset_tab, explore_tab = st.tabs(["Dataset", "Eksplorasi Data"])
    with dataset_tab:
        page_dataset()
    with explore_tab:
        page_eda()


def page_train_evaluate():
    train_tab, result_tab, threshold_tab, insight_tab = st.tabs(["Latih Model", "Hasil Model", "Threshold & Kurva", "Fitur Penting"])
    with train_tab:
        page_training()
    with result_tab:
        page_evaluation()
    with threshold_tab:
        page_threshold()
        page_curves()
    with insight_tab:
        page_importance()


PAGES = {
    "Ringkasan Proyek": page_overview,
    "Data dan Eksplorasi": page_data_explore,
    "Preprocessing": page_preprocess,
    "Model dan Hasil": page_train_evaluate,
    "Prediksi dan Export": page_predict_export,
}

with st.sidebar:
    st.markdown("## CHURN / CI")
    st.caption("Customer retention workspace")
    for page_name in PAGES:
        active = st.session_state.nav_page == page_name
        if st.button(page_name, key=f"nav_{page_name}", use_container_width=True, type="primary" if active else "secondary"):
            st.session_state.nav_page = page_name
            st.rerun()
    st.divider()
    df = st.session_state.raw_data
    st.caption("DATASET AKTIF")
    st.write(st.session_state.dataset_name)
    if not df.empty:
        st.caption(f"{len(df):,} records / {len(df.columns)} columns")
    st.caption("MODEL STATUS")
    st.write("Ready" if trained() else "Belum dilatih")

PAGES[st.session_state.nav_page]()
