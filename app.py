# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ====== Dependências opcionais ======
HAVE_SKLEARN = True
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
except Exception:
    HAVE_SKLEARN = False

HAVE_SCIPY = True
try:
    from scipy import stats
except Exception:
    HAVE_SCIPY = False

# =========================
# Configuração da página
# =========================
st.set_page_config(page_title="Análise de Dados CSV", page_icon="📊", layout="wide")
plt.style.use('dark_background')

# =========================
# Estado da sessão
# =========================
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'data_context' not in st.session_state:
    st.session_state.data_context = None

# =========================
# IA - chamada com fallback
# =========================
def call_ai_api(api_choice, api_key, messages, model):
    """
    Lazy-import dos SDKs e compatibilidade com versões antigas/novas.
    Retorna string com a resposta ou uma mensagem de erro amigável.
    """
    try:
        if api_choice == "OpenAI":
            try:
                import openai as openai_pkg
            except Exception:
                return "Erro: o pacote 'openai' não está instalado. Rode: pip install -U openai"

            # SDK novo (>=1.0) ou legado
            if hasattr(openai_pkg, "OpenAI"):
                try:
                    client = openai_pkg.OpenAI(api_key=api_key)
                    resp = client.chat.completions.create(
                        model=model, messages=messages, max_tokens=2000, temperature=0.7
                    )
                    return resp.choices[0].message.content
                except Exception:
                    try:  # fallback p/ legado
                        openai_pkg.api_key = api_key
                        resp = openai_pkg.ChatCompletion.create(
                            model=model, messages=messages, max_tokens=2000, temperature=0.7
                        )
                        return resp.choices[0].message["content"]
                    except Exception as e2:
                        return f"Erro OpenAI: {e2}"
            else:
                try:
                    openai_pkg.api_key = api_key
                    resp = openai_pkg.ChatCompletion.create(
                        model=model, messages=messages, max_tokens=2000, temperature=0.7
                    )
                    return resp.choices[0].message["content"]
                except Exception as e:
                    return f"Erro OpenAI: {e}"

        elif api_choice == "Groq":
            try:
                from groq import Groq
            except Exception:
                return "Erro: o pacote 'groq' não está instalado. Rode: pip install -U groq"
            try:
                client = Groq(api_key=api_key)
                resp = client.chat.completions.create(
                    model=model, messages=messages, max_tokens=2000, temperature=0.7
                )
                return resp.choices[0].message.content
            except Exception as e:
                return f"Erro Groq: {e}"

        elif api_choice == "Gemini":
            try:
                import google.generativeai as genai
            except Exception:
                return "Erro: o pacote 'google-generativeai' não está instalado. Rode: pip install -U google-generativeai"
            try:
                genai.configure(api_key=api_key)
                model_instance = genai.GenerativeModel(model)
                prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
                resp = model_instance.generate_content(prompt)
                return resp.text
            except Exception as e:
                return f"Erro Gemini: {e}"

        return "Erro: API não reconhecida."
    except Exception as e:
        return f"Erro inesperado: {e}"

# =========================
# Análises - artefatos p/ chat
# =========================
def compute_analysis_artifacts(df: pd.DataFrame):
    """Calcula artefatos de TODAS as abas para o chat."""
    artifacts = {}

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    artifacts["numeric_columns"] = numeric_cols
    artifacts["categororical_columns"] = categorical_cols  # legado
    artifacts["categorical_columns"] = categorical_cols

    # Estatísticas + variância (amostral, ddof=1)
    if numeric_cols:
        desc = df[numeric_cols].describe().T
        desc["var"] = df[numeric_cols].var()
        artifacts["numeric_describe"] = (
            desc.reset_index().rename(columns={"index": "coluna"}).to_dict(orient="records")
        )
    else:
        artifacts["numeric_describe"] = []

    # Nulos
    artifacts["missing_values"] = (
        df.isnull().sum().rename("nulos").reset_index().rename(columns={"index": "coluna"}).to_dict("records")
    )

    # Top categorias (limita a 5 colunas)
    top_cats = {}
    for col in categorical_cols[:5]:
        vc = df[col].astype(str).value_counts(dropna=False).head(10)
        top_cats[col] = [{"categoria": str(i), "freq": int(v)} for i, v in vc.items()]
    artifacts["top_categories"] = top_cats

    # Correlações significativas (ordenadas por |r|)
    sig_corr = []
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr()
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                c = corr.iloc[i, j]
                if np.isfinite(c):
                    sig_corr.append({
                        "var1": corr.columns[i],
                        "var2": corr.columns[j],
                        "r": float(c),
                        "forca": "Forte" if abs(c) > 0.7 else "Moderada" if abs(c) > 0.3 else "Fraca"
                    })
        sig_corr = sorted(sig_corr, key=lambda x: abs(x["r"]), reverse=True)
    artifacts["significant_correlations"] = sig_corr

    # Outliers (IQR e z-score)
    outliers_summary = []
    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            outliers_summary.append({"variavel": col, "outliers_iqr": 0, "pct_iqr": 0.0, "outliers_z": 0, "pct_z": 0.0})
            continue
        Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
        IQR = Q3 - Q1
        lb, ub = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        iqr_mask = (df[col] < lb) | (df[col] > ub)
        n_iqr = int(iqr_mask.sum())
        pct_iqr = float(n_iqr / len(df) * 100)

        if HAVE_SCIPY:
            try:
                z = np.abs(stats.zscore(series))
                n_z = int((z > 3).sum())
            except Exception:
                n_z = 0
        else:
            n_z = 0
        pct_z = float(n_z / len(df) * 100)

        outliers_summary.append({
            "variavel": col,
            "outliers_iqr": n_iqr,
            "pct_iqr": pct_iqr,
            "outliers_z": n_z,
            "pct_z": pct_z
        })

    artifacts["outliers_summary"] = sorted(outliers_summary, key=lambda x: x["pct_iqr"], reverse=True)[:50]

    # Básicos
    artifacts["sample_data"] = df.head(5).to_dict("records")
    artifacts["shape"] = df.shape
    artifacts["columns"] = df.columns.tolist()

    # Campos de cluster (inicialmente vazios)
    artifacts.update({
        "cluster_available": False,
        "cluster_k": None,
        "cluster_silhouette": None,
        "cluster_labels": [],
        "cluster_pca_2d": [],
        "cluster_profile": [],
        "cluster_feat_names": [],
        "cluster_sample_labeled": [],
        "cluster_silhouettes_grid": {},
    })
    return artifacts

# ---------- clustering helpers ----------
def prepare_features_for_clustering(df: pd.DataFrame, numeric_cols, categorical_cols, max_cat_levels=20):
    if not HAVE_SKLEARN:
        raise RuntimeError("scikit-learn ausente")

    work = pd.DataFrame(index=df.index)

    # Numéricas (imputação mediana)
    if numeric_cols:
        num_df = df[numeric_cols].copy()
        for c in num_df.columns:
            if num_df[c].isnull().any():
                num_df[c] = num_df[c].fillna(num_df[c].median())
        work = pd.concat([work, num_df], axis=1)

    # Categóricas (one-hot com top categorias)
    cat_dfs = []
    for c in (categorical_cols or []):
        vc = df[c].astype(str).value_counts()
        keep = vc.index[:max_cat_levels]
        col = df[c].astype(str).where(df[c].astype(str).isin(keep), other="_OUTROS_")
        cat_dfs.append(pd.get_dummies(col, prefix=c, dummy_na=False))
    if cat_dfs:
        work = pd.concat([work] + cat_dfs, axis=1)

    feat_names = work.columns.tolist()

    scaler = StandardScaler(with_mean=True, with_std=True)
    X = scaler.fit_transform(work.values)
    return X, feat_names, work

def auto_kmeans(X, k_min=2, k_max=8, random_state=42):
    best_k, best_score, best_model = None, -1, None
    silhouettes = {}
    for k in range(k_min, min(k_max, len(X) - 1) + 1):
        try:
            km = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
            labels = km.fit_predict(X)
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(X, labels)
            silhouettes[k] = score
            if score > best_score:
                best_score, best_k, best_model = score, k, km
        except Exception:
            continue
    return best_k, best_model, silhouettes

def compute_cluster_artifacts(df: pd.DataFrame, numeric_cols, categorical_cols,
                              auto_k=True, k_manual=3, k_range=(2, 8), random_state=42):
    if not HAVE_SKLEARN:
        return {"cluster_available": False}

    if not numeric_cols and not categorical_cols:
        return {"cluster_available": False}

    X, feat_names, df_model = prepare_features_for_clustering(df, numeric_cols, categorical_cols)

    silhouettes = {}
    if auto_k:
        best_k, model, silhouettes = auto_kmeans(X, k_min=k_range[0], k_max=k_range[1], random_state=random_state)
        if not model:
            return {"cluster_available": False}
        k = best_k
        labels = model.predict(X)
        sil = silhouette_score(X, labels) if len(set(labels)) > 1 else None
    else:
        k = max(2, int(k_manual))
        model = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
        labels = model.fit_predict(X)
        sil = silhouette_score(X, labels) if len(set(labels)) > 1 else None

    pca = PCA(n_components=2, random_state=random_state)
    X2 = pca.fit_transform(X)

    prof_df = df_model.copy()
    prof_df["__cluster__"] = labels
    profile = (prof_df.groupby("__cluster__").mean(numeric_only=True)
               .reset_index()
               .rename(columns={"__cluster__": "cluster"}))

    sample_labeled = df.copy()
    sample_labeled["cluster"] = labels
    sample_rows = sample_labeled.sample(min(5, len(sample_labeled)), random_state=random_state).to_dict("records")

    return {
        "cluster_available": True,
        "cluster_k": int(k),
        "cluster_silhouette": float(sil) if sil is not None else None,
        "cluster_labels": labels.tolist(),
        "cluster_pca_2d": X2.tolist(),
        "cluster_profile": profile.to_dict("records"),
        "cluster_feat_names": feat_names,
        "cluster_sample_labeled": sample_rows,
        "cluster_silhouettes_grid": {int(kk): float(v) for kk, v in silhouettes.items()},
    }

# ---------- prompt builder ----------
def build_chat_context(art):
    """Contexto compacto com resultados de TODAS as abas (inclui Clusters)."""
    def fmt_rows(rows, keys, max_rows=10):
        rows = rows[:max_rows]
        out = []
        for r in rows:
            out.append(", ".join([f"{k}={r.get(k)}" for k in keys if k in r]))
        return "\n".join(out) if out else "—"

    # Estatísticas
    num_stats = art.get("numeric_describe", [])
    stat_keys = [k for k in ["coluna", "count", "mean", "std", "var", "min", "25%", "50%", "75%", "max"] if any(k in d for d in num_stats)]
    stats_txt = fmt_rows(num_stats, stat_keys, max_rows=15)

    # Correlações
    corr = art.get("significant_correlations", [])[:15]
    corr_txt = fmt_rows(corr, ["var1", "var2", "r", "forca"], max_rows=15)

    # Outliers
    outs = art.get("outliers_summary", [])[:15]
    outs_txt = fmt_rows(outs, ["variavel", "outliers_iqr", "pct_iqr", "outliers_z", "pct_z"], max_rows=15)

    # Categóricas
    cats = art.get("top_categories", {})
    cats_txt_parts = []
    for col, items in cats.items():
        snippet = ", ".join([f"{d['categoria']}({d['freq']})" for d in items[:8]])
        cats_txt_parts.append(f"{col}: {snippet}")
    cats_txt = "\n".join(cats_txt_parts) if cats_txt_parts else "—"

    # Nulos
    nulos = art.get("missing_values", [])
    nulos_txt = fmt_rows(nulos, ["coluna", "nulos"], max_rows=30)

    # Clusters
    if art.get("cluster_available", False):
        k = art.get("cluster_k")
        sil = art.get("cluster_silhouette")
        prof = art.get("cluster_profile", [])[:10]
        def fmt_profile_row(r: dict, top_n=10):
            items = list(r.items())
            items = sorted(items, key=lambda kv: (kv[0] != "cluster", kv[0]))
            shown, count = [], 0
            for k2, v2 in items:
                if k2 == "cluster":
                    shown.append(f"{k2}={v2}")
                else:
                    if count < top_n:
                        try:
                            shown.append(f"{k2}={round(float(v2), 3)}")
                        except Exception:
                            shown.append(f"{k2}={v2}")
                        count += 1
            return ", ".join(shown)
        prof_txt = "\n".join([fmt_profile_row(r) for r in prof])
        silhouettes_grid = art.get("cluster_silhouettes_grid", {})
        if silhouettes_grid:
            grid_txt = ", ".join([f"K={kk}: {round(v,3)}" for kk, v in sorted(silhouettes_grid.items())])
        else:
            grid_txt = "—"
        clusters_txt = f"K={k}, silhouette={round(sil,3) if sil is not None else '—'}\nSilhouette por K: {grid_txt}\nPerfis (médias/proporções):\n{prof_txt}"
    else:
        clusters_txt = "—"

    context = f"""
Você é um especialista em análise de dados. Responda usando APENAS o contexto abaixo, que resume as abas do app:

[Visão Geral]
- Formato: {art.get('shape')}
- Colunas: {', '.join(art.get('columns', [])[:60])}
- Variáveis numéricas: {', '.join(art.get('numeric_columns', [])[:60])}
- Variáveis categóricas: {', '.join(art.get('categorical_columns', [])[:60])}
- Valores nulos (por coluna):
{nulos_txt}

[Estatísticas]
{stats_txt}

[Correlações significativas (topo)]
{corr_txt}

[Outliers - resumo (topo por %IQR)]
{outs_txt}

[Categóricas - top frequências]
{cats_txt}

[Clusters]
{clusters_txt}

[Amostra (5 linhas)]
{art.get('sample_data')}
"""
    return context

# =========================
# UI
# =========================
st.title("🤖 Análise Exploratória de Dados")
st.markdown("**Análise completa de CSV com IA conversacional**")

# Upload
uploaded_file = st.file_uploader("Carregue seu arquivo CSV", type=['csv'])

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file, sep=None, engine="python")
        st.success(f"✅ Arquivo carregado: {data.shape[0]} linhas x {data.shape[1]} colunas")

        # ---- calcula EDA e PRESERVA clusters no rerun ----
        artifacts = compute_analysis_artifacts(data)
        if st.session_state.data_context is None:
            st.session_state.data_context = artifacts
        else:
            # preserva chaves de cluster (se já existirem) antes de atualizar EDA
            preserved_keys = [
                "cluster_available","cluster_k","cluster_silhouette","cluster_labels",
                "cluster_pca_2d","cluster_profile","cluster_feat_names",
                "cluster_sample_labeled","cluster_silhouettes_grid"
            ]
            preserved = {k: st.session_state.data_context.get(k) for k in preserved_keys}
            # atualiza com nova EDA
            st.session_state.data_context.update(artifacts)
            # restaura clusters se existirem
            for k, v in preserved.items():
                if v not in (None, [], {}, False):
                    st.session_state.data_context[k] = v

        numeric_cols = st.session_state.data_context["numeric_columns"]
        categorical_cols = st.session_state.data_context["categorical_columns"]

        # Abas
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "📋 Visão Geral", "📊 Distribuições", "🔍 Correlações",
            "📈 Tendências", "⚠️ Anomalias", "🤖 Chat IA", "🧩 Clusters"
        ])

        with tab1:
            st.header("📋 Visão Geral")
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Informações Básicas")
                st.write(f"**Linhas:** {data.shape[0]:,}")
                st.write(f"**Colunas:** {data.shape[1]:,}")
                st.write(f"**Numéricas:** {len(numeric_cols)}")
                st.write(f"**Categóricas:** {len(categorical_cols)}")

                tipos_dados = pd.DataFrame({
                    'Coluna': data.columns,
                    'Tipo': data.dtypes.astype(str),
                    'Únicos': [data[col].nunique() for col in data.columns],
                    'Nulos': [data[col].isnull().sum() for col in data.columns],
                    '% Nulos': [f"{(data[col].isnull().sum()/len(data)*100):.1f}%" for col in data.columns]
                })
                st.dataframe(tipos_dados, use_container_width=True)

            with col2:
                st.subheader("Primeiras Linhas")
                st.dataframe(data.head(10), use_container_width=True)

                if numeric_cols:
                    st.subheader("Estatísticas")
                    stats_df = data[numeric_cols].describe().T
                    stats_df["Variância"] = data[numeric_cols].var()  # ddof=1
                    st.dataframe(stats_df.T, use_container_width=True)

                # Recalcular EDA PRESERVANDO clusters
                if st.button("🔄 Atualizar análise (recalcular artefatos)"):
                    new_art = compute_analysis_artifacts(data)
                    preserved_keys = [
                        "cluster_available","cluster_k","cluster_silhouette","cluster_labels",
                        "cluster_pca_2d","cluster_profile","cluster_feat_names",
                        "cluster_sample_labeled","cluster_silhouettes_grid"
                    ]
                    for k in preserved_keys:
                        new_art[k] = st.session_state.data_context.get(k)
                    st.session_state.data_context = new_art
                    st.success("Artefatos recalculados (clusters preservados).")

        with tab2:
            st.header("📊 Distribuições")

            # Numéricas
            if numeric_cols:
                st.subheader("Variáveis Numéricas")
                n_cols = len(numeric_cols)
                cols_per_row = 3
                rows = (n_cols + cols_per_row - 1) // cols_per_row

                fig, axes = plt.subplots(rows, cols_per_row, figsize=(15, 5*rows))
                fig.patch.set_facecolor('#0E1117')
                axes = np.atleast_1d(axes).ravel()

                for i, col in enumerate(numeric_cols):
                    ax = axes[i]
                    data[col].hist(ax=ax, bins=30, alpha=0.7, color='cyan', edgecolor='white')
                    ax.set_title(f'{col}', color='white', fontsize=10)
                    ax.set_facecolor('#0E1117')
                    ax.tick_params(colors='white', labelsize=8)
                    ax.grid(True, alpha=0.3)

                for i in range(n_cols, len(axes)):
                    axes[i].set_visible(False)

                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            # Categóricas
            if categorical_cols:
                st.subheader("Variáveis Categóricas")
                for col in categorical_cols[:5]:
                    value_counts = data[col].value_counts().head(10)

                    fig, ax = plt.subplots(figsize=(10, 6))
                    fig.patch.set_facecolor('#0E1117')

                    bars = ax.bar(range(len(value_counts)), value_counts.values, color='lightcoral', alpha=0.8)
                    ax.set_title(f'{col}', color='white', fontsize=14)
                    ax.set_xticks(range(len(value_counts)))
                    ax.set_xticklabels(value_counts.index, rotation=45, ha='right', color='white')
                    ax.set_facecolor('#0E1117')
                    ax.tick_params(colors='white')
                    ax.grid(True, alpha=0.3, axis='y')

                    for bar, value in zip(bars, value_counts.values):
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(value_counts.values)*0.01,
                                f'{value:,}', ha='center', va='bottom', color='white', fontsize=9)

                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

        with tab3:
            st.header("🔍 Correlações")

            if len(numeric_cols) > 1:
                correlation_matrix = data[numeric_cols].corr()

                fig, ax = plt.subplots(figsize=(12, 10))
                fig.patch.set_facecolor('#0E1117')

                sns.heatmap(correlation_matrix, annot=True, cmap='RdYlBu_r', center=0,
                            square=True, fmt='.2f', cbar_kws={'shrink': 0.8}, ax=ax)

                ax.set_title('Matriz de Correlação', color='white', fontsize=16)
                ax.set_facecolor('#0E1117')
                plt.xticks(rotation=45, ha='right', color='white')
                plt.yticks(rotation=0, color='white')

                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                # Pares significativos
                st.subheader("Correlações Significativas")
                correlations = []
                for i in range(len(correlation_matrix.columns)):
                    for j in range(i+1, len(correlation_matrix.columns)):
                        corr_value = correlation_matrix.iloc[i, j]
                        if abs(corr_value) > 0.1:
                            correlations.append({
                                'Variável 1': correlation_matrix.columns[i],
                                'Variável 2': correlation_matrix.columns[j],
                                'Correlação': f"{corr_value:.3f}",
                                'Força': 'Forte' if abs(corr_value) > 0.7 else 'Moderada' if abs(corr_value) > 0.3 else 'Fraca'
                            })
                if correlations:
                    st.dataframe(pd.DataFrame(correlations), use_container_width=True)
                else:
                    st.info("Não há correlações significativas.")
            else:
                st.info("Necessário pelo menos 2 variáveis numéricas.")

        with tab4:
            st.header("📈 Tendências")

            time_cols = [col for col in data.columns if any(k in col.lower() for k in ['time', 'date', 'timestamp', 'year', 'month'])]

            if time_cols and numeric_cols:
                st.subheader("Tendências Temporais")
                c1, c2 = st.columns(2)
                with c1:
                    time_col = st.selectbox("Coluna temporal:", time_cols, key="time_col_sel")
                with c2:
                    numeric_col = st.selectbox("Variável numérica:", numeric_cols, key="num_col_sel")

                if time_col and numeric_col:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    fig.patch.set_facecolor('#0E1117')

                    data_sorted = data.sort_values(time_col)
                    x_values = list(range(len(data_sorted)))
                    y_values = data_sorted[numeric_col]

                    ax.scatter(x_values, y_values, alpha=0.6, color='cyan', s=20, label='Dados')

                    try:
                        mask = ~pd.isna(y_values)
                        if mask.sum() > 1:
                            x_clean = np.array(x_values)[mask]
                            y_clean = np.array(y_values)[mask]
                            z = np.polyfit(x_clean, y_clean, 1)
                            p = np.poly1d(z)
                            ax.plot(x_clean, p(x_clean), "r--", linewidth=2, label=f'Tendência (slope: {z[0]:.4f})')
                    except Exception:
                        pass

                    ax.set_title(f'Tendência: {numeric_col} vs {time_col}', color='white', fontsize=14)
                    ax.set_xlabel('Índice Temporal', color='white')
                    ax.set_ylabel(numeric_col, color='white')
                    ax.set_facecolor('#0E1117')
                    ax.tick_params(colors='white')
                    ax.grid(True, alpha=0.3)
                    ax.legend()

                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

            if len(numeric_cols) >= 2:
                st.subheader("Correlação Visual")
                corr_matrix = data[numeric_cols].corr()
                corr_abs = corr_matrix.abs()
                np.fill_diagonal(corr_abs.values, 0)

                if corr_abs.max().max() > 0:
                    max_idx = np.unravel_index(corr_abs.values.argmax(), corr_abs.shape)
                    var1, var2 = corr_matrix.columns[max_idx[0]], corr_matrix.columns[max_idx[1]]
                    corr_value = corr_matrix.loc[var1, var2]

                    fig, ax = plt.subplots(figsize=(10, 6))
                    fig.patch.set_facecolor('#0E1117')

                    ax.scatter(data[var1], data[var2], alpha=0.6, color='lightgreen', s=30)

                    try:
                        mask = ~(pd.isna(data[var1]) | pd.isna(data[var2]))
                        if mask.sum() > 1:
                            z = np.polyfit(data[var1][mask], data[var2][mask], 1)
                            p = np.poly1d(z)
                            x_range = np.linspace(data[var1].min(), data[var1].max(), 100)
                            ax.plot(x_range, p(x_range), "r--", linewidth=2)
                    except Exception:
                        pass

                    ax.set_title(f'{var1} vs {var2} (r = {corr_value:.3f})', color='white', fontsize=14)
                    ax.set_xlabel(var1, color='white')
                    ax.set_ylabel(var2, color='white')
                    ax.set_facecolor('#0E1117')
                    ax.tick_params(colors='white')
                    ax.grid(True, alpha=0.3)

                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

        with tab5:
            st.header("⚠️ Anomalias")

            if numeric_cols:
                st.subheader("Resumo de Outliers")
                outliers_summary = st.session_state.data_context.get("outliers_summary", [])
                if outliers_summary:
                    st.dataframe(pd.DataFrame(outliers_summary), use_container_width=True)
                else:
                    st.info("Sem resumo de outliers calculado.")

                st.subheader("Boxplots")
                n_cols = len(numeric_cols)
                cols_per_row = 4
                rows = (n_cols + cols_per_row - 1) // cols_per_row

                fig, axes = plt.subplots(rows, cols_per_row, figsize=(16, 4*rows))
                fig.patch.set_facecolor('#0E1117')
                axes = np.atleast_1d(axes).ravel()

                for i, col in enumerate(numeric_cols):
                    ax = axes[i]
                    ax.boxplot(
                        data[col].dropna(), patch_artist=True,
                        boxprops=dict(facecolor='lightblue', alpha=0.7),
                        medianprops=dict(color='red', linewidth=2)
                    )
                    ax.set_title(f'{col}', color='white', fontsize=10)
                    ax.set_facecolor('#0E1117')
                    ax.tick_params(colors='white')
                    ax.grid(True, alpha=0.3)

                for i in range(n_cols, len(axes)):
                    axes[i].set_visible(False)

                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("Não há variáveis numéricas.")

        with tab6:
            st.header("🤖 Chat com IA")

            c1, c2, c3 = st.columns(3)
            with c1:
                api_choice = st.selectbox("API:", ["OpenAI", "Groq", "Gemini"])
            with c2:
                api_key = st.text_input(f"Chave {api_choice}:", type="password")
            with c3:
                if api_choice == "OpenAI":
                    models = ["gpt-3.5-turbo", "gpt-4"]
                elif api_choice == "Groq":
                    models = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
                else:
                    models = ["gemini-pro"]
                model = st.selectbox("Modelo:", models)

            if st.session_state.conversation_history:
                st.subheader("Conversa")
                for msg in st.session_state.conversation_history[-6:]:
                    if msg["role"] == "user":
                        st.markdown(f"**👤:** {msg['content']}")
                    else:
                        st.markdown(f"**🤖:** {msg['content']}")
                st.markdown("---")

            user_question = st.text_area(
                "Sua pergunta:",
                placeholder="Ex: Existem clusters? O que diferencia cada grupo? Quais outliers tratar primeiro?",
                height=100
            )

            c1, c2 = st.columns([1, 1])
            with c1:
                if st.button("🚀 Enviar", type="primary"):
                    if user_question.strip() and api_key:
                        st.session_state.conversation_history.append({"role": "user", "content": user_question})
                        with st.spinner("Processando..."):
                            context = build_chat_context(st.session_state.data_context)
                            messages = [{"role": "system", "content": context}]
                            messages.extend(st.session_state.conversation_history[-4:])
                            response = call_ai_api(api_choice, api_key, messages, model)
                            st.session_state.conversation_history.append({"role": "assistant", "content": response})
                            st.success("✅ Resposta:")
                            st.markdown(response)
                    else:
                        st.error("❌ Insira pergunta e chave da API.")

            with c2:
                if st.button("🗑️ Limpar"):
                    st.session_state.conversation_history = []
                    st.rerun()

            if not api_key:
                st.info(f"🔑 Como obter chave {api_choice}:")
                if api_choice == "OpenAI":
                    st.markdown("1. https://platform.openai.com → API Keys")
                elif api_choice == "Groq":
                    st.markdown("1. https://console.groq.com → API Keys")
                else:
                    st.markdown("1. https://makersuite.google.com → Get API Key")

        with tab7:
            st.header("🧩 Análise de Agrupamento (KMeans)")
            if not HAVE_SKLEARN:
                st.warning("Aba desativada: instale scikit-learn para habilitar clustering. Ex.: `pip install -U scikit-learn`")
            elif not (numeric_cols or categorical_cols):
                st.info("O dataset não possui colunas suficientes para clusterização.")
            else:
                toggle_fn = getattr(st, "toggle", None)
                c1, c2, c3 = st.columns([1,1,2])
                with c1:
                    auto_k = toggle_fn("K automático (silhouette)", value=True) if toggle_fn else st.checkbox("K automático (silhouette)", value=True)
                with c2:
                    k_manual = st.number_input("K (se manual)", min_value=2, max_value=15, value=3, step=1, disabled=auto_k)
                with c3:
                    kmin, kmax = st.slider("Faixa para busca de K", min_value=2, max_value=15, value=(2, 8), disabled=not auto_k)

                if st.button("▶️ Rodar clustering"):
                    with st.spinner("Clusterizando..."):
                        cl_art = compute_cluster_artifacts(
                            data, numeric_cols, categorical_cols,
                            auto_k=auto_k, k_manual=k_manual, k_range=(kmin, kmax), random_state=42
                        )
                        if not cl_art.get("cluster_available", False):
                            st.error("Não foi possível formar clusters estáveis (ou scikit-learn ausente).")
                        else:
                            # >>> registra clusters na 'memória' e mantém nos reruns <<<
                            st.session_state.data_context.update(cl_art)

                            if cl_art['cluster_silhouette'] is not None:
                                st.success(f"Clusters formados: K={cl_art['cluster_k']} (silhouette={cl_art['cluster_silhouette']:.3f})")
                            else:
                                st.success(f"Clusters formados: K={cl_art['cluster_k']} (silhouette=—)")

                            if cl_art.get("cluster_silhouettes_grid"):
                                st.subheader("Silhouette por K testado")
                                sil_df = pd.DataFrame(
                                    [{"K": k, "silhouette": v} for k, v in cl_art["cluster_silhouettes_grid"].items()]
                                ).sort_values("K")
                                st.dataframe(sil_df, use_container_width=True)

                            st.subheader("Visualização PCA 2D")
                            X2 = np.array(cl_art["cluster_pca_2d"])
                            labels = np.array(cl_art["cluster_labels"])
                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.scatter(X2[:, 0], X2[:, 1], c=labels, s=20, alpha=0.85)
                            ax.set_title("Clusters (PCA 2D)")
                            ax.set_xlabel("PC1")
                            ax.set_ylabel("PC2")
                            ax.grid(True, alpha=0.3)
                            st.pyplot(fig)
                            plt.close(fig)

                            st.subheader("Perfis dos Clusters (médias / proporções)")
                            prof_df = pd.DataFrame(cl_art["cluster_profile"])
                            st.dataframe(prof_df, use_container_width=True)

                            st.subheader("Amostra de linhas com rótulo do cluster")
                            st.dataframe(pd.DataFrame(cl_art["cluster_sample_labeled"]), use_container_width=True)

    except Exception as e:
        st.error(f"❌ Erro: {str(e)}")

else:
    st.markdown("""
    ## 🎯 Análise Exploratória de Dados com IA

    **Funcionalidades:**
    - 📊 **Distribuições**: Histogramas e gráficos de barras
    - 🔍 **Correlações**: Matriz e pares significativos
    - 📈 **Tendências**: Dispersão temporal + linha de tendência
    - ⚠️ **Anomalias**: Outliers (IQR/z-score) e boxplots
    - 🧩 **Clusters**: KMeans com seleção automática de K e PCA 2D
    - 🤖 **Chat IA**: Conversa usando o contexto gerado (inclui clusters)

    **Dica**: Se alguma aba não aparecer, verifique as dependências indicadas no topo do código.
    """)
