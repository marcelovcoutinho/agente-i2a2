import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import duckdb
import json
import re
from io import StringIO
from groq import Groq

# ==========================
# Configuração da página
# ==========================
st.set_page_config(
    page_title="Agente de Análise de Dados CSV",
    page_icon="📊",
    layout="wide"
)

# ==========================
# Utilidades
# ==========================

def sanitize_column(name: str) -> str:
    """Normaliza nomes de colunas para uso seguro em SQL/plots, mantendo um mapa para o original."""
    safe = re.sub(r"[^0-9a-zA-Z_]+", "_", name).strip("_")
    if re.match(r"^[0-9]", safe):
        safe = f"c_{safe}"
    return safe or "col"

@st.cache_data(show_spinner=False)
def compute_schema(df: pd.DataFrame):
    schema = []
    for c in df.columns:
        sample_vals = df[c].dropna().unique()[:5]
        schema.append({
            "name": c,
            "dtype": str(df[c].dtype),
            "examples": [str(v) for v in sample_vals]
        })
    return schema

@st.cache_data(show_spinner=False)
def get_numeric_and_categorical(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    return numeric_cols, categorical_cols

@st.cache_data(show_spinner=False)
def compute_outlier_summary(df: pd.DataFrame, numeric_cols):
    rows = []
    for col in numeric_cols:
        s = df[col].dropna()
        if s.empty:
            rows.append({
                "Variável": col,
                "Total de Outliers": 0,
                "Percentual": "0.00%",
                "Limite Inferior": None,
                "Limite Superior": None,
                "Valor Mínimo": None,
                "Valor Máximo": None,
            })
            continue
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (s < lower) | (s > upper)
        rows.append({
            "Variável": col,
            "Total de Outliers": int(mask.sum()),
            "Percentual": f"{(mask.mean() * 100):.2f}%",
            "Limite Inferior": float(lower) if np.isfinite(lower) else None,
            "Limite Superior": float(upper) if np.isfinite(upper) else None,
            "Valor Mínimo": float(s.min()) if not s.empty else None,
            "Valor Máximo": float(s.max()) if not s.empty else None,
        })
    return pd.DataFrame(rows)

# ==========================
# Memória conversacional
# ==========================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # [{"role": "user|assistant", "content": str, "sql": str|None}]

if "colmap" not in st.session_state:
    st.session_state.colmap = {}

# ==========================
# Título
# ==========================
st.title("🤖 Agente de Análise Exploratória de Dados (E.D.A.)")
st.markdown("**Ferramenta inteligente para análise de qualquer arquivo CSV com IA Groq**")

# ==========================
# Upload
# ==========================
uploaded_file = st.file_uploader(
    "Carregue seu arquivo CSV para análise",
    type=["csv"],
    help="Selecione um arquivo CSV para realizar a análise exploratória completa"
)

if uploaded_file is None:
    st.info("Faça o upload de um CSV para começar.")
    st.stop()

# ==========================
# Carregar dados
# ==========================
try:
    data = pd.read_csv(uploaded_file)
except Exception:
    uploaded_file.seek(0)
    data = pd.read_csv(uploaded_file, sep=";")

st.success(f"✅ Arquivo carregado: {data.shape[0]:,} linhas × {data.shape[1]:,} colunas.")

numeric_cols, categorical_cols = get_numeric_and_categorical(data)
schema = compute_schema(data)

# Mapa de nomes seguros ↔ originais (para SQL)
safe_to_orig = {}
orig_to_safe = {}
for c in data.columns:
    safe = sanitize_column(c)
    # Evita colisões
    i = 1
    base = safe
    while safe in safe_to_orig and safe_to_orig[safe] != c:
        safe = f"{base}_{i}"
        i += 1
    safe_to_orig[safe] = c
    orig_to_safe[c] = safe

# View com nomes seguros
safe_df = data.copy()
safe_df.columns = [orig_to_safe[c] for c in data.columns]

# ==========================
# Abas
# ==========================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📋 Visão Geral",
    "📊 Distribuições",
    "🔍 Correlações",
    "📈 Tendências",
    "⚠️ Anomalias",
    "🤖 Consulta IA"
])

# ==========================
# 📋 Visão Geral
# ==========================
with tab1:
    st.header("📋 Visão Geral dos Dados")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Informações Básicas")
        st.write(f"**Número de linhas:** {data.shape[0]:,}")
        st.write(f"**Número de colunas:** {data.shape[1]:,}")
        try:
            mem_mb = data.memory_usage(deep=True).sum() / 1024 ** 2
            st.write(f"**Tamanho em memória:** {mem_mb:.2f} MB")
        except Exception:
            pass

        st.subheader("Tipos de Dados e Nulos")
        tipos = pd.DataFrame({
            "Coluna": data.columns,
            "Tipo": [str(t) for t in data.dtypes.values],
            "Valores Únicos": [data[c].nunique(dropna=True) for c in data.columns],
            "Valores Nulos": [int(data[c].isna().sum()) for c in data.columns],
            "% Nulos": [f"{(data[c].isna().mean()*100):.1f}%" for c in data.columns],
        })
        st.dataframe(tipos, use_container_width=True, hide_index=True)

    with col2:
        st.subheader("Primeiras linhas")
        st.dataframe(data.head(20), use_container_width=True)

        if numeric_cols:
            st.subheader("Estatísticas Descritivas (numéricas)")
            st.dataframe(data[numeric_cols].describe().T, use_container_width=True)

# ==========================
# 📊 Distribuições (agora de TODAS as variáveis)
# ==========================
with tab2:
    st.header("📊 Distribuição das Variáveis")

    # Seleção de colunas
    all_cols = numeric_cols + categorical_cols
    sel_cols = st.multiselect(
        "Escolha colunas para visualizar (vazio = todas)",
        options=all_cols,
        default=[]
    )
    if not sel_cols:
        sel_cols = all_cols

    page_size = st.slider("Gráficos por página", 6, 36, 12, step=6)
    total = len(sel_cols)
    page = st.number_input("Página", min_value=1, max_value=max(1, int(np.ceil(total / page_size))), value=1)
    start = (page - 1) * page_size
    end = min(total, start + page_size)

    to_plot = sel_cols[start:end]

    for col in to_plot:
        st.subheader(f"Distribuição: {col}")
        if col in numeric_cols:
            s = data[col].dropna()
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(s, bins=30, alpha=0.8)
            ax.set_xlabel(col)
            ax.set_ylabel("Frequência")
            st.pyplot(fig)
            plt.close(fig)
        else:
            vc = data[col].astype(str).value_counts().head(50)
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(vc.index, vc.values)
            ax.set_xticklabels(vc.index, rotation=45, ha='right')
            ax.set_ylabel("Contagem")
            st.pyplot(fig)
            plt.close(fig)

# ==========================
# 🔍 Correlações
# ==========================
with tab3:
    st.header("🔍 Correlações entre Variáveis")
    if len(numeric_cols) > 1:
        corr = data[numeric_cols].corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr, cmap="RdYlBu_r", center=0, annot=False, square=True, ax=ax)
        ax.set_title("Matriz de Correlação")
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Top correlações (|r| > 0.3)")
        pairs = []
        cols = corr.columns.tolist()
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                r = corr.iloc[i, j]
                if abs(r) > 0.3:
                    pairs.append({"Variável 1": cols[i], "Variável 2": cols[j], "Correlação": float(r)})
        if pairs:
            st.dataframe(pd.DataFrame(pairs).sort_values("Correlação", key=np.abs, ascending=False), use_container_width=True)
        else:
            st.info("Nenhuma correlação relevante encontrada com o limiar definido.")
    else:
        st.info("É necessário pelo menos 2 colunas numéricas.")

# ==========================
# 📈 Tendências (com linha de tendência)
# ==========================
with tab4:
    st.header("📈 Análise de Tendências")

    # Identificar possíveis colunas temporais
    time_candidates = [c for c in data.columns if any(k in c.lower() for k in ["time", "date", "data", "timestamp"]) ]
    time_col = None
    if time_candidates:
        time_col = st.selectbox("Selecione a coluna temporal", options=time_candidates)
    else:
        st.info("Nenhuma coluna temporal aparente encontrada. Você pode ainda assim visualizar por índice.")

    target_cols = st.multiselect("Selecione variáveis numéricas para série temporal", options=numeric_cols, default=numeric_cols[:1])
    trend_option = st.selectbox("Linha de tendência", ["Nenhuma", "Linear", "Média móvel"], index=1 if target_cols else 0)
    window = 10
    if trend_option == "Média móvel":
        window = st.slider("Janela da média móvel", 3, 100, 10)

    if target_cols:
        for tgt in target_cols:
            st.subheader(f"{tgt}")
            if time_col:
                df = data[[time_col, tgt]].dropna().copy()
                # Tenta converter a tempo
                try:
                    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
                except Exception:
                    pass
                df = df.dropna(subset=[time_col]).sort_values(time_col)
                x = df[time_col]
                y = df[tgt]
            else:
                df = data[[tgt]].dropna().reset_index().rename(columns={"index": "idx"})
                x = df["idx"]
                y = df[tgt]

            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(x, y, label=tgt, alpha=0.8)

            # Tendência
            if trend_option == "Linear" and len(df) >= 2:
                # Converte x para numérico (ordinal para datas)
                if time_col and pd.api.types.is_datetime64_any_dtype(df[time_col]):
                    xn = df[time_col].map(pd.Timestamp.toordinal).to_numpy()
                else:
                    xn = np.arange(len(df))
                yn = y.to_numpy()
                try:
                    m, b = np.polyfit(xn, yn, 1)
                    yhat = m * xn + b
                    # R2
                    r = np.corrcoef(xn, yn)[0, 1]
                    r2 = float(r ** 2) if not np.isnan(r) else 0.0
                    if time_col and pd.api.types.is_datetime64_any_dtype(df[time_col]):
                        ax.plot(x, yhat, linestyle="--", label=f"Tendência linear (R²={r2:.2f})")
                    else:
                        ax.plot(xn, yhat, linestyle="--", label=f"Tendência linear (R²={r2:.2f})")
                except Exception:
                    pass
            elif trend_option == "Média móvel":
                mm = y.rolling(window=window, min_periods=max(2, window//2)).mean()
                ax.plot(x, mm, linestyle="--", label=f"Média móvel ({window})")

            ax.legend()
            ax.set_xlabel(time_col or "Índice")
            ax.set_ylabel(tgt)
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)

    # Padrões em variáveis categóricas
    if categorical_cols:
        st.subheader("Padrões em Variáveis Categóricas")
        cat_col = st.selectbox("Selecione uma variável categórica", options=categorical_cols)
        if cat_col:
            vc = data[cat_col].value_counts(dropna=False)
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Mais frequentes**")
                st.dataframe(vc.head(10).reset_index().rename(columns={"index": cat_col, cat_col: "contagem"}), use_container_width=True)
            with col2:
                st.write("**Menos frequentes**")
                st.dataframe(vc.tail(10).reset_index().rename(columns={"index": cat_col, cat_col: "contagem"}), use_container_width=True)

# ==========================
# ⚠️ Anomalias (todos os gráficos)
# ==========================
with tab5:
    st.header("⚠️ Detecção de Anomalias")
    if numeric_cols:
        out_df = compute_outlier_summary(data, numeric_cols)
        st.dataframe(out_df, use_container_width=True, hide_index=True)

        only_with_outliers = st.checkbox("Mostrar apenas variáveis com outliers", value=True)
        vars_to_plot = out_df[out_df["Total de Outliers"] > 0]["Variável"].tolist() if only_with_outliers else numeric_cols

        st.subheader("Boxplots")
        for col in vars_to_plot:
            s = data[col].dropna()
            if s.empty:
                continue
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.boxplot(s, vert=True, patch_artist=True)
            ax.set_title(col)
            st.pyplot(fig)
            plt.close(fig)

        # Download dos índices de outliers
        with st.expander("Exportar outliers"):
            rows = []
            for col in numeric_cols:
                s = data[col].dropna()
                if s.empty:
                    continue
                q1 = s.quantile(0.25)
                q3 = s.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                mask = (data[col] < lower) | (data[col] > upper)
                if mask.any():
                    for idx, val in data.loc[mask, col].items():
                        rows.append({"coluna": col, "index": idx, "valor": val})
            if rows:
                out_csv = pd.DataFrame(rows).to_csv(index=False).encode("utf-8")
                st.download_button("Baixar CSV de outliers", data=out_csv, file_name="outliers.csv", mime="text/csv")
    else:
        st.info("Não há variáveis numéricas para análise de outliers.")

# ==========================
# 🤖 Consulta IA (com memória + SQL/execução)
# ==========================
with tab6:
    st.header("🤖 Consulta Inteligente com IA (com acesso aos dados)")
    st.markdown("Faça perguntas sobre seus dados. O agente **gera SQL para o DuckDB** e executa localmente no seu DataFrame, devolvendo tabelas e gráficos. O histórico da conversa é preservado.")

    api_key = st.text_input("🔑 Chave da API Groq", type="password", help="Sua chave não é armazenada.")

    # Exibe o esquema para o usuário
    with st.expander("Esquema das colunas"):
        schema_df = pd.DataFrame(schema)
        st.dataframe(schema_df, use_container_width=True, hide_index=True)

    user_q = st.text_area("💭 Pergunte algo (ex.: média de Amount por Class; top 10 registros com maior Amount)", height=100)
    auto_plot = st.checkbox("Gerar gráfico automaticamente quando fizer sentido", value=True)

    def llm_plan_to_sql(question: str, schema: list, history_messages: list, api_key: str) -> dict:
        """Pede ao LLM para gerar um plano JSON com SQL DuckDB + sugestão de gráfico.
        Retorna dict com chaves: sql (str), plot (dict|None), summary (str)
        """
        client = Groq(api_key=api_key)
        schema_str = "\n".join([f'- "{c["name"]}" ({c["dtype"]}) exemplos: {", ".join(c["examples"])}' for c in schema])

        # Limita histórico a 5 últimas trocas
        hist_text = "\n".join([f"Q: {m['content']}\nA: {m.get('summary','')}\nSQL: {m.get('sql','')}" for m in history_messages[-5:] if m['role']=="assistant" or m['role']=="user"]) or "(sem histórico)"

        system = {
            "role": "system",
            "content": (
                "Você é um analista de dados especializado em gerar consultas SQL para DuckDB, com a tabela 'data'. "
                "Responda **exclusivamente** em JSON com o formato: {\"sql\": str, \"plot\": {\"kind\": oneof['bar','line','scatter','hist'], \"x\": str|null, \"y\": str|list|null, \"agg\": str|null}|null, \"summary\": str}. "
                "Regras: 1) Use nomes de coluna **exatamente** como aparecem abaixo, mas troque por nomes "
                "sanitizados se indicado entre colchetes; 2) Sempre gere um SELECT; 3) Se o resultado puder ser muito grande, inclua LIMIT 200; "
                "4) Quando apropriado, proponha um gráfico coerente com as colunas; 5) Não inclua texto fora do JSON."
            )
        }

        # Mapeamento de nomes originais -> seguros (para o SQL)
        mapping = "\n".join([f'"{orig}" -> {safe}' for safe, orig in safe_to_orig.items()])

        user = {
            "role": "user",
            "content": (
                f"Esquema das colunas (use nomes originais entre aspas, mas no SQL utilize o nome sanitizado em colchetes quando existir):\n"
                f"{schema_str}\n\nMapeamento nomes -> sanitizados:\n{mapping}\n\n"
                f"Histórico:\n{hist_text}\n\nPergunta: {question}"
            )
        }

        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[system, user],
            max_tokens=800,
            temperature=0.2
        )
        content = resp.choices[0].message.content.strip()
        # Tenta extrair JSON
        try:
            j = json.loads(content)
        except json.JSONDecodeError:
            # Heurística: pega o primeiro bloco { ... }
            m = re.search(r"\{[\s\S]*\}", content)
            if not m:
                raise ValueError("LLM não retornou JSON válido.")
            j = json.loads(m.group(0))
        return j

    def run_duckdb_sql(sql: str, df_safe: pd.DataFrame) -> pd.DataFrame:
        if not sql.strip().lower().startswith("select"):
            raise ValueError("Apenas consultas SELECT são permitidas.")
        if ";" in sql.strip().rstrip(";"):
            raise ValueError("Apenas uma instrução SQL é permitida.")
        con = duckdb.connect()
        con.register("data", df_safe)
        try:
            res = con.execute(sql).df()
        finally:
            con.close()
        return res

    if st.button("🚀 Perguntar à IA", type="primary"):
        if not api_key:
            st.warning("Informe a chave da API Groq.")
        elif not user_q.strip():
            st.warning("Digite uma pergunta.")
        else:
            with st.spinner("Analisando pergunta e consultando dados..."):
                try:
                    plan = llm_plan_to_sql(user_q, schema, st.session_state.chat_history, api_key)
                    sql = plan.get("sql", "").replace("`", "\"")  # normaliza aspas
                    plot = plan.get("plot")
                    summary = plan.get("summary", "")

                    # Ajusta nomes sanitizados no SQL conforme mapeamento (espera-se uso dos nomes seguros)
                    # Apenas executa
                    df_res = run_duckdb_sql(sql, safe_df)

                    st.success("✅ Consulta executada com sucesso!")
                    st.code(sql, language="sql")
                    st.dataframe(df_res, use_container_width=True)

                    # Plot opcional
                    if auto_plot and plot:
                        kind = plot.get("kind")
                        x = plot.get("x")
                        y = plot.get("y")
                        agg = plot.get("agg")

                        try:
                            fig, ax = plt.subplots(figsize=(8, 4))
                            dplot = df_res.copy()
                            # Se y é lista, tenta melt/plot adequado
                            if kind == "bar":
                                if y is None and x and x in dplot.columns:
                                    dplot.set_index(x).plot(kind="bar", ax=ax)
                                elif x and y:
                                    if isinstance(y, list):
                                        dplot.plot(kind="bar", x=x, y=y, ax=ax)
                                    else:
                                        dplot.plot(kind="bar", x=x, y=y, ax=ax)
                                else:
                                    dplot.plot(kind="bar", ax=ax)
                            elif kind == "line":
                                if x and y:
                                    dplot.plot(kind="line", x=x, y=y, ax=ax)
                                else:
                                    dplot.plot(kind="line", ax=ax)
                            elif kind == "scatter" and x and y and not isinstance(y, list):
                                ax.scatter(dplot[x], dplot[y])
                                ax.set_xlabel(x); ax.set_ylabel(y)
                            elif kind == "hist" and y:
                                arr = dplot[y] if isinstance(y, list) else [dplot[y]]
                                for series in arr:
                                    ax.hist(series.dropna(), bins=30, alpha=0.6)
                            st.pyplot(fig)
                            plt.close(fig)
                        except Exception:
                            st.info("Gráfico sugerido não pôde ser gerado automaticamente.")

                    if summary:
                        st.markdown("### 🧠 Resumo do agente")
                        st.write(summary)

                    # Atualiza memória
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": user_q
                    })
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": str(df_res.head(3).to_dict()),
                        "sql": sql,
                        "summary": summary
                    })

                except Exception as e:
                    st.error(f"❌ Erro: {e}")
                    st.stop()

    # Histórico simples
    with st.expander("Histórico da conversa"):
        if st.session_state.chat_history:
            for i, m in enumerate(st.session_state.chat_history[-10:]):
                role = "👤" if m["role"] == "user" else "🤖"
                st.markdown(f"**{role}** {m['content'][:800]}")
        else:
            st.write("Sem mensagens ainda.")
