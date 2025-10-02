import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from groq import Groq
import openai
import google.generativeai as genai
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configuração da página
st.set_page_config(page_title="Análise de Dados CSV", page_icon="📊", layout="wide")
plt.style.use('dark_background')

# Inicializar sessão
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'data_context' not in st.session_state:
    st.session_state.data_context = None

def call_ai_api(api_choice, api_key, messages, model):
    try:
        if api_choice == "OpenAI":
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(model=model, messages=messages, max_tokens=2000, temperature=0.7)
            return response.choices[0].message.content
        elif api_choice == "Groq":
            client = Groq(api_key=api_key)
            response = client.chat.completions.create(model=model, messages=messages, max_tokens=2000, temperature=0.7)
            return response.choices[0].message.content
        elif api_choice == "Gemini":
            genai.configure(api_key=api_key)
            model_instance = genai.GenerativeModel(model)
            prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
            response = model_instance.generate_content(prompt)
            return response.text
    except Exception as e:
        return f"Erro: {str(e)}"

# Título
st.title("🤖 Análise Exploratória de Dados")
st.markdown("**Análise completa de CSV com IA conversacional**")

# Upload
uploaded_file = st.file_uploader("Carregue seu arquivo CSV", type=['csv'])

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.success(f"✅ Arquivo carregado: {data.shape[0]} linhas x {data.shape[1]} colunas")
        
        # Preparar contexto para IA
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        st.session_state.data_context = {
            'shape': data.shape,
            'columns': data.columns.tolist(),
            'numeric_columns': numeric_cols,
            'categorical_columns': categorical_cols,
            'basic_stats': data.describe().to_dict() if numeric_cols else {},
            'missing_values': data.isnull().sum().to_dict(),
            'sample_data': data.head(5).to_dict('records')
        }
        
        # Abas
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📋 Visão Geral", "📊 Distribuições", "🔍 Correlações", "📈 Tendências", "⚠️ Anomalias", "🤖 Chat IA"])
        
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
                    st.dataframe(data[numeric_cols].describe(), use_container_width=True)
        
        with tab2:
            st.header("📊 Distribuições")
            
            # Variáveis numéricas
            if numeric_cols:
                st.subheader("Variáveis Numéricas")
                n_cols = len(numeric_cols)
                cols_per_row = 3
                rows = (n_cols + cols_per_row - 1) // cols_per_row
                
                fig, axes = plt.subplots(rows, cols_per_row, figsize=(15, 5*rows))
                fig.patch.set_facecolor('#0E1117')
                
                if rows == 1 and cols_per_row == 1:
                    axes = [axes]
                elif rows == 1:
                    axes = [axes[i] for i in range(cols_per_row)]
                elif cols_per_row == 1:
                    axes = [axes[i] for i in range(rows)]
                else:
                    axes = axes.flatten()
                
                for i, col in enumerate(numeric_cols):
                    ax = axes[i]
                    data[col].hist(ax=ax, bins=30, alpha=0.7, color='cyan', edgecolor='white')
                    ax.set_title(f'{col}', color='white', fontsize=10)
                    ax.set_facecolor('#0E1117')
                    ax.tick_params(colors='white', labelsize=8)
                    ax.grid(True, alpha=0.3)
                
                # Remover subplots vazios
                for i in range(n_cols, len(axes)):
                    fig.delaxes(axes[i])
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            
            # Variáveis categóricas
            if categorical_cols:
                st.subheader("Variáveis Categóricas")
                for col in categorical_cols[:5]:  # Máximo 5
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
                    plt.close()
        
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
                plt.close()
                
                # Correlações significativas
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
                    corr_df = pd.DataFrame(correlations)
                    st.dataframe(corr_df, use_container_width=True)
                else:
                    st.info("Não há correlações significativas.")
            else:
                st.info("Necessário pelo menos 2 variáveis numéricas.")
        
        with tab4:
            st.header("📈 Tendências")
            
            # Detectar colunas temporais
            time_cols = [col for col in data.columns if any(keyword in col.lower() for keyword in ['time', 'date', 'timestamp', 'year', 'month'])]
            
            if time_cols and numeric_cols:
                st.subheader("Tendências Temporais")
                col1, col2 = st.columns(2)
                with col1:
                    time_col = st.selectbox("Coluna temporal:", time_cols)
                with col2:
                    numeric_col = st.selectbox("Variável numérica:", numeric_cols)
                
                if time_col and numeric_col:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    fig.patch.set_facecolor('#0E1117')
                    
                    data_sorted = data.sort_values(time_col)
                    x_values = range(len(data_sorted))
                    y_values = data_sorted[numeric_col]
                    
                    ax.scatter(x_values, y_values, alpha=0.6, color='cyan', s=20, label='Dados')
                    
                    # Linha de tendência
                    try:
                        mask = ~pd.isna(y_values)
                        if mask.sum() > 1:
                            x_clean = np.array(x_values)[mask]
                            y_clean = np.array(y_values)[mask]
                            z = np.polyfit(x_clean, y_clean, 1)
                            p = np.poly1d(z)
                            ax.plot(x_clean, p(x_clean), "r--", linewidth=2, label=f'Tendência (slope: {z[0]:.4f})')
                    except:
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
                    plt.close()
            
            # Correlação visual
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
                    
                    # Linha de regressão
                    try:
                        mask = ~(pd.isna(data[var1]) | pd.isna(data[var2]))
                        if mask.sum() > 1:
                            z = np.polyfit(data[var1][mask], data[var2][mask], 1)
                            p = np.poly1d(z)
                            x_range = np.linspace(data[var1].min(), data[var1].max(), 100)
                            ax.plot(x_range, p(x_range), "r--", linewidth=2)
                    except:
                        pass
                    
                    ax.set_title(f'{var1} vs {var2} (r = {corr_value:.3f})', color='white', fontsize=14)
                    ax.set_xlabel(var1, color='white')
                    ax.set_ylabel(var2, color='white')
                    ax.set_facecolor('#0E1117')
                    ax.tick_params(colors='white')
                    ax.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
        
        with tab5:
            st.header("⚠️ Anomalias")
            
            if numeric_cols:
                st.subheader("Resumo de Outliers")
                
                outliers_summary = []
                for col in numeric_cols:
                    Q1 = data[col].quantile(0.25)
                    Q3 = data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
                    
                    try:
                        z_scores = np.abs(stats.zscore(data[col].dropna()))
                        z_outliers = len(z_scores[z_scores > 3])
                    except:
                        z_outliers = 0
                    
                    outliers_summary.append({
                        'Variável': col,
                        'Outliers IQR': len(outliers),
                        '% IQR': f"{(len(outliers)/len(data)*100):.2f}%",
                        'Outliers Z-score': z_outliers,
                        '% Z-score': f"{(z_outliers/len(data)*100):.2f}%"
                    })
                
                st.dataframe(pd.DataFrame(outliers_summary), use_container_width=True)
                
                # Boxplots
                st.subheader("Boxplots")
                n_cols = len(numeric_cols)
                cols_per_row = 4
                rows = (n_cols + cols_per_row - 1) // cols_per_row
                
                fig, axes = plt.subplots(rows, cols_per_row, figsize=(16, 4*rows))
                fig.patch.set_facecolor('#0E1117')
                
                if rows == 1 and cols_per_row == 1:
                    axes = [axes]
                elif rows == 1:
                    axes = [axes[i] for i in range(cols_per_row)]
                elif cols_per_row == 1:
                    axes = [axes[i] for i in range(rows)]
                else:
                    axes = axes.flatten()
                
                for i, col in enumerate(numeric_cols):
                    ax = axes[i]
                    ax.boxplot(data[col].dropna(), patch_artist=True,
                              boxprops=dict(facecolor='lightblue', alpha=0.7),
                              medianprops=dict(color='red', linewidth=2))
                    ax.set_title(f'{col}', color='white', fontsize=10)
                    ax.set_facecolor('#0E1117')
                    ax.tick_params(colors='white')
                    ax.grid(True, alpha=0.3)
                
                for i in range(n_cols, len(axes)):
                    fig.delaxes(axes[i])
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            else:
                st.info("Não há variáveis numéricas.")
        
        with tab6:
            st.header("🤖 Chat com IA")
            
            # Configuração
            col1, col2, col3 = st.columns(3)
            
            with col1:
                api_choice = st.selectbox("API:", ["OpenAI", "Groq", "Gemini"])
            
            with col2:
                api_key = st.text_input(f"Chave {api_choice}:", type="password")
            
            with col3:
                if api_choice == "OpenAI":
                    models = ["gpt-3.5-turbo", "gpt-4"]
                elif api_choice == "Groq":
                    models = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
                else:
                    models = ["gemini-pro"]
                model = st.selectbox("Modelo:", models)
            
            # Histórico
            if st.session_state.conversation_history:
                st.subheader("Conversa")
                for msg in st.session_state.conversation_history[-6:]:
                    if msg["role"] == "user":
                        st.markdown(f"**👤:** {msg['content']}")
                    else:
                        st.markdown(f"**🤖:** {msg['content']}")
                st.markdown("---")
            
            # Input
            user_question = st.text_area("Sua pergunta:", placeholder="Ex: Quais são os principais insights deste dataset?", height=100)
            
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("🚀 Enviar", type="primary"):
                    if user_question.strip() and api_key:
                        st.session_state.conversation_history.append({"role": "user", "content": user_question})
                        
                        with st.spinner("Processando..."):
                            context = f"""
                            Você é um especialista em análise de dados. Dados disponíveis:
                            - Formato: {st.session_state.data_context['shape'][0]} linhas x {st.session_state.data_context['shape'][1]} colunas
                            - Colunas: {', '.join(st.session_state.data_context['columns'])}
                            - Numéricas: {', '.join(st.session_state.data_context['numeric_columns'])}
                            - Categóricas: {', '.join(st.session_state.data_context['categorical_columns'])}
                            - Estatísticas: {str(st.session_state.data_context['basic_stats'])}
                            - Valores nulos: {str(st.session_state.data_context['missing_values'])}
                            - Amostra: {str(st.session_state.data_context['sample_data'])}
                            
                            Responda de forma técnica mas acessível.
                            """
                            
                            messages = [{"role": "system", "content": context}]
                            messages.extend(st.session_state.conversation_history[-4:])
                            
                            response = call_ai_api(api_choice, api_key, messages, model)
                            st.session_state.conversation_history.append({"role": "assistant", "content": response})
                            
                            st.success("✅ Resposta:")
                            st.markdown(response)
                    else:
                        st.error("❌ Insira pergunta e chave da API.")
            
            with col2:
                if st.button("🗑️ Limpar"):
                    st.session_state.conversation_history = []
                    st.rerun()
            
            # Instruções
            if not api_key:
                st.info(f"🔑 Como obter chave {api_choice}:")
                if api_choice == "OpenAI":
                    st.markdown("1. [platform.openai.com](https://platform.openai.com) → API Keys")
                elif api_choice == "Groq":
                    st.markdown("1. [console.groq.com](https://console.groq.com) → API Keys")
                else:
                    st.markdown("1. [makersuite.google.com](https://makersuite.google.com) → Get API Key")
    
    except Exception as e:
        st.error(f"❌ Erro: {str(e)}")

else:
    st.markdown("""
    ## 🎯 Análise Exploratória de Dados com IA
    
    **Funcionalidades:**
    - 📊 **Distribuições**: Histogramas e gráficos de barras para todas as variáveis
    - 🔍 **Correlações**: Matriz de correlação e identificação de relações
    - 📈 **Tendências**: Linhas de tendência e análise temporal
    - ⚠️ **Anomalias**: Detecção de outliers com boxplots
    - 🤖 **Chat IA**: Conversação com memória sobre os dados
    
    **APIs Suportadas:** OpenAI, Groq, Gemini
    
    **Como usar:** Carregue um arquivo CSV e explore as análises automáticas.
    """)
