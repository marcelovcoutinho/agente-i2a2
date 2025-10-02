import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from groq import Groq
import openai
import google.generativeai as genai
import os
from io import StringIO
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Configuração da página
st.set_page_config(
    page_title="Agente de Análise de Dados CSV - Melhorado",
    page_icon="📊",
    layout="wide"
)

# Configurar estilo matplotlib para alto contraste
plt.style.use('dark_background')

# Função para inicializar o histórico de conversação
def init_conversation_history():
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'data_context' not in st.session_state:
        st.session_state.data_context = None

# Função para adicionar mensagem ao histórico
def add_to_conversation(role, content):
    st.session_state.conversation_history.append({"role": role, "content": content})

# Função para preparar contexto completo dos dados
def prepare_data_context(data):
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    
    context = {
        'shape': data.shape,
        'columns': data.columns.tolist(),
        'dtypes': data.dtypes.to_dict(),
        'numeric_columns': numeric_cols,
        'categorical_columns': categorical_cols,
        'missing_values': data.isnull().sum().to_dict(),
        'basic_stats': data.describe().to_dict() if numeric_cols else {},
        'categorical_stats': {col: data[col].value_counts().head(10).to_dict() for col in categorical_cols},
        'sample_data': data.head(5).to_dict('records')
    }
    return context

# Função para chamar diferentes APIs
def call_ai_api(api_choice, api_key, messages, model=None):
    try:
        if api_choice == "OpenAI":
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=model or "gpt-3.5-turbo",
                messages=messages,
                max_tokens=2000,
                temperature=0.7
            )
            return response.choices[0].message.content
            
        elif api_choice == "Groq":
            client = Groq(api_key=api_key)
            response = client.chat.completions.create(
                model=model or "llama-3.3-70b-versatile",
                messages=messages,
                max_tokens=2000,
                temperature=0.7
            )
            return response.choices[0].message.content
            
        elif api_choice == "Gemini":
            genai.configure(api_key=api_key)
            model_instance = genai.GenerativeModel(model or 'gemini-pro')
            
            # Converter mensagens para formato do Gemini
            prompt = ""
            for msg in messages:
                if msg["role"] == "system":
                    prompt += f"Instruções: {msg['content']}\n\n"
                elif msg["role"] == "user":
                    prompt += f"Usuário: {msg['content']}\n\n"
                elif msg["role"] == "assistant":
                    prompt += f"Assistente: {msg['content']}\n\n"
            
            response = model_instance.generate_content(prompt)
            return response.text
            
    except Exception as e:
        return f"Erro ao chamar a API: {str(e)}"

# Função para criar gráficos de distribuição melhorados
def create_distribution_plots(data, numeric_cols, categorical_cols):
    plots_created = []
    
    # Gráficos para variáveis numéricas
    if numeric_cols:
        st.subheader("📊 Distribuições - Variáveis Numéricas")
        
        # Calcular número de colunas e linhas para o grid
        n_cols = len(numeric_cols)
        cols_per_row = 3
        rows = (n_cols + cols_per_row - 1) // cols_per_row
        
        fig, axes = plt.subplots(rows, cols_per_row, figsize=(18, 6*rows))
        fig.patch.set_facecolor('#0E1117')
        
        # Garantir que axes seja sempre um array 2D
        if rows == 1:
            if cols_per_row == 1:
                axes = np.array([[axes]])
            else:
                axes = axes.reshape(1, -1)
        elif cols_per_row == 1:
            axes = axes.reshape(-1, 1)
        
        for i, col in enumerate(numeric_cols):
            row = i // cols_per_row
            col_idx = i % cols_per_row
            ax = axes[row, col_idx]
            
            # Criar histograma com densidade
            data[col].hist(ax=ax, bins=30, alpha=0.7, color='cyan', density=True, edgecolor='white')
            
            # Adicionar curva de densidade
            try:
                data_clean = data[col].dropna()
                if len(data_clean) > 1:
                    kde = stats.gaussian_kde(data_clean)
                    x_range = np.linspace(data_clean.min(), data_clean.max(), 100)
                    ax.plot(x_range, kde(x_range), color='orange', linewidth=2, label='Densidade')
                    ax.legend()
            except:
                pass
            
            ax.set_title(f'Distribuição: {col}', color='white', fontsize=12, pad=10)
            ax.set_xlabel(col, color='white')
            ax.set_ylabel('Densidade', color='white')
            ax.set_facecolor('#0E1117')
            ax.tick_params(colors='white')
            ax.grid(True, alpha=0.3)
        
        # Remover subplots vazios
        for i in range(n_cols, rows * cols_per_row):
            row = i // cols_per_row
            col_idx = i % cols_per_row
            fig.delaxes(axes[row, col_idx])
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        plots_created.append("Distribuições numéricas")
    
    # Gráficos para variáveis categóricas
    if categorical_cols:
        st.subheader("📊 Distribuições - Variáveis Categóricas")
        
        for col in categorical_cols:
            value_counts = data[col].value_counts().head(15)  # Top 15 valores
            
            fig, ax = plt.subplots(figsize=(12, 8))
            fig.patch.set_facecolor('#0E1117')
            
            bars = ax.bar(range(len(value_counts)), value_counts.values, 
                         color='lightcoral', alpha=0.8, edgecolor='white')
            
            ax.set_title(f'Distribuição: {col}', color='white', fontsize=16, pad=20)
            ax.set_xticks(range(len(value_counts)))
            ax.set_xticklabels(value_counts.index, rotation=45, ha='right', color='white')
            ax.set_ylabel('Frequência', color='white')
            ax.set_facecolor('#0E1117')
            ax.tick_params(colors='white')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Adicionar valores nas barras
            for bar, value in zip(bars, value_counts.values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height + max(value_counts.values)*0.01,
                       f'{value:,}', ha='center', va='bottom', color='white', fontsize=10)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            plots_created.append(f"Distribuição categórica: {col}")
    
    return plots_created

# Função para análise de tendências melhorada
def create_trend_analysis(data, numeric_cols, categorical_cols):
    st.header("📈 Análise de Tendências Melhorada")
    
    # Detectar colunas temporais
    time_cols = []
    for col in data.columns:
        if any(keyword in col.lower() for keyword in ['time', 'date', 'timestamp', 'year', 'month', 'day']):
            time_cols.append(col)
    
    if time_cols and numeric_cols:
        st.subheader("📅 Tendências Temporais")
        
        col1, col2 = st.columns(2)
        with col1:
            time_col = st.selectbox("Selecione a coluna temporal:", time_cols)
        with col2:
            numeric_col = st.selectbox("Selecione a variável numérica:", numeric_cols)
        
        if time_col and numeric_col:
            fig, ax = plt.subplots(figsize=(14, 8))
            fig.patch.set_facecolor('#0E1117')
            
            # Ordenar dados por tempo
            data_sorted = data.sort_values(time_col).copy()
            x_values = range(len(data_sorted))
            y_values = data_sorted[numeric_col]
            
            # Plotar dados originais
            ax.scatter(x_values, y_values, alpha=0.6, color='cyan', s=20, label='Dados')
            
            # Adicionar linha de tendência
            try:
                # Remover valores nulos
                mask = ~(pd.isna(y_values))
                x_clean = np.array(x_values)[mask]
                y_clean = np.array(y_values)[mask]
                
                if len(x_clean) > 1:
                    # Calcular linha de tendência
                    z = np.polyfit(x_clean, y_clean, 1)
                    p = np.poly1d(z)
                    ax.plot(x_clean, p(x_clean), "r--", alpha=0.8, linewidth=2, label=f'Tendência (slope: {z[0]:.4f})')
                    
                    # Adicionar média móvel
                    if len(y_clean) > 10:
                        window = min(20, len(y_clean) // 5)
                        moving_avg = pd.Series(y_clean).rolling(window=window, center=True).mean()
                        ax.plot(x_clean, moving_avg, color='orange', linewidth=2, alpha=0.8, label=f'Média Móvel ({window})')
            except Exception as e:
                st.warning(f"Não foi possível calcular a linha de tendência: {e}")
            
            ax.set_title(f'Tendência Temporal: {numeric_col} vs {time_col}', color='white', fontsize=16, pad=20)
            ax.set_xlabel('Índice Temporal', color='white')
            ax.set_ylabel(numeric_col, color='white')
            ax.set_facecolor('#0E1117')
            ax.tick_params(colors='white')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    
    # Análise de correlação temporal para variáveis numéricas
    if len(numeric_cols) > 1:
        st.subheader("🔄 Correlações e Tendências entre Variáveis")
        
        # Matriz de correlação
        corr_matrix = data[numeric_cols].corr()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        fig.patch.set_facecolor('#0E1117')
        
        # Heatmap de correlação
        sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', center=0,
                   square=True, fmt='.2f', cbar_kws={'shrink': 0.8}, ax=ax1)
        ax1.set_title('Matriz de Correlação', color='white', fontsize=14)
        ax1.set_facecolor('#0E1117')
        
        # Scatter plot das duas variáveis mais correlacionadas
        if len(numeric_cols) >= 2:
            # Encontrar a correlação mais forte (excluindo diagonal)
            corr_abs = corr_matrix.abs()
            np.fill_diagonal(corr_abs.values, 0)
            max_corr_idx = np.unravel_index(corr_abs.values.argmax(), corr_abs.shape)
            var1, var2 = corr_matrix.columns[max_corr_idx[0]], corr_matrix.columns[max_corr_idx[1]]
            corr_value = corr_matrix.loc[var1, var2]
            
            ax2.scatter(data[var1], data[var2], alpha=0.6, color='lightgreen', s=30)
            
            # Adicionar linha de regressão
            try:
                mask = ~(pd.isna(data[var1]) | pd.isna(data[var2]))
                if mask.sum() > 1:
                    z = np.polyfit(data[var1][mask], data[var2][mask], 1)
                    p = np.poly1d(z)
                    x_range = np.linspace(data[var1].min(), data[var1].max(), 100)
                    ax2.plot(x_range, p(x_range), "r--", alpha=0.8, linewidth=2)
            except:
                pass
            
            ax2.set_title(f'Correlação: {var1} vs {var2}\n(r = {corr_value:.3f})', color='white', fontsize=14)
            ax2.set_xlabel(var1, color='white')
            ax2.set_ylabel(var2, color='white')
            ax2.set_facecolor('#0E1117')
            ax2.tick_params(colors='white')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# Função para detecção completa de outliers
def create_complete_outlier_analysis(data, numeric_cols):
    st.header("⚠️ Análise Completa de Outliers")
    
    if not numeric_cols:
        st.info("Não há variáveis numéricas para análise de outliers.")
        return
    
    # Tabela resumo de outliers
    st.subheader("📊 Resumo de Outliers por Variável")
    
    outliers_summary = []
    outliers_data = {}
    
    for col in numeric_cols:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
        outliers_data[col] = outliers
        
        # Z-score outliers
        z_scores = np.abs(stats.zscore(data[col].dropna()))
        z_outliers = len(z_scores[z_scores > 3])
        
        outliers_summary.append({
            'Variável': col,
            'Outliers IQR': len(outliers),
            '% IQR': f"{(len(outliers) / len(data) * 100):.2f}%",
            'Outliers Z-score': z_outliers,
            '% Z-score': f"{(z_outliers / len(data) * 100):.2f}%",
            'Limite Inferior': f"{lower_bound:.2f}",
            'Limite Superior': f"{upper_bound:.2f}",
            'Min': f"{data[col].min():.2f}",
            'Max': f"{data[col].max():.2f}"
        })
    
    outliers_df = pd.DataFrame(outliers_summary)
    st.dataframe(outliers_df, use_container_width=True)
    
    # Visualizações completas de outliers
    st.subheader("📈 Visualizações de Outliers - Todos os Gráficos")
    
    # Boxplots para todas as variáveis
    n_cols = len(numeric_cols)
    cols_per_row = 4
    rows = (n_cols + cols_per_row - 1) // cols_per_row
    
    fig, axes = plt.subplots(rows, cols_per_row, figsize=(20, 5*rows))
    fig.patch.set_facecolor('#0E1117')
    
    # Garantir que axes seja sempre um array 2D
    if rows == 1:
        if cols_per_row == 1:
            axes = np.array([[axes]])
        else:
            axes = axes.reshape(1, -1)
    elif cols_per_row == 1:
        axes = axes.reshape(-1, 1)
    
    for i, col in enumerate(numeric_cols):
        row = i // cols_per_row
        col_idx = i % cols_per_row
        ax = axes[row, col_idx]
        
        # Boxplot
        bp = ax.boxplot(data[col].dropna(), patch_artist=True, 
                       boxprops=dict(facecolor='lightblue', alpha=0.7),
                       medianprops=dict(color='red', linewidth=2),
                       whiskerprops=dict(color='white'),
                       capprops=dict(color='white'),
                       flierprops=dict(marker='o', markerfacecolor='red', markersize=4, alpha=0.7))
        
        ax.set_title(f'{col}', color='white', fontsize=12)
        ax.set_facecolor('#0E1117')
        ax.tick_params(colors='white')
        ax.grid(True, alpha=0.3)
        
        # Adicionar estatísticas
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        median = data[col].median()
        ax.text(0.02, 0.98, f'Q1: {Q1:.2f}\nMediana: {median:.2f}\nQ3: {Q3:.2f}', 
               transform=ax.transAxes, verticalalignment='top', 
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.7),
               color='white', fontsize=8)
    
    # Remover subplots vazios
    for i in range(n_cols, rows * cols_per_row):
        row = i // cols_per_row
        col_idx = i % cols_per_row
        fig.delaxes(axes[row, col_idx])
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # Gráficos de dispersão para identificar outliers multivariados
    if len(numeric_cols) >= 2:
        st.subheader("🎯 Outliers Multivariados - Scatter Plots")
        
        # Pegar as 4 variáveis com mais outliers
        top_outlier_vars = outliers_df.nlargest(4, 'Outliers IQR')['Variável'].tolist()
        
        if len(top_outlier_vars) >= 2:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.patch.set_facecolor('#0E1117')
            
            combinations = [(top_outlier_vars[i], top_outlier_vars[j]) 
                          for i in range(len(top_outlier_vars)) 
                          for j in range(i+1, len(top_outlier_vars))][:4]
            
            for idx, (var1, var2) in enumerate(combinations):
                row = idx // 2
                col = idx % 2
                ax = axes[row, col]
                
                # Scatter plot normal
                ax.scatter(data[var1], data[var2], alpha=0.6, color='lightblue', s=30, label='Normal')
                
                # Destacar outliers
                outliers_var1 = outliers_data[var1]
                outliers_var2 = outliers_data[var2]
                
                if len(outliers_var1) > 0:
                    ax.scatter(outliers_var1[var1], outliers_var1[var2], 
                             color='red', s=50, alpha=0.8, label=f'Outliers {var1}')
                
                if len(outliers_var2) > 0:
                    ax.scatter(outliers_var2[var1], outliers_var2[var2], 
                             color='orange', s=50, alpha=0.8, label=f'Outliers {var2}')
                
                ax.set_title(f'{var1} vs {var2}', color='white', fontsize=12)
                ax.set_xlabel(var1, color='white')
                ax.set_ylabel(var2, color='white')
                ax.set_facecolor('#0E1117')
                ax.tick_params(colors='white')
                ax.grid(True, alpha=0.3)
                ax.legend()
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

# Título principal
st.title("🤖 Agente de Análise Exploratória de Dados - VERSÃO MELHORADA")
st.markdown("**Ferramenta inteligente com memória conversacional e múltiplas APIs de IA**")

# Inicializar histórico de conversação
init_conversation_history()

# Upload do arquivo
uploaded_file = st.file_uploader(
    "Carregue seu arquivo CSV para análise", 
    type=['csv'],
    help="Selecione um arquivo CSV para realizar a análise exploratória completa"
)

if uploaded_file is not None:
    # Carregar os dados
    try:
        data = pd.read_csv(uploaded_file)
        st.success(f"✅ Arquivo carregado com sucesso! {data.shape[0]} linhas e {data.shape[1]} colunas.")
        
        # Preparar contexto dos dados para IA
        st.session_state.data_context = prepare_data_context(data)
        
        # Criar abas para organizar a análise
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📋 Visão Geral", 
            "📊 Distribuições", 
            "🔍 Correlações", 
            "📈 Tendências", 
            "⚠️ Anomalias", 
            "🤖 Chat com IA"
        ])
        
        # Separar variáveis numéricas e categóricas
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        with tab1:
            st.header("📋 Visão Geral dos Dados")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Informações Básicas")
                st.write(f"**Número de linhas:** {data.shape[0]:,}")
                st.write(f"**Número de colunas:** {data.shape[1]:,}")
                st.write(f"**Tamanho em memória:** {data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
                st.write(f"**Variáveis numéricas:** {len(numeric_cols)}")
                st.write(f"**Variáveis categóricas:** {len(categorical_cols)}")
                
                st.subheader("Tipos de Dados")
                tipos_dados = pd.DataFrame({
                    'Coluna': data.dtypes.index,
                    'Tipo': data.dtypes.values.astype(str),
                    'Valores Únicos': [data[col].nunique() for col in data.columns],
                    'Valores Nulos': [data[col].isnull().sum() for col in data.columns],
                    '% Nulos': [f"{(data[col].isnull().sum() / len(data) * 100):.1f}%" for col in data.columns]
                })
                st.dataframe(tipos_dados, use_container_width=True)
            
            with col2:
                st.subheader("Primeiras 10 Linhas")
                st.dataframe(data.head(10), use_container_width=True)
                
                if numeric_cols:
                    st.subheader("Estatísticas Descritivas")
                    st.dataframe(data[numeric_cols].describe(), use_container_width=True)
        
        with tab2:
            create_distribution_plots(data, numeric_cols, categorical_cols)
        
        with tab3:
            st.header("🔍 Correlações entre Variáveis")
            
            if len(numeric_cols) > 1:
                # Matriz de correlação detalhada
                correlation_matrix = data[numeric_cols].corr()
                
                fig, ax = plt.subplots(figsize=(14, 12))
                fig.patch.set_facecolor('#0E1117')
                
                # Usar colormap com bom contraste
                mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
                sns.heatmap(correlation_matrix, 
                           mask=mask,
                           annot=True, 
                           cmap='RdYlBu_r', 
                           center=0,
                           square=True,
                           fmt='.3f',
                           cbar_kws={'shrink': 0.8},
                           ax=ax)
                
                ax.set_title('Matriz de Correlação (Triângulo Superior)', color='white', fontsize=16, pad=20)
                ax.set_facecolor('#0E1117')
                plt.xticks(rotation=45, ha='right', color='white')
                plt.yticks(rotation=0, color='white')
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                # Correlações mais fortes
                st.subheader("🎯 Correlações Mais Significativas")
                correlations = []
                for i in range(len(correlation_matrix.columns)):
                    for j in range(i+1, len(correlation_matrix.columns)):
                        corr_value = correlation_matrix.iloc[i, j]
                        if abs(corr_value) > 0.1:  # Apenas correlações significativas
                            correlations.append({
                                'Variável 1': correlation_matrix.columns[i],
                                'Variável 2': correlation_matrix.columns[j],
                                'Correlação': corr_value,
                                'Correlação Abs': abs(corr_value),
                                'Força': 'Muito Forte' if abs(corr_value) > 0.9 else 'Forte' if abs(corr_value) > 0.7 else 'Moderada' if abs(corr_value) > 0.3 else 'Fraca'
                            })
                
                if correlations:
                    corr_df = pd.DataFrame(correlations).sort_values('Correlação Abs', ascending=False)
                    st.dataframe(corr_df[['Variável 1', 'Variável 2', 'Correlação', 'Força']], use_container_width=True)
                else:
                    st.info("Não foram encontradas correlações significativas entre as variáveis.")
            else:
                st.info("É necessário ter pelo menos 2 variáveis numéricas para calcular correlações.")
        
        with tab4:
            create_trend_analysis(data, numeric_cols, categorical_cols)
        
        with tab5:
            create_complete_outlier_analysis(data, numeric_cols)
        
        with tab6:
            st.header("🤖 Chat Inteligente com Memória")
            st.markdown("Converse com a IA sobre seus dados! A IA tem acesso completo aos dados e lembra da conversa.")
            
            # Configuração da API
            col1, col2, col3 = st.columns(3)
            
            with col1:
                api_choice = st.selectbox(
                    "🔧 Escolha a API:",
                    ["OpenAI", "Groq", "Gemini"],
                    help="Selecione qual API de IA usar"
                )
            
            with col2:
                api_key = st.text_input(
                    f"🔑 Chave da API {api_choice}:", 
                    type="password",
                    help="Sua chave será usada apenas para esta sessão"
                )
            
            with col3:
                # Modelos por API
                if api_choice == "OpenAI":
                    model_options = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview"]
                elif api_choice == "Groq":
                    model_options = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "openai/gpt-oss-120b"]
                else:  # Gemini
                    model_options = ["gemini-pro", "gemini-pro-vision"]
                
                selected_model = st.selectbox("🧠 Modelo:", model_options)
            
            # Mostrar histórico de conversação
            if st.session_state.conversation_history:
                st.subheader("💬 Histórico da Conversa")
                for msg in st.session_state.conversation_history[-10:]:  # Últimas 10 mensagens
                    if msg["role"] == "user":
                        st.markdown(f"**👤 Você:** {msg['content']}")
                    else:
                        st.markdown(f"**🤖 IA:** {msg['content']}")
                st.markdown("---")
            
            # Input para nova pergunta
            user_question = st.text_area(
                "💭 Faça uma pergunta sobre seus dados:",
                placeholder="Exemplo: Quais são os principais insights deste dataset? Existe alguma correlação interessante? Quais outliers são mais preocupantes?",
                height=100,
                key="user_input"
            )
            
            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                if st.button("🚀 Enviar Pergunta", type="primary"):
                    if user_question.strip() and api_key:
                        # Adicionar pergunta ao histórico
                        add_to_conversation("user", user_question)
                        
                        with st.spinner(f"🤖 Processando com {api_choice}..."):
                            # Preparar contexto completo
                            context_prompt = f"""
                            Você é um especialista em análise de dados. Você tem acesso completo aos seguintes dados:
                            
                            INFORMAÇÕES DO DATASET:
                            - Formato: {st.session_state.data_context['shape'][0]} linhas x {st.session_state.data_context['shape'][1]} colunas
                            - Colunas: {', '.join(st.session_state.data_context['columns'])}
                            - Colunas numéricas: {', '.join(st.session_state.data_context['numeric_columns'])}
                            - Colunas categóricas: {', '.join(st.session_state.data_context['categorical_columns'])}
                            
                            ESTATÍSTICAS BÁSICAS:
                            {str(st.session_state.data_context['basic_stats'])}
                            
                            VALORES AUSENTES:
                            {str(st.session_state.data_context['missing_values'])}
                            
                            AMOSTRA DOS DADOS:
                            {str(st.session_state.data_context['sample_data'])}
                            
                            ESTATÍSTICAS CATEGÓRICAS:
                            {str(st.session_state.data_context['categorical_stats'])}
                            
                            Responda de forma detalhada e técnica, mas acessível. Use os dados reais para fundamentar suas respostas.
                            """
                            
                            # Preparar mensagens incluindo histórico
                            messages = [{"role": "system", "content": context_prompt}]
                            
                            # Adicionar histórico recente (últimas 6 mensagens para não exceder limite)
                            recent_history = st.session_state.conversation_history[-6:]
                            for msg in recent_history:
                                messages.append(msg)
                            
                            # Chamar API
                            response = call_ai_api(api_choice, api_key, messages, selected_model)
                            
                            # Adicionar resposta ao histórico
                            add_to_conversation("assistant", response)
                            
                            # Mostrar resposta
                            st.success("✅ Resposta recebida!")
                            st.markdown("### 🎯 Resposta da IA:")
                            st.markdown(response)
                    
                    elif not api_key:
                        st.error("❌ Por favor, insira sua chave da API.")
                    else:
                        st.error("❌ Por favor, digite uma pergunta.")
            
            with col2:
                if st.button("🗑️ Limpar Chat"):
                    st.session_state.conversation_history = []
                    st.rerun()
            
            with col3:
                st.markdown("**💡 Dicas de perguntas:**")
                st.markdown("• Quais são os principais insights?")
                st.markdown("• Existem padrões interessantes?")
                st.markdown("• Como interpretar os outliers?")
                st.markdown("• Que análises adicionais recomendar?")
            
            # Instruções para obter chaves das APIs
            if not api_key:
                st.info(f"🔑 Insira sua chave da API {api_choice} para usar o chat inteligente.")
                
                if api_choice == "OpenAI":
                    st.markdown("""
                    **Como obter chave OpenAI:**
                    1. Acesse [platform.openai.com](https://platform.openai.com)
                    2. Faça login → API Keys → Create new key
                    """)
                elif api_choice == "Groq":
                    st.markdown("""
                    **Como obter chave Groq:**
                    1. Acesse [console.groq.com](https://console.groq.com)
                    2. Crie conta gratuita → API Keys → Create API Key
                    """)
                else:  # Gemini
                    st.markdown("""
                    **Como obter chave Gemini:**
                    1. Acesse [makersuite.google.com](https://makersuite.google.com)
                    2. Get API Key → Create API Key
                    """)
    
    except Exception as e:
        st.error(f"❌ Erro ao carregar o arquivo: {str(e)}")
        st.info("Verifique se o arquivo está no formato CSV correto.")

else:
    # Página inicial
    st.markdown("""
    ## 🎯 Bem-vindo ao Agente de Análise de Dados MELHORADO!
    
    ### 🚀 Principais Melhorias:
    
    **🧠 Chat Inteligente com Memória**
    - Conversação contínua que lembra do contexto
    - Acesso completo aos dados carregados
    - Suporte para OpenAI, Groq e Gemini
    
    **📊 Visualizações Completas**
    - Todas as distribuições (numéricas e categóricas)
    - Gráficos de densidade e estatísticas
    - Análise completa de outliers
    
    **📈 Análise de Tendências Avançada**
    - Linhas de tendência automáticas
    - Médias móveis
    - Correlações visuais
    
    **⚠️ Detecção Completa de Outliers**
    - Métodos IQR e Z-score
    - Visualizações para todas as variáveis
    - Análise multivariada
    
    ### 📤 Como usar:
    1. **Carregue** seu arquivo CSV
    2. **Explore** as análises automáticas
    3. **Converse** com a IA sobre seus dados
    4. **Obtenha insights** personalizados
    
    **💡 A IA tem acesso completo aos seus dados e pode responder qualquer pergunta sobre eles!**
    """)
