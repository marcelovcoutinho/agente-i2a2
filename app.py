import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import openai
import os
from io import StringIO

# Configuração da página
st.set_page_config(
    page_title="Agente de Análise de Dados CSV",
    page_icon="📊",
    layout="wide"
)

# Título principal
st.title("🤖 Agente de Análise Exploratória de Dados (E.D.A.)")
st.markdown("**Ferramenta inteligente para análise de qualquer arquivo CSV**")

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
        
        # Criar abas para organizar a análise
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📋 Visão Geral", 
            "📊 Distribuições", 
            "🔍 Correlações", 
            "📈 Tendências", 
            "⚠️ Anomalias", 
            "🤖 Consulta IA"
        ])
        
        with tab1:
            st.header("📋 Visão Geral dos Dados")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Informações Básicas")
                st.write(f"**Número de linhas:** {data.shape[0]:,}")
                st.write(f"**Número de colunas:** {data.shape[1]:,}")
                st.write(f"**Tamanho em memória:** {data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
                
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
                
                st.subheader("Estatísticas Descritivas")
                st.dataframe(data.describe(), use_container_width=True)
        
        with tab2:
            st.header("📊 Distribuição das Variáveis")
            
            # Separar variáveis numéricas e categóricas
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if numeric_cols:
                st.subheader("Variáveis Numéricas")
                
                # Configurar estilo com alto contraste
                plt.style.use('dark_background')
                
                # Criar histogramas para variáveis numéricas
                num_cols_to_show = min(len(numeric_cols), 12)  # Limitar para não sobrecarregar
                cols_per_row = 3
                rows = (num_cols_to_show + cols_per_row - 1) // cols_per_row
                
                fig, axes = plt.subplots(rows, cols_per_row, figsize=(15, 5*rows))
                fig.patch.set_facecolor('#0E1117')
                
                if rows == 1:
                    axes = axes.reshape(1, -1) if num_cols_to_show > 1 else [axes]
                
                for i, col in enumerate(numeric_cols[:num_cols_to_show]):
                    row = i // cols_per_row
                    col_idx = i % cols_per_row
                    
                    ax = axes[row][col_idx] if rows > 1 else axes[col_idx]
                    
                    # Remover outliers extremos para melhor visualização
                    Q1 = data[col].quantile(0.01)
                    Q3 = data[col].quantile(0.99)
                    filtered_data = data[col][(data[col] >= Q1) & (data[col] <= Q3)]
                    
                    ax.hist(filtered_data, bins=30, color='cyan', alpha=0.7, edgecolor='white')
                    ax.set_title(f'Distribuição: {col}', color='white', fontsize=10)
                    ax.set_facecolor('#0E1117')
                    ax.tick_params(colors='white', labelsize=8)
                    ax.grid(True, alpha=0.3)
                
                # Remover subplots vazios
                for i in range(num_cols_to_show, rows * cols_per_row):
                    row = i // cols_per_row
                    col_idx = i % cols_per_row
                    if rows > 1:
                        fig.delaxes(axes[row][col_idx])
                    else:
                        fig.delaxes(axes[col_idx])
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            
            if categorical_cols:
                st.subheader("Variáveis Categóricas")
                
                for col in categorical_cols[:6]:  # Limitar a 6 variáveis categóricas
                    value_counts = data[col].value_counts().head(10)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    fig.patch.set_facecolor('#0E1117')
                    
                    bars = ax.bar(range(len(value_counts)), value_counts.values, color='lightcoral', alpha=0.8)
                    ax.set_title(f'Distribuição: {col}', color='white', fontsize=14)
                    ax.set_xticks(range(len(value_counts)))
                    ax.set_xticklabels(value_counts.index, rotation=45, ha='right', color='white')
                    ax.set_facecolor('#0E1117')
                    ax.tick_params(colors='white')
                    ax.grid(True, alpha=0.3)
                    
                    # Adicionar valores nas barras
                    for bar, value in zip(bars, value_counts.values):
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(value_counts.values)*0.01,
                               f'{value:,}', ha='center', va='bottom', color='white', fontsize=9)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
        
        with tab3:
            st.header("🔍 Correlações entre Variáveis")
            
            if len(numeric_cols) > 1:
                # Matriz de correlação
                correlation_matrix = data[numeric_cols].corr()
                
                fig, ax = plt.subplots(figsize=(12, 10))
                fig.patch.set_facecolor('#0E1117')
                
                # Usar colormap com bom contraste
                sns.heatmap(correlation_matrix, 
                           annot=True, 
                           cmap='RdYlBu_r', 
                           center=0,
                           square=True,
                           fmt='.2f',
                           cbar_kws={'shrink': 0.8},
                           ax=ax)
                
                ax.set_title('Matriz de Correlação', color='white', fontsize=16, pad=20)
                ax.set_facecolor('#0E1117')
                plt.xticks(rotation=45, ha='right', color='white')
                plt.yticks(rotation=0, color='white')
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                # Correlações mais fortes
                st.subheader("Correlações Mais Significativas")
                correlations = []
                for i in range(len(correlation_matrix.columns)):
                    for j in range(i+1, len(correlation_matrix.columns)):
                        corr_value = correlation_matrix.iloc[i, j]
                        if abs(corr_value) > 0.1:  # Apenas correlações significativas
                            correlations.append({
                                'Variável 1': correlation_matrix.columns[i],
                                'Variável 2': correlation_matrix.columns[j],
                                'Correlação': corr_value,
                                'Força': 'Forte' if abs(corr_value) > 0.7 else 'Moderada' if abs(corr_value) > 0.3 else 'Fraca'
                            })
                
                if correlations:
                    corr_df = pd.DataFrame(correlations).sort_values('Correlação', key=abs, ascending=False)
                    st.dataframe(corr_df, use_container_width=True)
                else:
                    st.info("Não foram encontradas correlações significativas entre as variáveis.")
            else:
                st.info("É necessário ter pelo menos 2 variáveis numéricas para calcular correlações.")
        
        with tab4:
            st.header("📈 Análise de Tendências")
            
            # Verificar se existe coluna de tempo/data
            time_cols = []
            for col in data.columns:
                if 'time' in col.lower() or 'date' in col.lower() or 'timestamp' in col.lower():
                    time_cols.append(col)
            
            if time_cols:
                st.subheader("Tendências Temporais")
                time_col = st.selectbox("Selecione a coluna temporal:", time_cols)
                
                if time_col and len(numeric_cols) > 0:
                    numeric_col = st.selectbox("Selecione a variável para análise temporal:", numeric_cols)
                    
                    fig, ax = plt.subplots(figsize=(12, 6))
                    fig.patch.set_facecolor('#0E1117')
                    
                    # Ordenar por tempo e plotar
                    data_sorted = data.sort_values(time_col)
                    ax.plot(range(len(data_sorted)), data_sorted[numeric_col], color='cyan', alpha=0.7)
                    ax.set_title(f'Tendência Temporal: {numeric_col}', color='white', fontsize=14)
                    ax.set_xlabel('Índice Temporal', color='white')
                    ax.set_ylabel(numeric_col, color='white')
                    ax.set_facecolor('#0E1117')
                    ax.tick_params(colors='white')
                    ax.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
            else:
                st.info("Não foram identificadas colunas temporais no dataset.")
            
            # Análise de padrões em variáveis categóricas
            if categorical_cols:
                st.subheader("Padrões em Variáveis Categóricas")
                cat_col = st.selectbox("Selecione uma variável categórica:", categorical_cols)
                
                if cat_col:
                    # Valores mais e menos frequentes
                    value_counts = data[cat_col].value_counts()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Valores Mais Frequentes:**")
                        st.dataframe(value_counts.head(10).reset_index())
                    
                    with col2:
                        st.write("**Valores Menos Frequentes:**")
                        st.dataframe(value_counts.tail(10).reset_index())
        
        with tab5:
            st.header("⚠️ Detecção de Anomalias")
            
            if numeric_cols:
                st.subheader("Outliers por Variável")
                
                outliers_summary = []
                
                for col in numeric_cols:
                    Q1 = data[col].quantile(0.25)
                    Q3 = data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
                    
                    outliers_summary.append({
                        'Variável': col,
                        'Total de Outliers': len(outliers),
                        'Percentual': f"{(len(outliers) / len(data) * 100):.2f}%",
                        'Limite Inferior': f"{lower_bound:.2f}",
                        'Limite Superior': f"{upper_bound:.2f}",
                        'Valor Mínimo': f"{data[col].min():.2f}",
                        'Valor Máximo': f"{data[col].max():.2f}"
                    })
                
                outliers_df = pd.DataFrame(outliers_summary)
                st.dataframe(outliers_df, use_container_width=True)
                
                # Boxplots para visualizar outliers
                st.subheader("Visualização de Outliers (Boxplots)")
                
                num_cols_to_plot = min(len(numeric_cols), 8)
                cols_per_row = 4
                rows = (num_cols_to_plot + cols_per_row - 1) // cols_per_row
                
                fig, axes = plt.subplots(rows, cols_per_row, figsize=(16, 4*rows))
                fig.patch.set_facecolor('#0E1117')
                
                if rows == 1:
                    axes = axes.reshape(1, -1) if num_cols_to_plot > 1 else [axes]
                
                for i, col in enumerate(numeric_cols[:num_cols_to_plot]):
                    row = i // cols_per_row
                    col_idx = i % cols_per_row
                    
                    ax = axes[row][col_idx] if rows > 1 else axes[col_idx]
                    
                    bp = ax.boxplot(data[col].dropna(), patch_artist=True)
                    bp['boxes'][0].set_facecolor('lightblue')
                    bp['boxes'][0].set_alpha(0.7)
                    
                    ax.set_title(f'{col}', color='white', fontsize=10)
                    ax.set_facecolor('#0E1117')
                    ax.tick_params(colors='white', labelsize=8)
                    ax.grid(True, alpha=0.3)
                
                # Remover subplots vazios
                for i in range(num_cols_to_plot, rows * cols_per_row):
                    row = i // cols_per_row
                    col_idx = i % cols_per_row
                    if rows > 1:
                        fig.delaxes(axes[row][col_idx])
                    else:
                        fig.delaxes(axes[col_idx])
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            else:
                st.info("Não há variáveis numéricas para análise de outliers.")
        
        with tab6:
            st.header("🤖 Consulta Inteligente com IA")
            st.markdown("Faça perguntas sobre seus dados e obtenha insights inteligentes!")
            
            # Input para chave da API
            api_key = st.text_input(
                "🔑 Insira sua chave da API da OpenAI:", 
                type="password",
                help="Sua chave será usada apenas para esta sessão e não será armazenada."
            )
            
            if api_key:
                # Configurar cliente OpenAI
                openai.api_key = api_key
                
                # Preparar contexto dos dados
                context = f"""
                CONTEXTO DO DATASET:
                - Número de linhas: {data.shape[0]:,}
                - Número de colunas: {data.shape[1]:,}
                - Colunas: {', '.join(data.columns.tolist())}
                
                TIPOS DE DADOS:
                {data.dtypes.to_string()}
                
                ESTATÍSTICAS DESCRITIVAS (variáveis numéricas):
                {data.describe().to_string() if len(numeric_cols) > 0 else 'Não há variáveis numéricas'}
                
                VALORES ÚNICOS POR COLUNA:
                {pd.Series({col: data[col].nunique() for col in data.columns}).to_string()}
                
                VALORES NULOS:
                {data.isnull().sum().to_string()}
                """
                
                # Input para pergunta do usuário
                user_question = st.text_area(
                    "💭 Sua pergunta sobre os dados:",
                    placeholder="Exemplo: Quais são as principais características deste dataset? Existem padrões interessantes? Como estão distribuídas as variáveis?",
                    height=100
                )
                
                if st.button("🚀 Analisar com IA", type="primary"):
                    if user_question.strip():
                        with st.spinner("🤖 Analisando seus dados..."):
                            try:
                                # Criar prompt otimizado
                                prompt = f"""
                                Você é um especialista em análise de dados. Analise o seguinte dataset e responda à pergunta do usuário de forma clara e detalhada.
                                
                                {context}
                                
                                PERGUNTA DO USUÁRIO: {user_question}
                                
                                Por favor, forneça uma resposta detalhada e insights úteis baseados nos dados apresentados. Se possível, sugira análises adicionais que poderiam ser interessantes.
                                """
                                
                                # Chamar API OpenAI
                                response = openai.ChatCompletion.create(
                                    model="gpt-3.5-turbo",
                                    messages=[
                                        {"role": "system", "content": "Você é um especialista em análise de dados e ciência de dados."},
                                        {"role": "user", "content": prompt}
                                    ],
                                    max_tokens=1000,
                                    temperature=0.7
                                )
                                
                                # Exibir resposta
                                st.success("✅ Análise concluída!")
                                st.markdown("### 🎯 Resposta da IA:")
                                st.markdown(response.choices[0].message.content)
                                
                            except Exception as e:
                                st.error(f"❌ Erro ao consultar a API da OpenAI: {str(e)}")
                                st.info("Verifique se sua chave da API está correta e se você tem créditos disponíveis.")
                    else:
                        st.warning("⚠️ Por favor, digite uma pergunta antes de analisar.")
            else:
                st.info("🔑 Insira sua chave da API da OpenAI para usar a funcionalidade de consulta inteligente.")
                st.markdown("""
                **Como obter sua chave da API:**
                1. Acesse [platform.openai.com](https://platform.openai.com)
                2. Faça login em sua conta
                3. Vá para API Keys
                4. Crie uma nova chave
                5. Cole a chave no campo acima
                """)
    
    except Exception as e:
        st.error(f"❌ Erro ao carregar o arquivo: {str(e)}")
        st.info("Verifique se o arquivo está no formato CSV correto.")

else:
    # Página inicial quando nenhum arquivo foi carregado
    st.markdown("""
    ## 🎯 Bem-vindo ao Agente de Análise de Dados!
    
    Esta ferramenta permite realizar análise exploratória completa de qualquer arquivo CSV de forma automática e inteligente.
    
    ### 🚀 Funcionalidades:
    
    **📋 Visão Geral**
    - Informações básicas do dataset
    - Tipos de dados e estatísticas descritivas
    - Identificação de valores nulos
    
    **📊 Distribuições**
    - Histogramas para variáveis numéricas
    - Gráficos de barras para variáveis categóricas
    - Visualizações com alto contraste
    
    **🔍 Correlações**
    - Matriz de correlação entre variáveis
    - Identificação de relações significativas
    - Análise de dependências
    
    **📈 Tendências**
    - Análise temporal (quando aplicável)
    - Padrões em variáveis categóricas
    - Valores mais e menos frequentes
    
    **⚠️ Anomalias**
    - Detecção automática de outliers
    - Visualização com boxplots
    - Estatísticas de anomalias
    
    **🤖 Consulta IA**
    - Integração com OpenAI GPT
    - Perguntas personalizadas sobre os dados
    - Insights inteligentes e sugestões
    
    ### 📤 Como usar:
    1. Carregue seu arquivo CSV usando o botão acima
    2. Explore as diferentes abas de análise
    3. Use a IA para fazer perguntas específicas sobre seus dados
    
    **💡 Dica:** A ferramenta funciona com qualquer arquivo CSV e se adapta automaticamente às suas características!
    """)
