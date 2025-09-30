"""
Interface Streamlit Ultra-Rápida
Carregamento instantâneo, análise em segundos
"""

import streamlit as st
import pandas as pd
import time
import tempfile
import os
from typing import Dict

# Import do agente rápido
from fast_eda import create_fast_agent

# Configuração da página
st.set_page_config(
    page_title="⚡ Agente EDA Ultra-Rápido",
    page_icon="⚡",
    layout="wide"
)

# CSS minimalista para velocidade
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-box {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .insight-box {
        background: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .speed-badge {
        background: #d4edda;
        color: #155724;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

def init_session():
    """Inicialização mínima do estado"""
    if 'agent' not in st.session_state:
        st.session_state.agent = create_fast_agent()
    if 'analysis' not in st.session_state:
        st.session_state.analysis = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'load_time' not in st.session_state:
        st.session_state.load_time = 0

def display_metrics(analysis: Dict):
    """Exibe métricas principais rapidamente"""
    basic = analysis['basic_info']
    missing = analysis['missing_values']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Linhas", f"{basic['rows']:,}")
    with col2:
        st.metric("📋 Colunas", basic['columns'])
    with col3:
        st.metric("💾 Memória", f"{basic['memory_mb']:.1f} MB")
    with col4:
        st.metric("⚠️ Missing", missing['total_missing'])

def display_insights(analysis: Dict):
    """Exibe insights rapidamente"""
    st.subheader("💡 Insights Instantâneos")
    
    for insight in analysis['insights']:
        st.markdown(f"""
        <div class="insight-box">
            {insight}
        </div>
        """, unsafe_allow_html=True)

def display_data_types(analysis: Dict):
    """Exibe tipos de dados"""
    dt = analysis['data_types']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🔢 Variáveis Numéricas:**")
        if dt['numeric']:
            for col in dt['numeric'][:10]:  # Máximo 10
                st.text(f"• {col}")
        else:
            st.text("Nenhuma")
    
    with col2:
        st.markdown("**📝 Variáveis Categóricas:**")
        if dt['categorical']:
            for col in dt['categorical'][:10]:  # Máximo 10
                st.text(f"• {col}")
        else:
            st.text("Nenhuma")

def display_statistics(analysis: Dict):
    """Exibe estatísticas básicas"""
    if not analysis['statistics']:
        st.info("📊 Nenhuma variável numérica para estatísticas")
        return
    
    stats = analysis['statistics']['summary']
    
    # Mostrar apenas primeiras 5 colunas para velocidade
    cols_to_show = list(stats.keys())[:5]
    
    if cols_to_show:
        st.subheader("📊 Estatísticas Descritivas")
        
        # Criar DataFrame para exibição
        display_stats = {}
        for col in cols_to_show:
            col_stats = stats[col]
            display_stats[col] = {
                'Média': f"{col_stats['mean']:.2f}",
                'Mediana': f"{col_stats['50%']:.2f}",
                'Desvio': f"{col_stats['std']:.2f}",
                'Min': f"{col_stats['min']:.2f}",
                'Max': f"{col_stats['max']:.2f}"
            }
        
        df_stats = pd.DataFrame(display_stats).T
        st.dataframe(df_stats, use_container_width=True)

def display_correlations(analysis: Dict):
    """Exibe correlações se existirem"""
    if not analysis['statistics'] or 'correlations' not in analysis['statistics']:
        return
    
    corrs = analysis['statistics']['correlations'].get('strong_correlations', [])
    
    if corrs:
        st.subheader("🔗 Correlações Fortes")
        
        for corr in corrs:
            strength = "Muito Forte" if abs(corr['correlation']) > 0.8 else "Forte"
            st.markdown(f"• **{corr['var1']}** ↔ **{corr['var2']}**: {corr['correlation']} ({strength})")

def display_chat():
    """Interface de chat simplificada"""
    st.subheader("💬 Chat Rápido")
    
    # Perguntas sugeridas
    with st.expander("💡 Perguntas Rápidas"):
        questions = [
            "Quantas linhas tem?",
            "Quais são as colunas?",
            "Existem valores ausentes?",
            "Quais são as estatísticas?",
            "Há correlações?",
            "Quais os insights?"
        ]
        
        cols = st.columns(3)
        for i, q in enumerate(questions):
            if cols[i % 3].button(q, key=f"q_{i}"):
                response = st.session_state.agent.answer_fast(q)
                st.session_state.chat_history.append({'q': q, 'a': response})
                st.rerun()
    
    # Input manual
    user_question = st.text_input("Sua pergunta:", placeholder="Ex: Quantas linhas tem este dataset?")
    
    if st.button("🚀 Perguntar") and user_question:
        start_time = time.time()
        response = st.session_state.agent.answer_fast(user_question)
        response_time = time.time() - start_time
        
        st.session_state.chat_history.append({
            'q': user_question, 
            'a': response, 
            'time': response_time
        })
        st.rerun()
    
    # Histórico
    if st.session_state.chat_history:
        st.markdown("### 📝 Histórico")
        for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):  # Últimas 5
            with st.expander(f"❓ {chat['q']}", expanded=(i==0)):
                st.markdown(chat['a'])
                if 'time' in chat:
                    st.markdown(f"<span class='speed-badge'>⚡ {chat['time']:.3f}s</span>", unsafe_allow_html=True)

def main():
    """Função principal ultra-otimizada"""
    init_session()
    
    # Cabeçalho
    st.markdown('<h1 class="main-header">⚡ Agente EDA Ultra-Rápido</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    🚀 **Versão Ultra-Otimizada:**
    - ⚡ Carregamento instantâneo
    - 📊 Análise em segundos
    - 💬 Respostas imediatas
    - 🎯 Zero dependências pesadas
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Upload de Dados")
        
        uploaded_file = st.file_uploader("Escolha um CSV", type=['csv'])
        
        if uploaded_file:
            start_time = time.time()
            
            # Salvar temporariamente
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            
            # Carregar e analisar
            with st.spinner("⚡ Analisando..."):
                result = st.session_state.agent.load_csv_fast(tmp_path)
            
            # Limpar arquivo temporário
            os.unlink(tmp_path)
            
            load_time = time.time() - start_time
            st.session_state.load_time = load_time
            
            if result['success']:
                st.session_state.analysis = result['analysis']
                st.success(f"✅ Carregado em {load_time:.2f}s!")
                st.session_state.chat_history = []  # Reset chat
                st.rerun()
            else:
                st.error(f"❌ {result['message']}")
        
        # Botão para dados de exemplo
        if st.button("📊 Dados de Exemplo"):
            start_time = time.time()
            
            # Criar dados de exemplo pequenos
            import numpy as np
            data = {
                'Vendas': np.random.randint(100, 1000, 500),
                'Lucro': np.random.randint(10, 100, 500),
                'Região': np.random.choice(['Norte', 'Sul', 'Leste', 'Oeste'], 500),
                'Produto': np.random.choice(['A', 'B', 'C'], 500)
            }
            df = pd.DataFrame(data)
            df.to_csv('/tmp/exemplo.csv', index=False)
            
            with st.spinner("⚡ Carregando exemplo..."):
                result = st.session_state.agent.load_csv_fast('/tmp/exemplo.csv')
            
            load_time = time.time() - start_time
            st.session_state.load_time = load_time
            
            if result['success']:
                st.session_state.analysis = result['analysis']
                st.success(f"✅ Exemplo carregado em {load_time:.2f}s!")
                st.session_state.chat_history = []
                st.rerun()
        
        # Estatísticas de performance
        if st.session_state.analysis:
            st.divider()
            st.header("📈 Performance")
            st.metric("⚡ Tempo de Carga", f"{st.session_state.load_time:.2f}s")
            st.metric("💬 Perguntas", len(st.session_state.chat_history))
    
    # Área principal
    if st.session_state.analysis:
        analysis = st.session_state.analysis
        
        # Indicador de velocidade
        st.markdown(f"""
        <div class="metric-box">
            ⚡ <strong>Análise Ultra-Rápida:</strong> Processado em {st.session_state.load_time:.2f}s 
            (sem dependências pesadas)
        </div>
        """, unsafe_allow_html=True)
        
        # Métricas principais
        display_metrics(analysis)
        
        # Tabs para organizar informações
        tab1, tab2, tab3, tab4 = st.tabs(["💡 Insights", "📊 Estatísticas", "🔗 Correlações", "💬 Chat"])
        
        with tab1:
            display_insights(analysis)
            st.subheader("📋 Tipos de Dados")
            display_data_types(analysis)
        
        with tab2:
            display_statistics(analysis)
        
        with tab3:
            display_correlations(analysis)
        
        with tab4:
            display_chat()
    
    else:
        # Tela inicial
        st.info("👆 Carregue um arquivo CSV ou use dados de exemplo")
        
        st.subheader("🎯 Características Ultra-Rápidas")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **⚡ Performance Extrema:**
            - Carregamento < 1 segundo
            - Análise < 3 segundos
            - Respostas instantâneas
            - Zero dependências pesadas
            """)
        
        with col2:
            st.markdown("""
            **🎯 Funcionalidades Essenciais:**
            - Estatísticas descritivas
            - Detecção de missing values
            - Correlações principais
            - Insights automáticos
            """)

if __name__ == "__main__":
    main()
