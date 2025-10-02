# Análise Exploratória de Dados com IA

Aplicação Streamlit para análise completa de arquivos CSV com chat inteligente.

## Funcionalidades

- **Distribuições**: Histogramas para todas as variáveis numéricas e gráficos de barras para categóricas
- **Correlações**: Matriz de correlação completa e identificação de relações significativas  
- **Tendências**: Linhas de tendência automáticas e análise temporal
- **Anomalias**: Detecção de outliers com métodos IQR e Z-score, boxplots para todas as variáveis
- **Chat IA**: Conversação com memória sobre os dados carregados

## APIs Suportadas

- **OpenAI**: GPT-3.5, GPT-4
- **Groq**: Llama 3.3 70B, Llama 3.1 8B  
- **Gemini**: Gemini Pro

## Como Usar

1. **Instalar dependências:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Executar aplicação:**
   ```bash
   streamlit run app.py
   ```

3. **Usar a aplicação:**
   - Carregue arquivo CSV
   - Explore as análises automáticas
   - Configure API de IA e faça perguntas sobre os dados

## Deploy no Streamlit Cloud

1. Faça upload dos 3 arquivos para repositório GitHub
2. Conecte repositório no [share.streamlit.io](https://share.streamlit.io)
3. Deploy automático

## Chaves das APIs

- **OpenAI**: [platform.openai.com](https://platform.openai.com) → API Keys
- **Groq**: [console.groq.com](https://console.groq.com) → API Keys  
- **Gemini**: [makersuite.google.com](https://makersuite.google.com) → Get API Key
