# 🤖 Agente de Análise Exploratória de Dados - VERSÃO MELHORADA

## 🎯 Principais Melhorias Implementadas

Esta versão resolve **todos os problemas identificados** na versão anterior e adiciona funcionalidades avançadas:

### ✅ **Problemas Corrigidos:**

1. **🧠 Memória Conversacional Implementada**
   - IA lembra de toda a conversa anterior
   - Contexto mantido entre perguntas
   - Histórico visual da conversa

2. **📊 Distribuições Completas**
   - Gráficos para **TODAS** as variáveis numéricas
   - Gráficos para **TODAS** as variáveis categóricas
   - Curvas de densidade adicionadas
   - Estatísticas detalhadas em cada gráfico

3. **📈 Análise de Tendências Avançada**
   - **Linhas de tendência automáticas**
   - **Médias móveis** calculadas
   - **Coeficientes de inclinação** mostrados
   - Correlações visuais entre variáveis

4. **⚠️ Outliers Completos**
   - **Todos os gráficos** de outliers gerados
   - Métodos **IQR e Z-score**
   - **Scatter plots multivariados**
   - Estatísticas detalhadas por variável

5. **🤖 IA com Acesso Total aos Dados**
   - IA tem acesso **completo** ao dataset
   - Pode responder **qualquer pergunta** sobre os dados
   - Contexto **rico e detalhado** enviado para IA
   - **Múltiplas APIs**: OpenAI, Groq, Gemini

## 🚀 Funcionalidades Principais

### 🧠 **Chat Inteligente com Memória**
- **Conversação contínua** que mantém contexto
- **Acesso completo** aos dados carregados
- **Histórico visual** das últimas 10 mensagens
- **Múltiplas APIs** suportadas:
  - 🔵 **OpenAI**: GPT-3.5, GPT-4, GPT-4 Turbo
  - 🟢 **Groq**: Llama 3.3 70B, Llama 3.1 8B, GPT OSS
  - 🔴 **Gemini**: Gemini Pro, Gemini Pro Vision

### 📊 **Visualizações Completas e Avançadas**
- **Distribuições numéricas**: Histogramas + curvas de densidade
- **Distribuições categóricas**: Gráficos de barras com valores
- **Correlações**: Heatmaps triangulares + scatter plots
- **Tendências**: Linhas de tendência + médias móveis
- **Outliers**: Boxplots + scatter plots multivariados

### 📈 **Análise de Tendências Melhorada**
- **Detecção automática** de colunas temporais
- **Linhas de regressão** com coeficientes
- **Médias móveis** adaptativas
- **Correlações visuais** entre variáveis mais correlacionadas

### ⚠️ **Detecção Avançada de Outliers**
- **Método IQR** (Interquartile Range)
- **Método Z-score** (> 3 desvios padrão)
- **Visualizações completas** para todas as variáveis
- **Análise multivariada** com scatter plots
- **Estatísticas detalhadas** por método

## 🛠️ Como Executar

### Pré-requisitos
```bash
Python 3.8+
pip install -r requirements_melhorado.txt
```

### Execução
```bash
streamlit run app_melhorado.py
```

### Dependências
- **streamlit** - Interface web
- **pandas** - Manipulação de dados
- **matplotlib** - Visualizações básicas
- **seaborn** - Visualizações estatísticas
- **numpy** - Computação numérica
- **scipy** - Estatísticas avançadas
- **scikit-learn** - Preprocessamento
- **groq** - API Groq
- **openai** - API OpenAI
- **google-generativeai** - API Gemini

## 🎯 Como Usar o Chat com IA

### 1. **Configuração**
- Escolha a API (OpenAI, Groq ou Gemini)
- Insira sua chave da API
- Selecione o modelo desejado

### 2. **Tipos de Perguntas que a IA Pode Responder**

#### 📊 **Sobre Estatísticas**
- "Quais são as principais estatísticas descritivas?"
- "Qual variável tem maior variabilidade?"
- "Como interpretar a mediana vs média?"

#### 🔍 **Sobre Correlações**
- "Quais variáveis estão mais correlacionadas?"
- "Existe multicolinearidade nos dados?"
- "Como interpretar a correlação de 0.85?"

#### 📈 **Sobre Tendências**
- "Existe tendência temporal nos dados?"
- "Qual a inclinação da linha de tendência?"
- "Os dados mostram sazonalidade?"

#### ⚠️ **Sobre Outliers**
- "Quais outliers são mais preocupantes?"
- "Como tratar os valores extremos?"
- "Os outliers indicam erro ou padrão real?"

#### 🎯 **Insights Gerais**
- "Quais são os principais insights?"
- "Que análises adicionais recomendar?"
- "Como melhorar a qualidade dos dados?"

### 3. **Memória Conversacional**
- A IA lembra de **toda a conversa**
- Pode fazer **perguntas de seguimento**
- **Contexto mantido** entre perguntas
- **Referências** a respostas anteriores

## 🔑 Como Obter Chaves das APIs

### 🔵 **OpenAI**
1. Acesse [platform.openai.com](https://platform.openai.com)
2. Faça login → **API Keys**
3. **Create new secret key**
4. Cole na aplicação

### 🟢 **Groq** (Recomendado - Mais Rápido)
1. Acesse [console.groq.com](https://console.groq.com)
2. Crie conta **gratuita**
3. **API Keys** → **Create API Key**
4. Cole na aplicação

### 🔴 **Gemini** (Google)
1. Acesse [makersuite.google.com](https://makersuite.google.com)
2. **Get API Key**
3. **Create API Key**
4. Cole na aplicação

## 📊 Comparação de APIs

| Característica | OpenAI | Groq | Gemini |
|---|---|---|---|
| **Velocidade** | Média | ⚡ Muito Rápida | Rápida |
| **Custo** | Médio | 💰 Baixo | Baixo |
| **Tier Gratuito** | Limitado | 🎁 Generoso | Generoso |
| **Qualidade** | ⭐ Excelente | ⭐ Excelente | ⭐ Muito Boa |
| **Modelos** | GPT-3.5/4 | Llama 3.3 | Gemini Pro |

## 🎨 Interface e Design

### **Alto Contraste**
- Fundo escuro (#0E1117)
- Texto branco para legibilidade
- Cores vibrantes (cyan, coral, orange)
- Grid sutil para orientação

### **Organização em Abas**
- 📋 **Visão Geral**: Informações básicas
- 📊 **Distribuições**: Todas as visualizações
- 🔍 **Correlações**: Análise de relacionamentos
- 📈 **Tendências**: Análise temporal
- ⚠️ **Anomalias**: Detecção de outliers
- 🤖 **Chat com IA**: Conversação inteligente

## 🏆 Diferenciais da Versão Melhorada

1. **🧠 IA Contextual**: Acesso completo aos dados
2. **💬 Memória Persistente**: Conversa contínua
3. **📊 Visualizações Completas**: Todos os gráficos
4. **📈 Tendências Avançadas**: Linhas de regressão
5. **⚠️ Outliers Completos**: Análise multivariada
6. **🔧 Múltiplas APIs**: OpenAI, Groq, Gemini
7. **🎯 Respostas Precisas**: Baseadas nos dados reais

## 🚀 Casos de Uso Avançados

### **Para Cientistas de Dados**
- Análise exploratória **completa e automatizada**
- **Insights baseados em dados reais**
- **Recomendações** de análises adicionais

### **Para Analistas de Negócios**
- **Interpretação inteligente** de padrões
- **Explicações em linguagem natural**
- **Identificação de oportunidades**

### **Para Estudantes**
- **Aprendizado interativo** de estatística
- **Explicações pedagógicas** de conceitos
- **Exemplos práticos** com dados reais

---

**🎯 Esta versão resolve TODOS os problemas identificados e oferece uma experiência completa de análise de dados com IA!**
