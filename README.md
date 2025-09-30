# 🤖 Agente de Análise Exploratória de Dados (E.D.A.)

## 📋 Descrição
Esta aplicação Streamlit é um **agente inteligente** que permite análise exploratória completa de **qualquer arquivo CSV** de forma automática e interativa. A ferramenta foi desenvolvida para atender aos requisitos da atividade obrigatória do Institut d'Intelligence Artificielle Appliquée.

## 🚀 Funcionalidades Principais

### 📋 Visão Geral
- **Informações básicas** do dataset (linhas, colunas, tamanho)
- **Tipos de dados** e identificação automática
- **Estatísticas descritivas** completas
- **Detecção de valores nulos** e únicos

### 📊 Distribuições
- **Histogramas automáticos** para variáveis numéricas
- **Gráficos de barras** para variáveis categóricas
- **Visualizações com alto contraste** para excelente legibilidade
- **Filtragem automática de outliers** para melhor visualização

### 🔍 Correlações
- **Matriz de correlação** interativa com heatmap
- **Identificação automática** de correlações significativas
- **Classificação por força** da correlação (forte, moderada, fraca)
- **Análise de dependências** entre variáveis

### 📈 Tendências
- **Detecção automática** de colunas temporais
- **Análise de tendências temporais** interativa
- **Padrões em variáveis categóricas**
- **Valores mais e menos frequentes**

### ⚠️ Anomalias
- **Detecção automática de outliers** usando método IQR
- **Visualização com boxplots** de alta qualidade
- **Estatísticas detalhadas** de anomalias por variável
- **Percentuais e limites** claramente definidos

### 🤖 Consulta Inteligente com IA
- **Integração com OpenAI GPT-3.5**
- **Consultas personalizadas** sobre os dados
- **Contexto automático** com estatísticas do dataset
- **Eficiência de custos** - API chamada apenas quando solicitado

## 🛠️ Como Executar Localmente

### Pré-requisitos
- Python 3.7+
- pip (gerenciador de pacotes Python)

### Instalação
1. Clone ou baixe este repositório
2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
3. Execute a aplicação:
   ```bash
   streamlit run app.py
   ```
4. Acesse no navegador: `http://localhost:8501`

## ☁️ Deploy no Streamlit Cloud

### Passo a Passo
1. **Fork este repositório** no GitHub
2. **Acesse** [share.streamlit.io](https://share.streamlit.io)
3. **Conecte sua conta** do Streamlit Cloud ao GitHub
4. **Selecione este repositório** para deploy
5. **Configure** o arquivo principal como `app.py`
6. **Deploy automático** será realizado

### URL de Acesso
Após o deploy, sua aplicação estará disponível em:
`https://[nome-do-app]-[seu-usuario].streamlit.app`

## 🔑 Uso da API OpenAI

### Como Obter sua Chave
1. Acesse [platform.openai.com](https://platform.openai.com)
2. Faça login em sua conta OpenAI
3. Navegue até **API Keys**
4. **Crie uma nova chave** secreta
5. **Cole a chave** na interface da aplicação

### Características de Eficiência
- ✅ **Consultas sob demanda** - API chamada apenas quando solicitado
- ✅ **Contexto otimizado** - Envia apenas estatísticas relevantes
- ✅ **Controle de custos** - Usuário insere sua própria chave
- ✅ **Sem armazenamento** - Chave não é salva ou compartilhada

## 📁 Estrutura dos Arquivos

```
streamlit_app/
├── app.py              # Aplicação principal Streamlit
├── requirements.txt    # Dependências Python
└── README.md          # Este arquivo
```

## 🎯 Casos de Uso

### Para Cientistas de Dados
- **Análise exploratória rápida** de novos datasets
- **Identificação automática** de padrões e anomalias
- **Geração de insights** com IA para relatórios

### Para Analistas de Negócios
- **Compreensão intuitiva** de dados complexos
- **Visualizações profissionais** prontas para apresentação
- **Perguntas em linguagem natural** sobre os dados

### Para Estudantes
- **Aprendizado prático** de análise de dados
- **Exemplos visuais** de conceitos estatísticos
- **Ferramenta educacional** interativa

## 🔧 Tecnologias Utilizadas

- **Streamlit** - Framework web para Python
- **Pandas** - Manipulação e análise de dados
- **Matplotlib** - Visualizações estáticas
- **Seaborn** - Visualizações estatísticas avançadas
- **NumPy** - Computação numérica
- **OpenAI** - Integração com GPT para consultas inteligentes

## 📊 Exemplo de Análise

A aplicação é capaz de analisar qualquer tipo de dataset CSV:
- **Dados financeiros** (transações, investimentos)
- **Dados de vendas** (produtos, clientes, receita)
- **Dados científicos** (experimentos, medições)
- **Dados de marketing** (campanhas, conversões)
- **Dados de RH** (funcionários, performance)

## 🎨 Design e Usabilidade

### Alto Contraste
- **Fundo escuro** (#0E1117) para reduzir fadiga visual
- **Texto branco** para máxima legibilidade
- **Cores vibrantes** (cyan, coral) para destacar dados
- **Grid sutil** para orientação visual

### Interface Intuitiva
- **Abas organizadas** por tipo de análise
- **Upload simples** de arquivos CSV
- **Feedback visual** em tempo real
- **Responsivo** para diferentes dispositivos

## 🏆 Diferenciais

1. **Genérico** - Funciona com qualquer CSV
2. **Automático** - Análise sem configuração manual
3. **Inteligente** - Integração com IA para insights
4. **Eficiente** - Otimizado para performance e custos
5. **Profissional** - Visualizações de alta qualidade
6. **Educativo** - Explica conceitos e resultados

---

**Desenvolvido com ❤️ para análise inteligente de dados**
