import streamlit as st
import pandas as pd

st.title('Análise de Fraudes em Cartões de Crédito')

# Carregar os dados
@st.cache_data
def load_data():
    data = pd.read_csv('creditcard.csv')
    return data

data = load_data()

st.write('Visualização inicial dos dados:')
st.write(data.head())



# Análise Descritiva
st.subheader("Análise Descritiva dos Dados")
st.write(data.describe())

# Tipos de Dados
st.subheader("Tipos de Dados")
st.write(data.dtypes.astype(str))




# Distribuição das Variáveis
st.subheader("Distribuição das Variáveis (Histogramas)")

import matplotlib.pyplot as plt
import os

# Criar diretório para salvar os gráficos
if not os.path.exists('histograms'):
    os.makedirs('histograms')

# Estilo dos gráficos com bom contraste
plt.style.use('seaborn-v0_8-darkgrid')

for column in data.columns:
    fig, ax = plt.subplots()
    data[column].hist(ax=ax, bins=50)
    ax.set_title(f'Distribuição de {column}', color='white')
    ax.tick_params(colors='white')
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#0E1117')
    plt.xlabel(column, color='white')
    plt.ylabel('Frequência', color='white')
    
    # Salvar o gráfico
    file_path = f'histograms/{column}_histogram.png'
    plt.savefig(file_path)
    plt.close(fig)
    
    # Exibir o gráfico no Streamlit
    st.image(file_path)




# Medidas de Tendência Central e Variabilidade
st.subheader("Medidas de Tendência Central e Variabilidade")

central_tendency = pd.DataFrame({
    'Média': data.mean(),
    'Mediana': data.median(),
    'Desvio Padrão': data.std(),
    'Variância': data.var()
})
st.write(central_tendency)




# Análise de Tendências Temporais
st.subheader("Tendências Temporais das Transações")

fig, ax = plt.subplots(figsize=(12, 6))
data["Time"].plot(ax=ax, title="Transações ao Longo do Tempo", color="cyan")
ax.set_title("Transações ao Longo do Tempo", color="white")
ax.tick_params(colors="white")
fig.patch.set_facecolor("#0E1117")
ax.set_facecolor("#0E1117")
plt.xlabel("Índice da Transação", color="white")
plt.ylabel("Tempo (segundos)", color="white")

# Salvar e exibir o gráfico
file_path = "temporal_trend.png"
plt.savefig(file_path)
plt.close(fig)
st.image(file_path)




# Consulta Interativa com OpenAI
st.subheader("Faça uma pergunta sobre os dados")

import openai

# Obter a chave da API do usuário
api_key = st.text_input("Insira sua chave da API da OpenAI:", type="password")

if api_key:
    openai.api_key = api_key
    user_question = st.text_input("Sua pergunta:")

    if user_question:
        # Criar o prompt para a API
        prompt = f"""
        Com base na seguinte análise de dados de um arquivo CSV sobre fraudes de cartão de crédito, responda à pergunta do usuário.

        **Análise Descritiva:**
        {data.describe().to_string()}

        **Tipos de Dados:**
        {data.dtypes.to_string()}

        **Pergunta do Usuário:** {user_question}
        """

        try:
            # Chamar a API da OpenAI
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=prompt,
                max_tokens=500,
                n=1,
                stop=None,
                temperature=0.7,
            )

            # Exibir a resposta
            st.write("**Resposta:**")
            st.write(response.choices[0].text.strip())
        except Exception as e:
            st.error(f"Ocorreu um erro ao consultar a API da OpenAI: {e}")

