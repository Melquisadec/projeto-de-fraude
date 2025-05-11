import streamlit as st
import pickle
import numpy as np

# Carregar o modelo de fraude treinado
with open('modelo_fraude.pkl', 'rb')as file:
    fraud_model = pickle.load(file)

st.title("Detecação de fraude em transação")
st.write("Este aplicativo utiliza um modelo de regressão logistica para prever se uma transação é fraudulenta")

## captura informações do usuario 
st.header("Detalhes da transação")

valor_transacao = st.number_input("valor da transação (em dólares)", min_value = 0, max_value = 10000, value = 50)
tempo_conta = st.number_input("Tempo de conta (em meses)", min_value = 0, max_value = 120, value = 3)
num_trans_ultimo_30d = st.number_input("Número de Transações nos Últimos 30 dias", min_value = 0, max_value = 1000, value = 3)

# para simplificar, usamos um selectbox para país de origem 

pais_origem_opcoes = {
    "Brasil": 0,
    "EUA": 1,
    "Outros": 2
}
pais_origem_escolhido = st.selectbox("País de Origem", list(pais_origem_opcoes.keys()))
pais_origem = pais_origem_opcoes[pais_origem_escolhido]

## botao de predicao
if st.button("verificar se é Fraude"):
    #construir um array adequado para o modelo.
    input_array = np.array([[valor_transacao, tempo_conta, num_trans_ultimo_30d, pais_origem]])

    #executa a predição
    pred = fraud_model.predict(input_array)
    proba = fraud_model.predict_proba(input_array)

    st.write("##Resultado da análise: ")
    if pred[0] == 1:
        st.error("Alerta: Transação com ALTO risco de fraude.")
    else:
        st.success("Transação aparentemente legítima.")

        # prob de fraude opcional 

    st.write(f"**Probabilidade de não fraude** {proba[0][0]:.2f}")
    st.write(f"**Probabilidade de fraude**{proba[0][1]:.2f}")
else:
    st.write("Informe os detalhes e clique em 'verificar se é fraude'.")