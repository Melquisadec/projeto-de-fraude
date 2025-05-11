import pandas as pd 
import pickle
from sklearn.linear_model import LogisticRegression

data = {
    'Valor_transacao':          [15,200,3500,60,500,9000,30,850,4999],
    'tempo_conta':              [2,24,1,12,36,0,6,48,3],
    'num_trans_ultimo_30d':     [1,5,0,3,10,20,2,15,4],
    'pais_origem':              [0,0,2,2,1,1,0,1,2],
    'fraude':                   [0,0,1,0,0,1,0,0,1]
}

df = pd.DataFrame(data)
X = df[['Valor_transacao','tempo_conta','num_trans_ultimo_30d','pais_origem']]
y = df['fraude']

## treinar o modelo 
model = LogisticRegression()
model.fit(X,y)

with open('modelo_fraude.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Modelo de fraude treinado e salvo em 'modelo_fraude.pkl'")