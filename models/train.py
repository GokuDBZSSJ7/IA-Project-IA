import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv('./data/dataset.csv')

# Ajustando os nomes das colunas para corresponderem ao dataset
X = df[['faltas', 'faltas_consecutivas', 'media', 'atrasos', 'bolsa', 'distancia']]
y = df['desistencia']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Acurácia do modelo: {accuracy:.2f}")
print("Relatório de Classificação:")
print(report)

with open('./models/model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Modelo treinado e salvo!")
