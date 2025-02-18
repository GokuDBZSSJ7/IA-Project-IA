import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

np.random.seed(42)
n_samples = 500

frequencia = np.random.randint(50, 100, n_samples)
performace = np.random.uniform(5, 10, n_samples)
mensalidade = np.random.uniform(500, 3000, n_samples)
renda_familiar = np.random.uniform(2000, 10000, n_samples)
atraso_pagamento = np.random.randint(0, 5, n_samples)
possui_bolsa = np.random.choice([0, 1], n_samples)
distancia = np.random.uniform(1, 50, n_samples)

dropout = (
    (frequencia < 70) * 3 + 
    (performace < 6) * 2 + 
    (mensalidade > 2000) * 4 + 
    (renda_familiar < 5000) * 6 + 
    (atraso_pagamento > 2) * 8 + 
    (possui_bolsa == 0) * 2 +
    (distancia > 30) * 1
)

dropout = (dropout > np.percentile(dropout, 70)).astype(int)

df = pd.DataFrame({
    "frequencia": frequencia,
    "performace": performace,
    "mensalidade": mensalidade,
    "renda_familiar": renda_familiar,
    "atraso_pagamento": atraso_pagamento,
    "possui_bolsa": possui_bolsa,
    "distancia": distancia,
    "dropout": dropout
})

X = df.drop(columns=["dropout"])
y = df["dropout"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia do modelo: {accuracy * 100:.2f}%")

print("Relatório de Classificação:")
print(classification_report(y_test, y_pred))

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Não Evasão", "Evasão"], yticklabels=["Não Evasão", "Evasão"])
plt.xlabel("Predito")
plt.ylabel("Real")
plt.title("Matriz de Confusão")
plt.show()

importances = model.feature_importances_
feature_names = X.columns
for feature, importance in zip(feature_names, importances):
    print(f"{feature}: {importance:.4f}")

joblib.dump(model, "model.pkl")
print("Modelo salvo como model.pkl")
