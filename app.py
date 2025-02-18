from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        formatted_data = {
            "frequencia": float(data["frequencia"]), 
            "performace": float(data["performace"]),
            "mensalidade": float(data["mensalidade"]),
            "renda_familiar": float(data["renda_familiar"]),
            "atraso_pagamento": int(data["atraso_pagamento"]),
            "possui_bolsa": int(data["possui_bolsa"]),
            "distancia": float(data["distancia"])
        }

        df = pd.DataFrame([formatted_data])

        expected_cols = ["frequencia", "performace", "mensalidade", "renda_familiar", "atraso_pagamento", "possui_bolsa", "distancia"]
        df = df[expected_cols]

        probabilidade = model.predict_proba(df)[:, 1][0]

        if probabilidade < 0.4:
            status = "Baixo"
        elif probabilidade < 0.7:
            status = "Médio"
        else:
            status = "Alto"
        
        return jsonify({"probabilidade": round(probabilidade * 100, 2), "status": status})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
