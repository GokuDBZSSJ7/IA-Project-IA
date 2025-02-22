from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

modelo = joblib.load("./models/model.pkl")

@app.route("/")
def home():
    return jsonify({"message": "API de previsão de desistência funcionando!"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        dados = request.json
        
        campos_esperados = ["faltas", "faltas_consecutivas", "media", "atrasos", "bolsa", "distancia"]
        if not all(key in dados for key in campos_esperados):
            return jsonify({"error": "Dados incompletos"}), 400

        entrada = np.array([[dados["faltas"], 
                             dados["faltas_consecutivas"], 
                             dados["media"], 
                             dados["atrasos"], 
                             dados["bolsa"], 
                             dados["distancia"]]])

        resultado = modelo.predict(entrada)

        print(f"Entrada recebida: {dados}")
        print(f"Previsão feita: {resultado[0]}")

        return jsonify({"desistencia": bool(resultado[0])})

    except Exception as e:
        print(f"Erro na previsão: {str(e)}")
        return jsonify({"error": "Erro interno no servidor"}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
