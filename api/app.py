from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS
import requests  # Para enviar alertas ao backend

app = Flask(__name__)
CORS(app)

# Carregar o modelo treinado
modelo = joblib.load("./models/model.pkl")

# URL do backend (ajuste conforme necessário)
BACKEND_URL = "http://localhost:8000/api/alertas"

@app.route("/")
def home():
    return jsonify({"message": "API de previsão de desistência funcionando!"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        dados = request.json
        
        # Verificar se todos os campos esperados estão presentes
        campos_esperados = ["faltas", "faltas_consecutivas", "media", "bolsa", "distancia"]
        if not all(key in dados for key in campos_esperados):
            return jsonify({"error": "Dados incompletos"}), 400

        # Converter os valores para float e criar o array de entrada
        entrada = np.array([[float(dados["faltas"]), 
                             float(dados["faltas_consecutivas"]), 
                             float(dados["media"]), 
                             float(dados["bolsa"]), 
                             float(dados["distancia"])]])
        
        # Fazer a previsão da probabilidade de desistência (0 = não desiste, 1 = desiste)
        probabilidade = modelo.predict_proba(entrada)[0][1]  # Pegamos a probabilidade de desistência (classe 1)
        probabilidade_percentual = round(probabilidade * 100, 2)  # Converter para percentual
        
        # Classificar nível de risco
        if probabilidade_percentual <= 30:
            nivel = "baixo"
        elif probabilidade_percentual <= 70:
            nivel = "médio"
            enviar_alerta(BACKEND_URL, dados, probabilidade_percentual, nivel)  # Alerta simples
        else:
            nivel = "alto"
            enviar_alerta(BACKEND_URL, dados, probabilidade_percentual, nivel, grave=True)  # Alerta sério
        
        print(f"Entrada recebida: {dados}")
        print(f"Probabilidade de desistência: {probabilidade_percentual}%")
        print(f"Nível de risco: {nivel}")

        return jsonify({
            "probabilidade_desistencia": probabilidade_percentual,
            "nivel_risco": nivel
        })

    except Exception as e:
        print(f"Erro na previsão: {str(e)}")
        return jsonify({"error": "Erro interno no servidor"}), 500


def enviar_alerta(url, dados, probabilidade, nivel, grave=False):
    """ Envia um alerta ao backend caso o nível de risco seja médio ou alto. """
    alerta = {
        "mensagem": f"Aluno com risco {nivel} de desistência ({probabilidade}%).",
        "detalhes": dados,
        "nivel": nivel,
        "grave": grave
    }
    try:
        resposta = requests.post(url, json=alerta)
        print(f"🔔 Alerta enviado: {resposta.status_code}")
    except Exception as e:
        print(f"⚠️ Erro ao enviar alerta ao backend: {str(e)}")


if __name__ == "__main__":
    app.run(debug=True, port=5000)
