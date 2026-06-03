# Guarda este archivo como: asistente_rnn/api/app.py
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify

app = Flask(__name__)

# Definición de rutas base de recursos
ROOT = Path(__file__).resolve().parent.parent / "modelo_guardado"
model = None
stoi = {}
itos = {}
BLOCK_SIZE = 64

def load_model_and_meta():
    """Carga los artefactos de la red neuronal y configura los mapeos de tokens"""
    global model, stoi, itos, BLOCK_SIZE
    meta_path = ROOT / "meta.json"
    model_path = ROOT / "asistente_model.keras"
    
    if not meta_path.is_file() or not model_path.is_file():
        raise FileNotFoundError("No se encontraron los archivos del modelo. Corre primero modelo/entrenar.py")
        
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    BLOCK_SIZE = int(meta["block_size"])
    chars = meta["chars"]
    
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    model = tf.keras.models.load_model(model_path)
    print(f"-> Servidor Flask: Modelo RNN Vanilla ({BLOCK_SIZE} context) cargado correctamente.")

def predict_completion(prefix, max_new=40, temperature=0.4):
    """Ejecuta el bucle autoregresivo carácter por carácter de forma optimizada"""
    # Mapeo seguro: si el caracter no está en el vocabulario, usa el espacio como fallback
    ids = [stoi.get(c, stoi.get(' ', 0)) for c in prefix]
    rng = np.random.default_rng()
    
    # Pre-reservamos el array de entrada para no reconstruirlo en cada ciclo for
    input_buffer = np.zeros((1, BLOCK_SIZE), dtype=np.int64)
    
    for _ in range(max_new):
        x = ids[-BLOCK_SIZE:]
        
        # Relleno manual rápido si el contexto es corto
        if len(x) < BLOCK_SIZE:
            pad_val = ids[0] if len(ids) > 0 else 0
            x = [pad_val] * (BLOCK_SIZE - len(x)) + x
            
        input_buffer[0, :] = x
        
        # ¡OPTIMIZACIÓN CLAVE!: Usar predict_on_batch evita la sobrecarga del grafo de TF
        # que provocaba el Request Timeout en CPUs al llamarlo secuencialmente.
        preds = model.predict_on_batch(input_buffer)
        logits = preds[0, -1, :]
        
        # Aplicación de temperatura balanceada para código estructurado
        logits = logits / max(temperature, 1e-6)
        logits = logits - logits.max()  # Estabilidad numérica
        
        probs = np.exp(logits) / np.sum(np.exp(logits))
        next_id = int(rng.choice(len(probs), p=probs))
        ids.append(next_id)
        
    full_text = "".join([itos.get(i, ' ') for i in ids])
    return full_text[len(prefix):]

@app.route('/api/complete', methods=['POST'])
def complete_endpoint():
    """Endpoint para autocompletado directo de código en la posición del cursor"""
    data = request.get_json() or {}
    prefix = data.get("prefix", "")
    max_new = int(data.get("max_new", 30)) # 
    temperature = float(data.get("temperature", 0.4)) 
    
    if not prefix:
        return jsonify({"ok": False, "error": "El campo 'prefix' es obligatorio."}), 400
        
    try:
        suffix = predict_completion(prefix, max_new, temperature)
        return jsonify({"ok": True, "suffix": suffix})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route('/api/suggest', methods=['POST'])
def suggest_endpoint():
    """Endpoint para obtener una lista de predicciones en el panel QuickPick del editor"""
    data = request.get_json() or {}
    prefix = data.get("prefix", "")
    n = int(data.get("n", 3))
    
    if not prefix:
        return jsonify({"ok": False, "error": "El campo 'prefix' es obligatorio."}), 400
        
    try:
        suggestions = []
        seen = set()
        # Generar muestras optimizadas
        for i in range(n * 2):
            suffix = predict_completion(prefix, max_new=20, temperature=0.3 + (i * 0.1))
            first_line = suffix.split("\n")[0]
            candidate = prefix + first_line
            if candidate not in seen and len(first_line.strip()) > 0:
                seen.add(candidate)
                suggestions.append(candidate)
            if len(suggestions) >= n:
                break
        return jsonify({"ok": True, "suggestions": suggestions})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

if __name__ == '__main__':
    load_model_and_meta()
    # Ejecución local en localhost
    app.run(host='127.0.0.1', port=5000, debug=False)