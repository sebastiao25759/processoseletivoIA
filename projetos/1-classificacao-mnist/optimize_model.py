import tensorflow as tf
import os

# ---------------------------------------------------------------------------
# Projeto 1 — Otimização do Modelo (MNIST)
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar o modelo treinado em "model.h5"
#   2. Converter para TensorFlow Lite usando tf.lite.TFLiteConverter
#   3. Aplicar uma técnica de otimização (ex: Dynamic Range Quantization,
#      via converter.optimizations = [tf.lite.Optimize.DEFAULT])
#   4. Salvar o resultado como "model.tflite"
# ---------------------------------------------------------------------------

# insira seu código aqui

# Carregando o modelo original

model = tf.keras.models.load_model("model.h5")

# Configurando a conversão com otimização (Dynamic Range Quantization)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

# Salvando o resultado como "model.tflite"
with open("model.tflite", "wb") as f:
    f.write(tflite_model)


# Tamanho dos modelos para colocar na documentação
tflite_model = os.path.getsize("model.tflite") / 1048576
model_size = os.path.getsize("model.h5") / 1048576

print(f"Tamanho do modelo original: {model_size:.2f} MB")
print(f"Tamanho do modelo otimizado: {tflite_model:.2f} MB")