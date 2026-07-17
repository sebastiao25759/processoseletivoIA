import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ---------------------------------------------------------------------------
# Projeto 1 — Classificação MNIST
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar o dataset MNIST via tf.keras.datasets.mnist
#   2. Normalizar as imagens para [0, 1] e ajustar o shape para (28, 28, 1)
#   3. Separar um conjunto de validação (ex: validation_split ou split manual)
#   4. Construir uma CNN com 3-4 blocos Conv2D + BatchNormalization + MaxPooling2D,
#      seguida de Dropout antes da camada de saída (10 classes, softmax)
#   5. Treinar com EarlyStopping monitorando a perda de validação
#   6. Exibir a acurácia de validação final no terminal
#   7. Salvar o modelo treinado como "model.h5"
# ---------------------------------------------------------------------------

# insira seu código aqui

# Carregamento e pre-processamento do dataset MNIST

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

# Definição do modelo

model = keras.Sequential([
    keras.Input(shape=(28,28,1)),

    layers.Conv2D(32,3,activation='relu',padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.Conv2D(64,3,activation='relu',padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.Conv2D(128,3,activation='relu',padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128,activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10,activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Treinamento

early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

with tf.device('/CPU:0'):
    history = model.fit(
        x_train,
        y_train,
        epochs=10,
        batch_size=64,
        validation_split=0.2,
        callbacks=[early_stop]
)

# Resultado e Exportação

val_accuracy = history.history['val_accuracy'][-1]
val_loss, val_accuracy = model.evaluate(
    x_train[-12000:],
    y_train[-12000:],
    verbose=0
)

print(f"Acurácia de validação final: {val_accuracy:.2f}")

model.save("model.h5")