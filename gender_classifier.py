import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt

# Configurações do caminho
data_dir = 'cuhk-face-sketch-database-cufs/photos'  # Insira o caminho correto do dataset

# Função para carregar e pré-processar as imagens
def load_images_labels(data_dir):
    images = []
    labels = []
    for img_name in os.listdir(data_dir):
        img_path = os.path.join(data_dir, img_name)
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(250, 200))
        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0  # Normalização
        images.append(img_array)
        
        # Atribuindo rótulos: 0 para masculino, 1 para feminino
        if 'm' in img_name:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(images), np.array(labels)

# Carregar os dados
images, labels = load_images_labels(data_dir)

# Divisão do dataset em 50% treino, 30% validação, 20% teste (com seed fixa)
X_temp, X_test, y_temp, y_test = train_test_split(images, labels, test_size=0.2, random_state=23)  # 20% teste
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.375, random_state=23)  # 30% val, 50% treino

# Verificando as proporções de treino, validação e teste
print(f'Tamanho do conjunto de treino: {len(X_train)}')
print(f'Tamanho do conjunto de validação: {len(X_val)}')
print(f'Tamanho do conjunto de teste: {len(X_test)}')

# Gerador de imagens (data augmentation opcional)
datagen = ImageDataGenerator(rotation_range=10, zoom_range=0.1, horizontal_flip=True)

# Definição da CNN
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(250, 200, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),  # Regularização para evitar overfitting
    layers.Dense(1, activation='sigmoid')  # Saída binária (0 ou 1)
])

# Compilação do modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Treinamento
history = model.fit(datagen.flow(X_train, y_train, batch_size=64), 
                    validation_data=(X_val, y_val), epochs=20)


# Avaliação do modelo
y_pred = model.predict(X_test)
y_pred_binary = (y_pred > 0.5).astype(int)

print(classification_report(y_test, y_pred_binary))

# AUC-ROC
roc_score = roc_auc_score(y_test, y_pred)
print(f"AUC-ROC: {roc_score}")

# Gráfico de perda e acurácia
plt.plot(history.history['accuracy'], label='Acurácia Treinamento')
plt.plot(history.history['val_accuracy'], label='Acurácia Validação')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Perda Treinamento')
plt.plot(history.history['val_loss'], label='Perda Validação')
plt.xlabel('Épocas')
plt.ylabel('Perda')
plt.legend()
plt.show()