import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from google.colab import drive
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import lightgbm as lgb
from sklearn.svm import SVC  # SVM modelini import et
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
import os

base_dir = r"C:\Users\Naz\Desktop\split-garbage-dataset"
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

print(f"Train Directory Exists: {os.path.exists(train_dir)}")
print(f"Test Directory Exists: {os.path.exists(test_dir)}")



# Aşama 4: Görüntü Boyutu ve Hiperparametreleri Tanımlama
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32

# Aşama 5: Veri Artırma ve Ön İşleme
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Aşama 6: Veri Setlerini Yükleme
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)


# MobileNet Tabanını Al
mobilenet_base = MobileNet(
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
    include_top=False,
    weights='imagenet'
)

# MobileNet Ağırlıklarını Dondur
mobilenet_base.trainable = False

# MobileNet'in Çıkışını Al
mobilenet_output = mobilenet_base.output

# Özel CNN Katmanlarını Ekleyelim
x = Conv2D(64, (3, 3), activation='relu')(mobilenet_output)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)

# Çıkış Katmanı
output_layer = Dense(len(train_generator.class_indices), activation='softmax')(x)

# Modeli Oluştur
combined_model = Model(inputs=mobilenet_base.input, outputs=output_layer)

# Modeli Derleme
combined_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Modeli Eğitme
history = combined_model.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator
)

# Modeli Değerlendirme
loss, accuracy = combined_model.evaluate(test_generator)
print(f"Test Doğruluğu: {accuracy * 100:.2f}%")

# Eğitim Sonuçlarını Görselleştirme
def plot_history(history):
    plt.figure(figsize=(12, 4))

    # Doğruluk
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
    plt.plot(history.history['val_accuracy'], label='Test Doğruluğu')
    plt.xlabel('Epoch')
    plt.ylabel('Doğruluk')
    plt.legend()
    plt.title('Model Doğruluğu')

    # Kayıp
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Eğitim Kaybı')
    plt.plot(history.history['val_loss'], label='Test Kaybı')
    plt.xlabel('Epoch')
    plt.ylabel('Kayıp')
    plt.legend()
    plt.title('Model Kaybı')

    plt.show()

plot_history(history)

# Veri setinden etiketleri çekme
train_labels = train_generator.classes  # train_generator'dan etiketleri al
test_labels = test_generator.classes  # test_generator'dan etiketleri al

# Sınıf etiketlerini sayılara dönüştürmek için
label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)
test_labels_encoded = label_encoder.transform(test_labels)

# Özellikleri (features) almak için MobileNet'ten çıkış alıyoruz
train_features = mobilenet_base.predict(train_generator, verbose=1)
test_features = mobilenet_base.predict(test_generator, verbose=1)

# Train ve Test verilerini düzleştiriyoruz
train_features_flattened = train_features.reshape(train_features.shape[0], -1)
test_features_flattened = test_features.reshape(test_features.shape[0], -1)

# Random Forest Modeli
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(train_features_flattened, train_labels_encoded)

# Karar Ağaçları Modeli
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(train_features_flattened, train_labels_encoded)

# SVM Modeli
svm_model = SVC(random_state=42)
svm_model.fit(train_features_flattened, train_labels_encoded)

# Test Verisi Üzerinde Değerlendirme
rf_predictions = rf_model.predict(test_features_flattened)
dt_predictions = dt_model.predict(test_features_flattened)
svm_predictions = svm_model.predict(test_features_flattened)

# Performans metriklerini hesapla
rf_accuracy = accuracy_score(test_labels_encoded, rf_predictions)
rf_precision = precision_score(test_labels_encoded, rf_predictions, average='macro')
rf_recall = recall_score(test_labels_encoded, rf_predictions, average='macro')
rf_f1 = f1_score(test_labels_encoded, rf_predictions, average='macro')

dt_accuracy = accuracy_score(test_labels_encoded, dt_predictions)
dt_precision = precision_score(test_labels_encoded, dt_predictions, average='macro')
dt_recall = recall_score(test_labels_encoded, dt_predictions, average='macro')
dt_f1 = f1_score(test_labels_encoded, dt_predictions, average='macro')

svm_accuracy = accuracy_score(test_labels_encoded, svm_predictions)
svm_precision = precision_score(test_labels_encoded, svm_predictions, average='macro')
svm_recall = recall_score(test_labels_encoded, svm_predictions, average='macro')
svm_f1 = f1_score(test_labels_encoded, svm_predictions, average='macro')

# Random Forest ve Decision Tree Modeli Performansını Yazdırma
print("\nRandom Forest Model Performansı:")
print(f"Accuracy: {rf_accuracy * 100:.2f}%")
print(f"Precision: {rf_precision * 100:.2f}%")
print(f"Recall: {rf_recall * 100:.2f}%")
print(f"F1 Score: {rf_f1 * 100:.2f}%")

print("\nDecision Tree Model Performansı:")
print(f"Accuracy: {dt_accuracy * 100:.2f}%")
print(f"Precision: {dt_precision * 100:.2f}%")
print(f"Recall: {dt_recall * 100:.2f}%")
print(f"F1 Score: {dt_f1 * 100:.2f}%")

print("\nSVM Model Performansı:")
print(f"Accuracy: {svm_accuracy * 100:.2f}%")
print(f"Precision: {svm_precision * 100:.2f}%")
print(f"Recall: {svm_recall * 100:.2f}%")
print(f"F1 Score: {svm_f1 * 100:.2f}%")

# Predictions'ı sınıf adlarına çevir
rf_predictions = label_encoder.inverse_transform(rf_predictions)
dt_predictions = label_encoder.inverse_transform(dt_predictions)
svm_predictions = label_encoder.inverse_transform(svm_predictions)

# Classification Report Yazdırma
print("\nRandom Forest Model Classification Report:")
print(classification_report(test_labels, rf_predictions))

print("\nDecision Tree Model Classification Report:")
print(classification_report(test_labels, dt_predictions))

print("\nSVM Model Classification Report:")
print(classification_report(test_labels, svm_predictions))

# Fotoğraflardan Tahmin Yapma
from tensorflow.keras.preprocessing import image


# Örnek Resmin Yolunu Belirtin
img_path = r"C:\\Users\\Naz\\Desktop\\split-garbage-dataset\\test\\plastic\\plastic49.jpg"


# Görüntüyü Yükleyin ve Boyutlandırın
img = image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0
# Tahmin Yapma
predictions = combined_model.predict(img_array)
predicted_class_index = np.argmax(predictions)

# Sınıf İsimlerini Al
class_labels = {v: k for k, v in train_generator.class_indices.items()}
predicted_class_label = class_labels[predicted_class_index]

print(f"Tahmin Edilen Sınıf: {predicted_class_label}")

# Tahmin Sonucunu Görselleştirme
plt.imshow(img)
plt.title(f"Tahmin: {predicted_class_label}")
plt.axis('off')
plt.show()

# Performans Tablosu (Her Epoch için)
epochs = range(1, len(history.history['accuracy']) + 1)
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
train_loss = history.history['loss']
val_loss = history.history['val_loss']

performance_table = pd.DataFrame({
    'Epoch': epochs,
    'Train Accuracy': train_accuracy,
    'Validation Accuracy': val_accuracy,
    'Train Loss': train_loss,
    'Validation Loss': val_loss
})

print(performance_table)

#GRAFİK ÇIKTISI VEREN KISIM

import matplotlib.pyplot as plt

# Verileri tanımlayalım
materials = ['Plastik', 'Metal', 'Kağıt', 'Karton', 'Cam', 'Çöp']
carbon_footprint = [6, 5, 4, 3, 2, 1]

# Grafik oluşturma
plt.figure(figsize=(8, 5))
plt.bar(materials, carbon_footprint, color=['red', 'orange', 'yellow', 'green', 'blue', 'purple'])

# Başlık ve etiketler
plt.title('Malzemeler ve Karbon Ayak İzi Katkıları', fontsize=14, fontweight='bold')
plt.xlabel('Malzemeler', fontsize=12)
plt.ylabel('Karbon Ayak İzi Katkısı (Ölçek: 1-6)', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# Görseli kaydedip gösterme
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Performans metriklerini hesapla ve yazdır
def print_performance(model_name, predictions, test_labels_encoded):
    accuracy = accuracy_score(test_labels_encoded, predictions)
    precision = precision_score(test_labels_encoded, predictions, average='weighted', zero_division=1)
    recall = recall_score(test_labels_encoded, predictions, average='weighted', zero_division=1)
    f1 = f1_score(test_labels_encoded, predictions, average='weighted', zero_division=1)


    print(f"\n{model_name} Performansı:")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1 Score: {f1 * 100:.2f}%")

# Random Forest Model Performansı
print_performance("Random Forest", rf_predictions, test_labels_encoded)

# Decision Tree Model Performansı
print_performance("Decision Tree", dt_predictions, test_labels_encoded)

# SVM Model Performansı
print_performance("SVM", svm_predictions, test_labels_encoded)

# Ekstra: MobileNet Model Performansı (Eğitim)
mobilenet_predictions = combined_model.predict(test_generator, verbose=1)
mobilenet_predictions = np.argmax(mobilenet_predictions, axis=1)
mobilenet_accuracy = accuracy_score(test_labels_encoded, mobilenet_predictions)
mobilenet_precision = precision_score(test_labels_encoded, mobilenet_predictions, average='macro')
mobilenet_recall = recall_score(test_labels_encoded, mobilenet_predictions, average='macro')
mobilenet_f1 = f1_score(test_labels_encoded, mobilenet_predictions, average='macro')

print(f"\nMobileNet Model Performansı:")
print(f"Accuracy: {mobilenet_accuracy * 100:.2f}%")
print(f"Precision: {mobilenet_precision * 100:.2f}%")
print(f"Recall: {mobilenet_recall * 100:.2f}%")
print(f"F1 Score: {mobilenet_f1 * 100:.2f}%")


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Tahminlerinizi alın (örneğin, MobileNet modelinden)
mobilenet_predictions = combined_model.predict(test_generator, verbose=1)
mobilenet_predictions = np.argmax(mobilenet_predictions, axis=1)  # Kategorik tahminler için argmax

# Gerçek etiketler
true_labels = test_labels_encoded  # Test etiketleri

# RMSE, MSE, MAE ve R² hesaplamak için
mse = mean_squared_error(true_labels, mobilenet_predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(true_labels, mobilenet_predictions)
r2 = r2_score(true_labels, mobilenet_predictions)

# Sonuçları yazdır
print(f"Model Performansı (MobileNet):")
print(f"MSE (Mean Squared Error): {mse:.4f}")
print(f"RMSE (Root Mean Squared Error): {rmse:.4f}")
print(f"MAE (Mean Absolute Error): {mae:.4f}")
print(f"R² (R-kare): {r2:.4f}")


def plot_loss(history):
    plt.plot(history.history['loss'], label='Eğitim Kaybı')
    plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
    plt.title('Eğitim ve Doğrulama Kaybı')
    plt.xlabel('Epochs')
    plt.ylabel('Kayb')
    plt.legend()
    plt.show()

plot_loss(history)


# Plastik türlerine göre karbon ayak izi
materials = ['Plastik', 'Metal', 'Kağıt', 'Karton', 'Cam', 'Çöp']
carbon_footprint = [6, 5, 4, 3, 2, 1]

# Scatter plot oluşturma
plt.figure(figsize=(8, 5))
plt.scatter(materials, carbon_footprint, color='blue', s=100)

# Başlık ve etiketler
plt.title('Malzemelerin Karbon Ayak İzi', fontsize=14)
plt.xlabel('Malzeme Türü', fontsize=12)
plt.ylabel('Karbon Ayak İzi Katkısı (Ölçek: 1-6)', fontsize=12)

plt.grid(True)
plt.show()


from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Tahmin Edilen')
    plt.ylabel('Gerçek')
    plt.show()

# Confusion matrix için etiketler
labels = label_encoder.classes_
plot_confusion_matrix(test_labels_encoded, rf_predictions, labels)


def plot_accuracy_loss(history):
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Accuracy
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Doğruluk', color='tab:blue')
    ax1.plot(history.history['accuracy'], label='Eğitim Doğruluğu', color='tab:blue')
    ax1.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu', color='tab:cyan')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Kayıp (Loss)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Kayıp', color='tab:red')
    ax2.plot(history.history['loss'], label='Eğitim Kaybı', color='tab:red')
    ax2.plot(history.history['val_loss'], label='Doğrulama Kaybı', color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    fig.tight_layout()
    plt.title('Eğitim ve Doğrulama Doğruluğu ve Kaybı')
    plt.legend(loc='upper left')
    plt.show()

plot_accuracy_loss(history)


def plot_feature_importance(model, features):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(12, 6))
    plt.title('Özelliklerin Önem Dereceleri')
    plt.bar(range(len(importances)), importances[indices], align='center')

    # Özellik isimlerini döngü ile sıralayıp göstermek
    sorted_features = [features[i] for i in indices]
    plt.xticks(range(len(importances)), sorted_features, rotation=90)

    plt.xlabel('Özellikler')
    plt.ylabel('Özellik Önem Derecesi')
    plt.tight_layout()
    plt.show()

# Özelliklerin isimlerini al
feature_names = [f"Feature {i}" for i in range(train_features_flattened.shape[1])]
plot_feature_importance(rf_model, feature_names)