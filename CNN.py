import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
else:
    print("No GPU devices found")

# Path dataset
data_path = os.path.dirname(os.path.abspath(__file__))

# Membuat label untuk kelas
class_names = ['Normal', 'Pneumonia']

# Memuat dataset
def load_data(data_directory, class_name):
    data = []
    labels = []
    path = os.path.join(data_directory, class_name)
    class_num = class_names.index(class_name)
    for img in os.listdir(path):
        try:
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            img_resized = cv2.resize(img_array, (150, 150))
            data.append(img_resized)
            labels.append(class_num)
        except Exception as e:
            pass
    return np.array(data), np.array(labels)

# Memuat data train dan data test
train_data_normal, train_labels_normal = load_data(os.path.join(data_path, 'train'), 'Normal')
train_data_pneumonia, train_labels_pneumonia = load_data(os.path.join(data_path, 'train'), 'Pneumonia')

test_data_normal, test_labels_normal = load_data(os.path.join(data_path, 'test'), 'Normal')
test_data_pneumonia, test_labels_pneumonia = load_data(os.path.join(data_path, 'test'), 'Pneumonia')

# Menggabungkan data train dan test
train_images = np.concatenate((train_data_normal, train_data_pneumonia), axis=0)
train_labels = np.concatenate((train_labels_normal, train_labels_pneumonia), axis=0)
test_images = np.concatenate((test_data_normal, test_data_pneumonia), axis=0)
test_labels = np.concatenate((test_labels_normal, test_labels_pneumonia), axis=0)

# Normalisasi data
train_images = train_images / 255.0
test_images = test_images / 255.0

# Membangun model CNN sederhana
model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Kompilasi model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Menambahkan dimensi channel pada data
train_images = np.expand_dims(train_images, axis=-1)
test_images = np.expand_dims(test_images, axis=-1)

# Augmentasi data
datagen = ImageDataGenerator(rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
datagen.fit(train_images)

# Pelatihan model
history = model.fit(datagen.flow(train_images, train_labels, batch_size=32), epochs=10, validation_data=(test_images, test_labels))

# Evaluasi model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

# Menampilkan plot akurasi
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

# Menampilkan plot loss
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim([0, 1])
plt.legend(loc='upper right')
plt.show()

# Fungsi untuk memuat gambar dan menguji
def load_and_test_image():
    file_path = filedialog.askopenfilename()
    img = Image.open(file_path)
    img = img.resize((150, 150), Image.BICUBIC)  # Menggunakan metode BICUBIC untuk redimensi gambar
    img = ImageTk.PhotoImage(img)
    panel.configure(image=img)
    panel.image = img

    img_array = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img_array, (150, 150))
    img_normalized = img_resized / 255.0
    img_expanded = np.expand_dims(img_normalized, axis=-1)
    result = model.predict(np.array([img_expanded]))
    if result[0][0] < 0.5:
        result_label.config(text=f'Normal, Probability: {1 - result[0][0]}')
    else:
        result_label.config(text=f'Pneumonia, Probability: {result[0][0]}')

# Membangun antarmuka
root = tk.Tk()
root.title("Chest X-ray Classification")

# Mendapatkan lebar dan tinggi layar
window_width = root.winfo_screenwidth()
window_height = root.winfo_screenheight()

# Menghitung posisi x dan y untuk GUI di tengah layar
x_position = int(window_width / 2 - root.winfo_reqwidth() / 2)
y_position = int(window_height / 2 - root.winfo_reqheight() / 2)

# Mengatur posisi GUI di tengah layar
root.geometry(f"+{x_position}+{y_position}")

# Panel untuk menampilkan gambar
panel = tk.Label(root)
panel.pack()

# Tombol untuk memuat gambar
button = tk.Button(root, text="Load Image", command=load_and_test_image)
button.pack()

# Label untuk hasil
result_label = tk.Label(root, text="")
result_label.pack()

# Menjalankan antarmuka
root.mainloop()
