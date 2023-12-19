import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Загрузка сохраненных моделей
model1 = load_model('model1.h5')
model2 = load_model('model2.h5')
model3 = load_model('model3.h5')
model4 = load_model('model4.h5')
model5 = load_model('model5.h5')
model6 = load_model('model6.h5')
model7 = load_model('model7.h5')

# Загрузка данных
train_dir = "D:/pfoto/Training"
test_dir = "D:/pfoto/Test"

# Генераторы данных
img_datagen = ImageDataGenerator(rescale=1./255,
                                 vertical_flip=True,
                                 horizontal_flip=True,
                                 rotation_range=40,
                                 width_shift_range=0.2,
                                 height_shift_range=0.2,
                                 zoom_range=0.1,
                                 validation_split=0.2)

# Генераторы данных для обучения и валидации
train_generator = img_datagen.flow_from_directory(train_dir,
                                                  shuffle=True,
                                                  batch_size=32,
                                                  subset='training',
                                                  target_size=(100, 100),
                                                  class_mode='categorical')

valid_generator = img_datagen.flow_from_directory(train_dir,
                                                  shuffle=True,
                                                  batch_size=16,
                                                  subset='validation',
                                                  target_size=(100, 100),
                                                  class_mode='categorical')

# Замените X_test и y_test на ваши тестовые данные
X_test, y_test = valid_generator.next()

# Предсказания вероятностей классов для каждой модели
pred1_probs = model1.predict(X_test)
pred2_probs = model2.predict(X_test)
pred3_probs = model3.predict(X_test)
pred4_probs = model4.predict(X_test)
pred5_probs = model5.predict(X_test)
pred6_probs = model6.predict(X_test)
pred7_probs = model7.predict(X_test)

# Среднее голосование по вероятностям классов
ensemble_probs = (pred1_probs + pred2_probs + pred3_probs + pred4_probs + pred5_probs + pred6_probs + pred7_probs) / 7

# Преобразование вероятностей в метки классов
ensemble_labels = np.argmax(ensemble_probs, axis=1)

# Оценка производительности ансамбля
accuracy = np.mean(ensemble_labels == np.argmax(y_test, axis=1))
print(f'Ensemble Accuracy: {accuracy}')
