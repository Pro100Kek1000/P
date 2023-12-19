import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os

def plot_loss(history, model_name):
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    plt.title(f'Loss for {model_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def plot_accuracy(history, model_name):
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='validation')
    plt.title(f'Accuracy for {model_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

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

# Модель 1
model1 = Sequential()
model1.add(Conv2D(32, kernel_size=(3,3), input_shape=(100,100,3), activation='relu', padding='same'))
model1.add(MaxPooling2D(pool_size=(2,2)))
model1.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'))
model1.add(MaxPooling2D(pool_size=(2,2)))
model1.add(Flatten())
model1.add(Dense(256, activation='relu'))
model1.add(Dropout(0.5))
model1.add(Dense(len(train_generator.class_indices), activation='softmax'))

model1.compile(optimizer='adam',
               loss='categorical_crossentropy',
               metrics=['accuracy'])

# Обучение модели 1
history1 = model1.fit(train_generator, validation_data=valid_generator,
                      steps_per_epoch=train_generator.n // train_generator.batch_size,
                      validation_steps=valid_generator.n // valid_generator.batch_size,
                      epochs=25)

# Построение графиков для модели 1
plot_loss(history1, 'Model 1')
plot_accuracy(history1, 'Model 1')

# Сохранение модели 1
model1.save('model1.h5')

# Модель 2
model2 = Sequential()
model2.add(Conv2D(32, kernel_size=(3,3), input_shape=(100,100,3), activation='relu', padding='same'))
model2.add(MaxPooling2D(pool_size=(2,2)))
model2.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'))
model2.add(MaxPooling2D(pool_size=(2,2)))
model2.add(Conv2D(128, kernel_size=(3,3), activation='relu', padding='same'))
model2.add(MaxPooling2D(pool_size=(2,2)))
model2.add(Flatten())
model2.add(Dense(256, activation='relu'))
model2.add(Dropout(0.5))
model2.add(Dense(len(train_generator.class_indices), activation='softmax'))

model2.compile(optimizer='adam',
               loss='categorical_crossentropy',
               metrics=['accuracy'])

# Обучение модели 2
history2 = model2.fit(train_generator, validation_data=valid_generator,
                      steps_per_epoch=train_generator.n // train_generator.batch_size,
                      validation_steps=valid_generator.n // valid_generator.batch_size,
                      epochs=25)
# Построение графиков для модели 1
plot_loss(history2, 'Model 2')
plot_accuracy(history2, 'Model 2')

# Сохранение модели 2
model2.save('model2.h5')

# Модель 3
model3 = Sequential()
model3.add(Conv2D(32, kernel_size=(3,3), input_shape=(100,100,3), activation='relu', padding='same'))
model3.add(MaxPooling2D(pool_size=(2,2)))
model3.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'))
model3.add(MaxPooling2D(pool_size=(2,2)))
model3.add(Conv2D(128, kernel_size=(3,3), activation='relu', padding='same'))
model3.add(MaxPooling2D(pool_size=(2,2)))
model3.add(Conv2D(256, kernel_size=(3,3), activation='relu', padding='same'))
model3.add(MaxPooling2D(pool_size=(2,2)))
model3.add(Flatten())
model3.add(Dense(512, activation='relu'))
model3.add(Dropout(0.5))
model3.add(Dense(len(train_generator.class_indices), activation='softmax'))

model3.compile(optimizer='adam',
               loss='categorical_crossentropy',
               metrics=['accuracy'])

# Обучение модели 3
history3 = model3.fit(train_generator, validation_data=valid_generator,
                      steps_per_epoch=train_generator.n // train_generator.batch_size,
                      validation_steps=valid_generator.n // valid_generator.batch_size,
                      epochs=25)
# Построение графиков для модели 1
plot_loss(history3, 'Model 3')
plot_accuracy(history3, 'Model 3')


# Сохранение модели 3
model3.save('model3.h5')

# Модель 4
model4 = Sequential()
model4.add(Conv2D(32, kernel_size=(3,3), input_shape=(100,100,3), activation='relu', padding='same'))
model4.add(MaxPooling2D(pool_size=(2,2)))
model4.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'))
model4.add(MaxPooling2D(pool_size=(2,2)))
model4.add(Conv2D(128, kernel_size=(3,3), activation='relu', padding='same'))
model4.add(MaxPooling2D(pool_size=(2,2)))
model4.add(Flatten())
model4.add(Dense(512, activation='relu'))
model4.add(Dropout(0.5))
model4.add(Dense(len(train_generator.class_indices), activation='softmax'))

model4.compile(optimizer='adam',
               loss='categorical_crossentropy',
               metrics=['accuracy'])

# Обучение модели 4
history4 = model4.fit(train_generator, validation_data=valid_generator,
                      steps_per_epoch=train_generator.n // train_generator.batch_size,
                      validation_steps=valid_generator.n // valid_generator.batch_size,
                      epochs=25)

# Построение графиков для модели 1
plot_loss(history4, 'Model 4')
plot_accuracy(history4, 'Model 4')

# Сохранение модели 4
model4.save('model4.h5')

# Модель 5
model5 = Sequential()
model5.add(Conv2D(32, kernel_size=(3,3), input_shape=(100,100,3), activation='relu', padding='same'))
model5.add(MaxPooling2D(pool_size=(2,2)))
model5.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'))
model5.add(MaxPooling2D(pool_size=(2,2)))
model5.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'))
model5.add(MaxPooling2D(pool_size=(2,2)))
model5.add(Flatten())
model5.add(Dense(256, activation='relu'))
model5.add(Dropout(0.5))
model5.add(Dense(len(train_generator.class_indices), activation='softmax'))

model5.compile(optimizer='adam',
               loss='categorical_crossentropy',
               metrics=['accuracy'])

# Обучение модели 5
history5 = model5.fit(train_generator, validation_data=valid_generator,
                      steps_per_epoch=train_generator.n // train_generator.batch_size,
                      validation_steps=valid_generator.n // valid_generator.batch_size,
                      epochs=25)
# Построение графиков для модели 1
plot_loss(history5, 'Model 5')
plot_accuracy(history5, 'Model 5')

# Сохранение модели 5
model5.save('model5.h5')

# Модель 6
model6 = Sequential()
model6.add(Conv2D(64, kernel_size=(3,3), input_shape=(100,100,3), activation='relu', padding='same'))
model6.add(MaxPooling2D(pool_size=(2,2)))
model6.add(Conv2D(128, kernel_size=(3,3), activation='relu', padding='same'))
model6.add(MaxPooling2D(pool_size=(2,2)))
model6.add(Conv2D(256, kernel_size=(3,3), activation='relu', padding='same'))
model6.add(MaxPooling2D(pool_size=(2,2)))
model6.add(Flatten())
model6.add(Dense(512, activation='relu'))
model6.add(Dropout(0.5))
model6.add(Dense(len(train_generator.class_indices), activation='softmax'))

model6.compile(optimizer='adam',
               loss='categorical_crossentropy',
               metrics=['accuracy'])

# Обучение модели 6
history6 = model6.fit(train_generator, validation_data=valid_generator,
                      steps_per_epoch=train_generator.n // train_generator.batch_size,
                      validation_steps=valid_generator.n // valid_generator.batch_size,
                      epochs=25)
# Построение графиков для модели 1
plot_loss(history6, 'Model 6')
plot_accuracy(history6, 'Model 6')

# Сохранение модели 6
model6.save('model6.h5')

# Модель 7
model7 = Sequential()
model7.add(Conv2D(32, kernel_size=(3,3), input_shape=(100,100,3), activation='relu', padding='same'))
model7.add(MaxPooling2D(pool_size=(2,2)))
model7.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'))
model7.add(MaxPooling2D(pool_size=(2,2)))
model7.add(Conv2D(128, kernel_size=(3,3), activation='relu', padding='same'))
model7.add(MaxPooling2D(pool_size=(2,2)))
model7.add(Flatten())
model7.add(Dense(256, activation='relu'))
model7.add(Dropout(0.5))
model7.add(Dense(len(train_generator.class_indices), activation='softmax'))

model7.compile(optimizer='adam',
               loss='categorical_crossentropy',
               metrics=['accuracy'])

# Обучение модели 7
history7 = model7.fit(train_generator, validation_data=valid_generator,
                      steps_per_epoch=train_generator.n // train_generator.batch_size,
                      validation_steps=valid_generator.n // valid_generator.batch_size,
                      epochs=25)
# Построение графиков для модели 7
plot_loss(history7, 'Model 7')
plot_accuracy(history7, 'Model 7')

# Сохранение модели 7
model7.save('model7.h5')