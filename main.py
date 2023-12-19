import logging
import os
from aiogram import Bot, Dispatcher, executor, types
from keras.models import load_model
from keras_preprocessing import image
from keras_preprocessing.image import img_to_array
import numpy as np
import tensorflow as tf
import cv2


logging.basicConfig(level=logging.INFO)
bot = Bot(token="5951261231:AAH0PvL2GZzaMj_HN8bEFogFvVSuXac6AtA")
dp = Dispatcher(bot)

# Словарь с соответствием номеров и названий папок
folder_names = {
    0: "Apple Braeburn",
    1: "Apple Golden 1",
    2: "Apple Golden 2",
    3: "Apple Golden 3",
    4: "Apple Granny Smith",
    5: "Apple Red 1",
    6: "Apple Red 2",
    7: "Apple Red 3",
    8: "Apple Red Delicious",
    9: "Apple Red Yellow",
    10: "Apricot",
    11: "Avocado",
    12: "Avocado ripe",
    13: "Banana",
    14: "Banana Red",
    15: "Cactus fruit",
    16: "Cantaloupe 1",
    17: "Cantaloupe 2",
    18: "Carambula",
    19: "Cherry 1",
    20: "Cherry 2",
    21: "Cherry Rainier",
    22: "Cherry Wax Black",
    23: "Cherry Wax Red",
    24: "Cherry Wax Yellow",
    25: "Clementine",
    26: "Cocos",
    27: "Dates",
    28: "Granadilla",
    29: "Grape Pink",
    30: "Grape White",
    31: "Grape White 2",
    32: "Grapefruit Pink",
    33: "Grapefruit White",
    34: "Guava",
    35: "Huckleberry",
    36: "Kaki",
    37: "Kiwi",
    38: "Kumquats",
    39: "Lemon",
    40: "Lemon Meyer",
    41: "Limes",
    42: "Lychee",
    43: "Mandarine",
    44: "Mango",
    45: "Maracuja",
    46: "Melon Piel de Sapo",
    47: "Mulberry",
    48: "Nectarine",
    49: "Orange",
    50: "Papaya",
    51: "Passion Fruit",
    52: "Peach",
    53: "Peach Flat",
    54: "Pear",
    55: "Pear Abate",
    56: "Pear Monster",
    57: "Pear Williams",
    58: "Pepino",
    59: "Physalis",
    60: "Physalis with Husk",
    61: "Pineapple",
    62: "Pineapple Mini",
    63: "Pitahaya Red",
    64: "Plum",
    65: "Pomegranate",
    66: "Quince",
    67: "Rambutan",
    68: "Raspberry",
    69: "Salak",
    70: "Strawberry",
    71: "Strawberry Wedge",
    72: "Tamarillo",
    73: "Tangelo",
    74: "Tomato 1",
    75: "Tomato 2",
    76: "Tomato 3",
    77: "Tomato 4",
    78: "Tomato Cherry Red",
    79: "Tomato Maroon",
    80: "Walnut"
}

@dp.message_handler(commands=['start'])
async def start(message: types.Message):
    await message.answer('Привет! Я бот, распознающий овощи и фрукты')

@dp.message_handler(commands=['help'])
async def help(message: types.Message):
    await message.answer('Просто отправьте мне изображение, которое содержит овощ или фрукт')

@dp.message_handler(content_types=[types.ContentType.PHOTO])
async def download_photo(message: types.Message):
    # загружаем фото в папку по умолчанию
    await message.photo[-1].download()
    # определяем путь к фото
    img_path = (await bot.get_file(message.photo[-1].file_id)).file_path
    # получаем предсказание
    pred = predictions(img_path)
    # Отправляем ответ пользователю
    await message.answer(f"Я думаю, что это {folder_names.get(pred, 'неизвестная папка')} 😊")

def predictions(img_path):
    model = load_model("ensemble_model.h5")
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(100, 100))
    img = img_to_array(img)
    img = img / 255.0
    try:
        prediction_image = np.array(img)
        prediction_image = np.expand_dims(img, axis=0)
        prediction = model.predict(prediction_image)
        value = np.argmax(prediction)
        return value
    except Exception:
        return "Я тебя не понимаю"

async def main():
    await dp.start_polling(bot)

if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)