#!/usr/bin/env python
# coding: utf-8

# In[3]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import json
import os


# In[6]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import json
import os

class MultiHeadAttentionLayer(layers.Layer):
    def __init__(self, num_heads, key_dim):
        super(MultiHeadAttentionLayer, self).__init__()
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)

    def call(self, inputs):
        return self.attention(inputs, inputs)

def create_transformer_model(num_objects, embedding_dim, num_heads, ff_dim):
    inputs = layers.Input(shape=(num_objects, embedding_dim))
    
    # Многоголовое внимание
    attention_output = MultiHeadAttentionLayer(num_heads, embedding_dim)(inputs)
    attention_output = layers.LayerNormalization(epsilon=1e-6)(attention_output + inputs)
    
    # Полносвязный слой
    ffn = tf.keras.Sequential([
        layers.Dense(ff_dim, activation='relu'),
        layers.Dense(embedding_dim)
    ])
    ffn_output = ffn(attention_output)
    ffn_output = layers.LayerNormalization(epsilon=1e-6)(ffn_output + attention_output)
    
    # Выходной слой
    outputs = layers.Dense(1, activation='sigmoid')(ffn_output)  # Для бинарной классификации (например, "близко" или "далеко")
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

# Параметры модели
num_objects = 10  # Количество объектов
embedding_dim = 16  # Размерность векторного представления
num_heads = 4  # Количество голов в многоголовом внимании
ff_dim = 32  # Размерность скрытого слоя

# Создание модели
model = create_transformer_model(num_objects, embedding_dim, num_heads, ff_dim)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[29]:


import os
import json
import numpy as np

def load_clevr_questions(data_dir):
    questions = []
    answers = []

    # Проходим по каждому разделу: train и val
    for split in ['CLEVR_train', 'CLEVR_val']:
        # Формируем путь к файлу вопросов
        questions_file = os.path.join(data_dir, f'{split}_questions.json')
        print(questions_file)  # Для проверки, выводим путь к файлу

        # Проверяем, существует ли файл перед его открытием
        if not os.path.exists(questions_file):
            raise FileNotFoundError(f'Файл не найден: {questions_file}')

        # Загружаем данные из файла вопросов
        with open(questions_file, 'r') as f:
            question_data = json.load(f)

        # Извлекаем вопросы и ответы
        for item in question_data['questions']:
            questions.append(item['question'])
            answers.append(item['answer'])

    return questions, answers

# Пример использования
CLEVR_DATA_DIR = 'C:/Users/Dasha/Desktop/CLEVR_v1.0'

# Загрузка данных
try:
    questions, answers = load_clevr_questions(CLEVR_DATA_DIR)
    print(f"Количество вопросов: {len(questions)}")
    print(f"Пример вопроса: {questions[0]}")
    print(f"Пример ответа: {answers[0]}")
except FileNotFoundError as e:
    print(e)
except json.JSONDecodeError:
    print("Ошибка при загрузке данных: файл не является корректным JSON.") 
except Exception as e:
    print(f"Произошла ошибка: {e}")


# In[31]:


# Пример простой токенизации (вам может понадобиться более сложный подход)
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
encoded_answers = label_encoder.fit_transform(answers)

# Преобразование вопросов в векторы (необходима реализация)
# Здесь вам нужно будет создать векторы для вопросов, возможно, используя Embedding слой.


# In[39]:


from keras.models import Sequential
from keras.layers import Dense, LSTM

# Параметры
num_objects = 10  # Количество объектов (например, временные шаги)
embedding_dim = 64  # Размерность встраивания (например, размер вектора признаков)

# Пример случайных данных для обучения (замените на ваши данные)
x_train = np.random.rand(1000, num_objects, embedding_dim)  # Случайные данные
y_train = np.random.randint(0, 2, size=(1000,))  # Случайные метки для бинарной классификации (0 или 1)

# Создание модели
model = Sequential()
model.add(LSTM(128, input_shape=(num_objects, embedding_dim), return_sequences=False))  # return_sequences=False для одного выхода
model.add(Dense(1, activation='sigmoid'))  # Выходной слой для бинарной классификации

# Компиляция модели
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Обучение модели
model.fit(x_train, y_train, epochs=10, batch_size=32)

# Вывод структуры модели
model.summary()

