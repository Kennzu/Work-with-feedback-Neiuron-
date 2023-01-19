from keras.datasets import imdb
import numpy as np
from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from keras import metrics
import matplotlib.pyplot as plt
from tensorflow import optimizers
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=8000) #Загрузка набора данных IMDB Аргумент num_words=  означает, что в обучающих данных будет сохранено только n слов, которые наиболее часто встречаются в обучающем наборе отзывов
print(train_data[0]) #train_data и test_data — списки отзывов; каждый отзыв — список индексов слов
print(train_labels[0]) #train_labels и test_labels — списки 0 и 1, где 0 - отрицательные отзывы, а 1 — положительные отзывы
print(max([max(sequence) for sequence in train_data])) #выводит максимальный индекс, зависит от используемых данных: сколько слов => столько индексов
print("Тест", len(test_data))
#декодирование одного из отзывов в последовательность слов на английском языке:
word_index = imdb.get_word_index() #word_index — словарь, который отображает слова в индексы (целочисленные)
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()]) # Здесь мы получаем обратное представление словаря, отображающее все реверснуто (то есть индекы в слова)
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]]) #Здесь отзыв декодируется. Индексы смещены на 3, так как индексы 0, 1 и 2 зарезервированы для слов

#Кодирование последовательностей целых чисел в бинарную матрицу
def vectorize_sequences(sequences, dimension=8000): #Создание матрицы с формой (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension)) 
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1 #Запись единицы в элемент с данным индексом
    return results
x_train = vectorize_sequences(train_data) #Обучающие данные в векторном виде (векторизация типа. Наверное...)
x_test = vectorize_sequences(test_data) #Контрольные данные в векторном виде (векторизация типа. Наверное...)
print(x_train[0])
y_train = np.asarray(train_labels).astype('float32') #Векторизация меток, тип 32 потому что он работет в 32-битной системе
y_test = np.asarray(test_labels).astype('float32') #Векторизация меток
epoch = 10 #задается кол-во эпох

# ЭТО БЫЛ НЕНУЖНЫЙ КОСТЫЛЬ РЕБЯТ. Я ВСЕ ЭПОХС ПОМЕНЯЛА НА ЭПОЧ И ЗАРАБОТАЛО


#Определение модели
model = models.Sequential()
model.add(layers.Dense(3, activation='softmax', input_shape=(8000,))) #Активация, добавление нейронных слоев. 3 - кол-во нейронов, activation - ф-ция активации input_shape - взымаемые данные
model.add(layers.Dense(3, activation='softmax')) #Активация, добавление нейронных слоев. 3 - кол-во нейронов, activation - ф-ция активации 
model.add(layers.Dense(2, activation='tanh')) #Активация, добавление нейронных слоев. 2 - кол-во нейронов, activation - ф-ция активации 
#model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy']) #Компиляция модели
#model.compile(optimizer=optimizers.RMSprop(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy']) #Настройка оптимизатора
model.compile(optimizer=optimizers.RMSprop(learning_rate=0.001), loss=losses.binary_crossentropy, metrics=[metrics.binary_accuracy]) #Использование нестандартных функций потерь и метрик, бинарная кроссэнтропия - единицы и нули
# ГУГЛАНИТЕ ЧТО ТАКОЕ ОПТИМИЗАТОР ПЛЗ 

#создание проверочного набора, выбрав n-ое кол-во образцов из оригинального набора обучающих данных
x_val = x_train[:8000]
partial_x_train = x_train[8000:]
print("Проверочные", len(partial_x_train))
y_val = y_train[:8000]
partial_y_train = y_train[8000:]

#обучение модели в течении n эпох пакетами по 512 образцов, и слежение за потерями и точностью на n oтложенных образцов
#model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])
history = model.fit(partial_x_train,partial_y_train,epochs=epoch,batch_size=512,validation_data=(x_val, y_val))
results = model.evaluate(x_test, y_test) 
# внос тестовых данных в модель

#словарь с данными обо всем происходившем в процессе обучения
history_dict = history.history
history_dict.keys()
print(history_dict.keys())

#Формирование графиков потерь на этапах обучения и проверки
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, (epoch) + 1)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
print(results)

#Формирование графиков точности на этапах обучения и проверк
plt.clf()
acc_values = history_dict['binary_accuracy']
val_acc_values = history_dict['val_binary_accuracy']
plt.plot(epochs, acc_values, 'bo', label='Training acc')
plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#Обучение новой модели с нуля
model = models.Sequential()
model.add(layers.Dense(3, activation='softmax', input_shape=(8000,)))
model.add(layers.Dense(3, activation='softmax'))
model.add(layers.Dense(2, activation='tanh'))
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(x_train, y_train, epochs=4, batch_size=512)
resultss = model.evaluate(x_test, y_test)
print(resultss)


