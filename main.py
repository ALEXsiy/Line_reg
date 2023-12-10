import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.linear_model import LinearRegression
# N = 30
# a_min = 0
# b_max = 250
#
# #https://www.kinonews.ru/kassa_top100/
#
# # # Рандомные точки для начала
# # x_0 = np.random.uniform(a_min, b_max, N)
# # y_0 = np.random.uniform(a_min, b_max, N)
# # # Объединяем массивы в один двумерный массив
# # combined_arr = np.vstack((x_0, y_0)).T  # T - транспонирование для правильного расположения данных
# # # Записываем объединенный массив в файл CSV
# # np.savetxt('combined_array.csv', combined_arr, delimiter=',')
# # plt.scatter(x_0,y_0)
# # plt.plot()
# # plt.show()
#
# # Считываем данные из файла CSV
# data = np.loadtxt('combined_array.csv', delimiter=',')
# # Разделяем данные на два массива
# x_1 = data[:, 0]  # Первый столбец
# y_1 = data[:, 1]  # Второй столбец
# # data = pd.read_csv('combined_array.csv')
# # data.head()
# X = pd.DataFrame(x_1)
# y = pd.DataFrame(y_1)
#
# # Создание объекта модели линейной регрессии
# model = LinearRegression()
#
# # Обучение модели
# model.fit(X, y)
#
# plt.scatter(x_1, y_1)
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('Linear Regression')
# plt.plot(X, model.predict(X), color='red', linewidth=1.5)
# plt.show()
#
# print(f"Coef: {model.coef_} "
#       f"\nIntercept: {model.intercept_} "
#       f"\nAccuracy: {model.score(X,y)*100} %")
#
#
#
#
# ##################   TOP FILMS   ########################
#
# # Считываем данные из файла CSV
# data = np.loadtxt('film.csv', delimiter=',')
# # Разделяем данные на два массива
# box_office = data[:, 0]  # Первый столбец
# budget = data[:, 1]  # Второй столбец
# # data = pd.read_csv('combined_array.csv')
# # data.head()
# X = pd.DataFrame(budget)
# y = pd.DataFrame(box_office)
#
# # Создание объекта модели линейной регрессии
# model = LinearRegression()
#
# # Обучение модели
# model.fit(X, y)
#
# plt.scatter(budget, box_office)
# plt.xlabel('Budget')
# plt.ylabel('Box office')
# plt.title('Films(budget - box office )')
# plt.plot(X, model.predict(X), color='red', linewidth=1.5)
# plt.show()
#
# print(f"\n\nFilms\nCoef: {model.coef_} "
#       f"\nIntercept: {model.intercept_} "
#       f"\nAccuracy: {model.score(X,y)*100} %")
#
#


###################################################################
def mse(y_true, y_predicted):
      cost = np.sum((y_true - y_predicted) ** 2) / len(y_true)
      return cost


def gradient(x, y, iterations=1000, coefff=0.0001, stop=1e-6):
      current_w = 0.1
      current_b = 0.01
      iterations = iterations
      coefff = coefff
      n = 20

      costs = []
      weights = []
      pred_cost = None

      for i in range(iterations):
            y_predicted = (current_w * x) + current_b
            current_cost = mse(y, y_predicted)


            if pred_cost and abs(pred_cost - current_cost) <= stop:
                  break

            pred_cost = current_cost

            costs.append(current_cost)
            weights.append(current_w)

            # gradients
            w_derivative = -(2 / n) * sum(x * (y - y_predicted))
            b_derivative = -(2 / n) * sum(y - y_predicted)


            current_w = current_w - (coefff * w_derivative)
            current_b = current_b - (coefff * b_derivative)


      plt.plot(weights, costs)
      plt.scatter(weights, costs, marker='o', color='red')
      plt.title("Cost vs W")
      plt.ylabel("Cost")
      plt.xlabel("W")
      plt.show()

      return current_w, current_b


X = np.random.uniform(0, 100, 20)
Y = [2 * i + 3 + random.uniform(0, 100) for i in X]


par_a, par_b = gradient(X, Y)
print(f"Estimated Weight: {par_a}\nEstimated Bias: {par_b}")

# Making predictions using estimated parameters
Y_pred = par_a * X + par_b

plt.scatter(X, Y, marker='o', color='red')
plt.plot(X, Y_pred, color='blue')
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

