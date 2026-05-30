# statistics-homework-2

[Ссылка на репозиторий](https://github.com/nikita786574-boop/statistics-homework-2)

## task1

В приложенном файле дано распределение 100 студентов по росту. Построить гистограмму относительных частот, самостоятельно выбрав подходящий интервал для разбиения (вручную или с использованием любых программ). Найти несмещённые оценки для выборочного среднего, выборочной дисперсии.

### solution

Количество интервалов возьмём равным

$$k = 1 + \log_{2}(n)=1+\log_{2}(100)=7.643856189775$$

Целое будет $k=7$


Код

``` Python
import pandas as pd
import matplotlib.pyplot as plt

k = 7
df = pd.read_csv("./height-dz2.dat", header=None, names=['height'])

df.sort_values('height', inplace = True)

df['groups'] = pd.cut(df['height'], bins = k)

df_groups_count = df['groups'].value_counts().reset_index()

df_groups_count = pd.DataFrame({
    'groups': df_groups_count['groups'],
    'count': df_groups_count['count'],
    'frequency': df_groups_count['count']/sum(df_groups_count['count'])
}).sort_values('groups')

df_groups_count['middle'] = df_groups_count['groups'].apply(lambda x: (x.left + x.right)/2)


fig, ax = plt.subplots()
item = df_groups_count['groups'][0]
width = item.left - item.right
bars = ax.bar(df_groups_count['middle'],
              df_groups_count['frequency'],
              width = width,
              align = 'center')

ax.set_xlabel('Частичный интевервал')
ax.set_ylabel('Частота на интервале')
ax.set_title('Гистограма относительных частот')

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x()+bar.get_width()/2, height, str(float(height)), ha = 'center')

print('Несмещенное выборочное среднее:')
print(round(df['height'].mean(), 3))

print('Несмещенная выборочная дисперсия:')
print(round(df['height'].var(), 3))

```

Гистограма относительных частот


![Гистограма относительных частот](task1/hist_freq_task_1.png)



Несмещенное выборочное среднее:

Считается по формуле

$$\overline{x} = \frac{1}{n} \cdot \sum_{i=1}^{n}x_i$$

$$\overline{x} = 171.383 $$

Несмещенная выборочная дисперсия:

Считается по формуле

$$\sigma^2 = \frac{1}{n-1} \cdot \sum_{i=1}^{n}(x_i - \overline{x})^2$$


$$\sigma^2 = 113.655$$


## task2

Воспользуйтесь встроенным датасетом Iris из библиотеки scikit-learn. Для этого выполните следующий код на python:

``` Python

from sklearn.datasets import load_iris
import pandas as pd

data = load_iris()
df = pd.DataFrame(data.data, columns = data.feature_names)

```

1. Вычислите выборочную ковариация между признаками `sepal length(cm)` и `petal length(cm)`.
2. Вычислите выборочный коэффициент корреляции между этими же признаками.
3. Проинтерпретируйте знак и значение коэффициента корреляции: укажите, существует ли линейная связь между признаками и насколько она сильна

### solution

#### 2.1
Выборочная ковариация считается по формуле

$$s_{xy} = \frac{1}{n-1}\cdot \sum_{i=1}^{n}(x_i - \overline{x})\cdot (y_i - \overline{y})$$

$$s_{xy} = 1.2743$$

#### 2.2

Выборочный коэффициент корреляции считается по формуле:

$$r_{xy}=\frac{s_{xy}}{s_{x} \cdot s_{y}}$$

Где 

$s_{x}$ - выборочное стандартное отклонение первой переменной

$s_{y}$ - выборочное стандартное отклонение второй переменной

$$r_{xy} = 0.8718$$

#### 2.3

Знаки +, значит наша модель будет предсказывать, что значение второй переменной будет увеличиваться при увеличении первой.

Корреляция получась близкой к 1, значит между переменными есть значительная связь.

Scatter plot

![scatter plot](task2/scatter_plot_task_2.png)

Из большого положительного коэффициента корреляции и графика хорошо видно, что зависимость почти линейная.

