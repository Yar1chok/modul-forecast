import pandas as pd
import numpy as np
import warnings
import holidays
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings('ignore')

# Параметры
target_article = 'COUPE'

# 1. Загрузка и фильтрация данных
df = pd.read_csv('bakery_sales.csv', parse_dates=['date'])
df_item = df[df['article'] == target_article].copy()

# 2. Агрегация по дате
daily_counts = (
    df_item.groupby(df_item['date'].dt.normalize())['Quantity']
    .sum()
)
daily_counts.index.name = 'date'
df_filtered = daily_counts.to_frame(name='count').asfreq('D')

# 3. Очистка выбросов
Q1 = df_filtered['count'].quantile(0.25)
Q3 = df_filtered['count'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df_filtered_no_outliers = df_filtered[(df_filtered['count'] >= lower_bound) & (df_filtered['count'] <= upper_bound)]

# 4. Учет праздников и интерполяция
ru_holidays = holidays.country_holidays('RU')
df_filtered['is_holiday'] = df_filtered.index.isin(ru_holidays).astype(int)
df_filtered['count'] = df_filtered['count'].interpolate()

# 5. Подготовка данных для Prophet
prophet_df = df_filtered_no_outliers['count'].reset_index()
prophet_df.columns = ['ds', 'y']
prophet_df = prophet_df.dropna()

# 5.1. Box-Cox или лог-преобразование
try:
    prophet_df['y_box'], lmbda = boxcox(prophet_df['y'] + 1)
    use_boxcox = True
except Exception:
    prophet_df['y_box'] = np.log1p(prophet_df['y'])
    lmbda = None
    use_boxcox = False

# 5.2. Разделение на train/test (80/20)
split_index = int(len(prophet_df) * 0.8)
train_df = prophet_df.iloc[:split_index].copy()
test_df = prophet_df.iloc[split_index:].copy()
test_dates = test_df['ds']
forecast_horizon = len(test_df)

# 6. Обучение Prophet на тренировочных данных
prophet_train_df = train_df[['ds', 'y_box']].rename(columns={'y_box': 'y'})
model_prophet = Prophet(weekly_seasonality=True, yearly_seasonality=True, changepoint_prior_scale=0.07)
model_prophet.add_country_holidays(country_name='RU')
model_prophet.fit(prophet_train_df)

# 7. Прогноз на длину тестового периода
future = model_prophet.make_future_dataframe(periods=forecast_horizon, freq='D')
forecast_prophet = model_prophet.predict(future)

# 8. Обратное преобразование
if use_boxcox:
    prophet_pred = inv_boxcox(forecast_prophet.set_index('ds')['yhat'], lmbda) - 1
else:
    prophet_pred = np.expm1(forecast_prophet.set_index('ds')['yhat'])

# 9. Оставляем только прогноз для теста
common_dates = prophet_pred.index.intersection(test_dates)
prophet_test_pred = prophet_pred.loc[common_dates]
test_real = test_df.set_index('ds').loc[common_dates]

# 10. Преобразование реальных значений теста обратно
test_y_box = test_df.set_index('ds').loc[common_dates]['y_box']

if use_boxcox:
    test_real = inv_boxcox(test_y_box, lmbda) - 1
else:
    test_real = np.expm1(test_y_box)

# 11. Визуализация прогноза на тесте
plt.figure(figsize=(12, 6))
plt.plot(common_dates, test_real, label='Факт (тест)', linewidth=2)
plt.plot(common_dates, prophet_test_pred, label='Прогноз Prophet', linestyle='--')
plt.legend()
plt.title(f'Прогноз продаж {target_article} на тестовом отрезке ({forecast_horizon} дней)')
plt.xlabel('Дата')
plt.ylabel('Продажи')
plt.tight_layout()
plt.show()

# 12. Метрики на тестовой выборке
mae = mean_absolute_error(test_real, prophet_test_pred)
rmse = np.sqrt(mean_squared_error(test_real, prophet_test_pred))
mape = (np.abs((test_real - prophet_test_pred) / test_real).mean()) * 100

print(f"[TEST] MAE: {mae:.2f}")
print(f"[TEST] RMSE: {rmse:.2f}")
print(f"[TEST] MAPE: {mape:.2f}%")
