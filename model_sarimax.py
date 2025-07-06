import pandas as pd
import numpy as np
import warnings
import holidays
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from pmdarima import auto_arima

# Подавление предупреждений
warnings.filterwarnings('ignore')

# Параметры
target_article = 'COUPE'  # выбор товара для анализа
forecast_horizon = 28         # число дней для прогноза

# 1. Загрузка и фильтрация данных
df = pd.read_csv('bakery_sales.csv', parse_dates=['date'])
# Фильтрация по статье
df_item = df[df['article'] == target_article].copy()

# 2. Агрегация количества проданных единиц по дате
daily_counts = (
    df_item.groupby(df_item['date'].dt.normalize())['Quantity']
           .sum()
)
daily_counts.index.name = 'date'
# Обеспечиваем ровный ряд по дням
df_filtered = daily_counts.to_frame(name='count').asfreq('D')

# 3. Праздники как экзогенная переменная
ru_holidays = holidays.country_holidays('RU')
df_filtered['is_holiday'] = df_filtered.index.isin(ru_holidays).astype(int)

df_filtered = df_filtered.asfreq('D')  # обеспечивает равномерную частоту
df_filtered['count'] = df_filtered['count'].interpolate()  # или .fillna(0)

# # Подготовка данных
# y = df_filtered['count'].copy()
#
# # Заполняем пропуски нулями или интерполяцией
# y = y.fillna(0)
# exog = df_filtered[['is_holiday']]
#
# # Автоматический подбор SARIMAX
# auto_model = auto_arima(
#     y,
#     exogenous=exog,
#     seasonal=True,
#     m=7,  # недельная сезонность
#     stepwise=True,
#     suppress_warnings=True,
#     trace=True,  # покажет, как ищет
#     error_action='ignore',
#     max_order=None,  # можно ограничить перебор
#     d=None, D=None  # автоматический выбор d и D
# )
#
# # Итоговая модель — смотри подобранные параметры
# print(auto_model.summary())


# 4. SARIMAX по количеству продаж
sarimax_model = SARIMAX(
    df_filtered['count'],
    order=(2,1,2),
    seasonal_order=(2,0,1,7),
    exog=df_filtered[['is_holiday']],
    enforce_invertibility=False
)
sarimax_res = sarimax_model.fit(disp=False)

# 5. Прогноз SARIMAX
date_last = df_filtered.index.max()
future_dates = pd.date_range(start=date_last + pd.Timedelta(days=1),
                             periods=forecast_horizon,
                             freq='D')
future_exog = pd.DataFrame(
    {'is_holiday': future_dates.isin(ru_holidays).astype(int)},
    index=future_dates
)
sarimax_pred = sarimax_res.get_forecast(
    steps=forecast_horizon,
    exog=future_exog
).predicted_mean

# 6. Подготовка данных для Prophet
prophet_df = df_filtered['count'].reset_index()
prophet_df.columns = ['ds', 'y']
prophet_df = prophet_df.dropna()

# 7. Box-Cox-преобразование с fallback на log1p
try:
    prophet_df['y_box'], lmbda = boxcox(prophet_df['y'] + 1)
    use_boxcox = True
except Exception:
    prophet_df['y_box'] = np.log1p(prophet_df['y'])
    lmbda = None
    use_boxcox = False

# 8. Обучение Prophet
prophet_model_df = prophet_df[['ds', 'y_box']].rename(columns={'y_box': 'y'})
model_prophet = Prophet(weekly_seasonality=True, yearly_seasonality=True)
model_prophet.add_country_holidays(country_name='RU')
model_prophet.fit(prophet_model_df)

# 9. Прогноз Prophet
future = model_prophet.make_future_dataframe(periods=forecast_horizon, freq='D')
forecast_prophet = model_prophet.predict(future)
if use_boxcox:
    prophet_pred = inv_boxcox(
        forecast_prophet.set_index('ds')['yhat'], lmbda
    ) - 1
else:
    prophet_pred = np.expm1(
        forecast_prophet.set_index('ds')['yhat']
    )
prophet_pred = prophet_pred.reindex(future_dates)

# 10. Визуализация: факт + прогнозы
plt.figure(figsize=(12, 6))
plt.plot(df_filtered.index, df_filtered['count'], label='Actual', linewidth=2)
plt.plot(future_dates, sarimax_pred, '--', label='SARIMAX Forecast')
plt.plot(future_dates, prophet_pred, ':', label='Prophet Forecast')
plt.legend()
plt.title(f'Количество продаж {target_article} в день: факт и прогноз на {forecast_horizon} дней')
plt.xlabel('Дата')
plt.ylabel('Количество проданных единиц')
plt.tight_layout()
plt.show()

# 11. Оценка моделей на обучении
valid_idx = df_filtered['count'].dropna().index

# SARIMAX метрики
sarimax_fit = sarimax_res.fittedvalues.loc[valid_idx]
mae_s = mean_absolute_error(df_filtered.loc[valid_idx, 'count'], sarimax_fit)
rmse_s = np.sqrt(mean_squared_error(df_filtered.loc[valid_idx, 'count'], sarimax_fit))
mape_s = (np.abs((df_filtered.loc[valid_idx, 'count'] - sarimax_fit) /
                df_filtered.loc[valid_idx, 'count']).mean()) * 100

# Prophet метрики
prophet_fit_series = forecast_prophet.set_index('ds')['yhat'].loc[valid_idx]
if use_boxcox:
    prophet_fit = inv_boxcox(prophet_fit_series, lmbda) - 1
else:
    prophet_fit = np.expm1(prophet_fit_series)
mae_p = mean_absolute_error(df_filtered.loc[valid_idx, 'count'], prophet_fit)
rmse_p = np.sqrt(mean_squared_error(df_filtered.loc[valid_idx, 'count'], prophet_fit))
mape_p = (np.abs((df_filtered.loc[valid_idx, 'count'] - prophet_fit) /
                df_filtered.loc[valid_idx, 'count']).mean()) * 100

# 12. График метрик
labels = ['SARIMAX', 'Prophet']
mae_vals = [mae_s, mae_p]
rmse_vals = [rmse_s, rmse_p]
mape_vals = [mape_s, mape_p]

x = np.arange(len(labels))
width = 0.25
plt.figure(figsize=(8,4))
plt.bar(x - width, mae_vals, width, label='MAE')
plt.bar(x, rmse_vals, width, label='RMSE')
plt.bar(x + width, mape_vals, width, label='MAPE')
plt.xticks(x, labels)
plt.legend()
plt.title('Метрики качества моделей')
plt.tight_layout()
plt.show()