import pandas as pd
import numpy as np
from prophet import Prophet
import holidays
import matplotlib.pyplot as plt
from prophet.diagnostics import cross_validation, performance_metrics

# Параметры
target_article = 'BAGUETTE'  # товар для анализа
forecast_horizon = 7         # число дней для прогноза

# 1. Загрузка данных
df = pd.read_csv('bakery_sales.csv', parse_dates=['date'])

# 2. Фильтрация по товару
df_item = df[df['article'] == target_article].copy()

# 3. Агрегация количества продаж по дням
daily_counts = (
    df_item.groupby(df_item['date'].dt.normalize())['Quantity']
           .sum()
)
daily_counts.index.name = 'date'

# 4. Формирование DataFrame с частотой 'D'
df_filtered = daily_counts.to_frame(name='count').asfreq('D')

# 5. Подготовка данных для Prophet
daily = df_filtered.reset_index().rename(columns={'date': 'ds', 'count': 'y'})

# 6. Обработка выбросов
lower, upper = daily['y'].quantile([0.01, 0.99])
daily['y_clipped'] = daily['y'].clip(lower=lower, upper=upper)

# 7. Лог-преобразование
daily['y_log'] = np.log1p(daily['y_clipped'])

# 8. Календарь праздников России
years = daily['ds'].dt.year.unique().tolist() + [2025]
ru_holidays = holidays.Russia(years=years)

holiday_df = (
    pd.DataFrame(
        [(date, name) for date, name in ru_holidays.items()],
        columns=['ds', 'holiday']
    )
    .drop_duplicates()
)
holiday_df['ds'] = pd.to_datetime(holiday_df['ds'])

# 9. Настройка и обучение модели Prophet
m = Prophet(
    weekly_seasonality=True,
    yearly_seasonality=True,
    seasonality_mode='multiplicative',
    changepoint_prior_scale=0.8,
    holidays=holiday_df
)
m.add_seasonality(name='monthly', period=30.5, fourier_order=12)

# Обучение на лог-преобразованных данных
daily_fit = daily[['ds', 'y_log']].rename(columns={'y_log': 'y'})
m.fit(daily_fit)

# 10. Прогнозирование
future = m.make_future_dataframe(periods=forecast_horizon)
forecast = m.predict(future)

# Обратное лог-преобразование
forecast['yhat_exp'] = np.expm1(forecast['yhat'])
forecast['yhat_lower_exp'] = np.expm1(forecast['yhat_lower'])
forecast['yhat_upper_exp'] = np.expm1(forecast['yhat_upper'])

# 11. Визуализация
plt.figure(figsize=(10, 6))
plt.plot(daily['ds'], daily['y'], label='Actual', linewidth=2)
plt.plot(forecast['ds'], forecast['yhat_exp'], label='Forecast', linestyle='--', linewidth=2)
plt.fill_between(
    forecast['ds'],
    forecast['yhat_lower_exp'],
    forecast['yhat_upper_exp'],
    alpha=0.3,
    label='Confidence Interval'
)

plt.title(f'Sales Forecast for {target_article}')
plt.xlabel('Date')
plt.ylabel('Quantity Sold')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 12. Кросс-валидация и метрики
df_cv = cross_validation(m, initial='365 days', period='30 days', horizon=f'{forecast_horizon} days')
df_p = performance_metrics(df_cv)

print("Cross-validation performance metrics:")
print(df_p[['horizon', 'mape', 'rmse', 'mae', 'coverage']])

# Сохранение метрик (опционально)
df_p.to_csv(f'perf_metrics_{target_article}.csv', index=False)
