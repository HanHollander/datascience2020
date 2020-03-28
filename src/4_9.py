import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

wd = os.getcwd()
data_file_path = wd[:-4] + '/data/KBESTS/GlobalLandTemperaturesByCountry.csv'
separator = ','

# read data
df = pd.read_csv(data_file_path, sep=separator, infer_datetime_format=True, parse_dates=['dt'])

# cleanup data (remove NaN)
df = df[df['AverageTemperature'].notnull()]
df = df[df['AverageTemperatureUncertainty'].notnull()]

# group data by country
temps = {g: x for g, x in df.groupby('Country')}

# list of countries to be used
countries = ['Norway', 'Finland', 'Singapore', 'Cambodia']

# a)

fig = plt.figure()
fig.suptitle('4.9 a) Average temperature plots per country', size=1)

selected_temps = {}

for i in range(0, len(countries)):
    country = countries[i]
    temp = temps[country]
    # resample data from per month to per year
    temp = temp.groupby(temp['dt'].dt.year).mean()
    print(country)
    print(temp)
    selected_temps[country] = temp
    location = int('22' + str(i + 1))
    ax = fig.add_subplot(location)
    ax.plot(temp['AverageTemperature'])
    ax.title.set_text(country)
    ax.set_xlabel('Date')
    ax.set_ylabel('Temperature')

fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(wd[:-4] + '/img/4_9_a.png')

pairs = []
for i in range(0, len(countries)):
    for j in range(0, len(countries)):
        if countries[i] != countries[j]:
            pairs.append((countries[i], countries[j]))

for pair in pairs:
    t1 = np.array(list(selected_temps[pair[0]]['AverageTemperature'].to_dict().items()))
    t2 = np.array(list(selected_temps[pair[1]]['AverageTemperature'].to_dict().items()))
    neigh = NearestNeighbors(n_neighbors=1)
    # print(t1.max())
    neigh.fit(t1)
    dists, inds = neigh.kneighbors(t2)
    # print(inds)
    print('{} {} {}'.format(pair[0], pair[1], np.mean(dists)))

# b)

for country in countries:
    temp = selected_temps[country]
    result = adfuller(temp['AverageTemperature'], autolag='AIC')
    print(country)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))

# c)

for country in countries:
    temp = selected_temps[country]
    decomposition = seasonal_decompose(temp['AverageTemperature'])
    trend = decomposition.trend
    print(trend)
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    fig = plt.figure()
    ax = fig.subplot(411)
    ax.plot(temp['AverageTemperature'][-150:], label='Original')
    ax.legend(loc='best')
    ax = fig.subplot(412)
    ax.plot(temp.index, trend[-150:].values, label='Trend')
    ax.legend(loc='best')
    ax = fig.subplot(413)
    ax.plot(ts_euro_week_log_select.index.to_pydatetime(), seasonal[-100:].values, label='Seasonality')
    ax.legend(loc='best')
    ax = fig.subplot(414)
    ax.plot(ts_euro_week_log_select.index.to_pydatetime(), residual[-100:].values, label='Residuals')
    ax.legend(loc='best')
    fig.tight_layout()


