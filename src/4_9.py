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
# //df = df[df['AverageTemperature'].notnull()]
# //df = df[df['AverageTemperatureUncertainty'].notnull()]

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
    temp = temp.set_index('dt')
    # select 1864 up to and including 2012 (consistent between countries)
    temp = temp[-(12 * 150) + 3:-9]
    selected_temps[country] = temp
    location = int('22' + str(i + 1))
    ax = fig.add_subplot(location)
    ax.plot(temp['AverageTemperature'])
    ax.title.set_text(country)
    ax.set_xlabel('Date')
    ax.set_ylabel('Temperature')

print('a)')
print('')
print('The data is shown in 4_9_a.png (per month).')
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(wd[:-4] + '/img/4_9_a.png')

pairs = []
for i in range(0, len(countries)):
    for j in range(0, len(countries)):
        if countries[i] != countries[j]:
            pairs.append((countries[i], countries[j]))

print('')
print('k nearest neighbour distances between two countries (mean):')
for pair in pairs:
    t1 = np.array([(g.year, x) for g, x in
                   list(selected_temps[pair[0]]['AverageTemperature'].to_dict().items()) if not np.isnan(x)])
    t2 = np.array([(g.year, x) for g, x in
                   list(selected_temps[pair[1]]['AverageTemperature'].to_dict().items()) if not np.isnan(x)])
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(t1)
    dists, inds = neigh.kneighbors(t2)
    print('\t{} {} {}'.format(pair[0], pair[1], np.mean(dists)))

print('')
print('We can see that the temperature data for Norway and Finland is quite similar, and so is the data for Cambodia\n'
      'and Singapore, since they have the least difference between them. This is to be expected, considering the\n'
      'geographical locations of the countries.')

# b)

print('')
print('b)')
print('')
print('Dickey-Fuller tests for the four countries:')
for country in countries:
    temp = selected_temps[country]
    temp = temp.resample('Y').mean()
    result = adfuller(temp['AverageTemperature'], autolag='AIC')
    print('\tDickey-Fuller test for ' + country)
    print('\tADF Statistic: %f' % result[0])
    print('\tp-value: %f' % result[1])
    print('\tCritical Values:')
    for key, value in result[4].items():
        print('\t\t%s: %.3f' % (key, value))

print('')
print('For Norway and Finland the outcome is good enough for the data to be considered stationary, but not for\n'
      'Cambodia and Singapore.')

# c)

print('')
print('c)')
print('')
print('The decompositions of the data for the countries is shown in 4_9_Decomposition_[country].png.')
decomposed_temps = {}

for country in countries:
    temp = selected_temps[country]
    decomposition = seasonal_decompose(temp['AverageTemperature'], extrapolate_trend='freq')
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    decomposed_temps[country] = decomposition

    fig = plt.figure()
    ax = fig.add_subplot(411)
    ax.plot(temp['AverageTemperature'], label='Original')
    ax.legend(loc='best')
    ax = fig.add_subplot(412)
    ax.plot(temp.index, trend.values, label='Trend')
    ax.legend(loc='best')
    ax = fig.add_subplot(413)
    ax.plot(temp.index, seasonal.values, label='Seasonality')
    ax.legend(loc='best')
    ax = fig.add_subplot(414)
    ax.plot(temp.index, residual.values, label='Residuals')
    ax.legend(loc='best')
    fig.suptitle('Decomposition for ' + country)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(wd[:-4] + '/img/4_9_b_Decompostion_' + country + '.png')

print('')
print('Dickey-Fuller tests for the four countries:')
for country in countries:
    residual = decomposed_temps[country].resid.values
    residual = np.mean(residual.reshape(-1, 12), axis=1)
    result = adfuller(residual, autolag='AIC')
    print('\tDickey-Fuller test after decomposition for ' + country)
    print('\tADF Statistic: %f' % result[0])
    print('\tp-value: %f' % result[1])
    print('\tCritical Values:')
    for key, value in result[4].items():
        print('\t\t%s: %.3f' % (key, value))

print('')
print('For all countries the outcome is good enough for the data to be considered stationary.')

print('')
print('k nearest neighbour distances between two countries (mean):')
for pair in pairs:
    t1 = np.array([(g.year, x) for g, x in
                   list(decomposed_temps[pair[0]].resid.to_dict().items()) if not np.isnan(x)])
    t2 = np.array([(g.year, x) for g, x in
                   list(decomposed_temps[pair[1]].resid.to_dict().items()) if not np.isnan(x)])
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(t1)
    dists, inds = neigh.kneighbors(t2)
    print('\t{} {} {}'.format(pair[0], pair[1], np.mean(dists)))

print('')
print('The differenced data for the countries is shown in 4_9_Differencing_[country].png.')
for country in countries:
    temp = selected_temps[country]
    temp_diff = temp['AverageTemperature'] - temp['AverageTemperature'].shift()

    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax.plot(temp['AverageTemperature'], label='Original')
    ax.legend(loc='best')
    ax = fig.add_subplot(212)
    ax.plot(temp_diff, label='After Differencing')
    ax.legend(loc='best')
    fig.suptitle('Decomposition for ' + country)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(wd[:-4] + '/img/4_9_b_Differencing_' + country + '.png')

print('')
print('Dickey-Fuller tests for the four countries:')
for country in countries:
    temp = selected_temps[country]
    temp_diff = temp['AverageTemperature'] - temp['AverageTemperature'].shift()
    temp_diff = temp_diff.resample('Y').mean()
    result = adfuller(temp_diff, autolag='AIC')
    print('\tDickey-Fuller test after differencing for ' + country)
    print('\tADF Statistic: %f' % result[0])
    print('\tp-value: %f' % result[1])
    print('\tCritical Values:')
    for key, value in result[4].items():
        print('\t\t%s: %.3f' % (key, value))

print('')
print('For all countries the outcome is good enough for the data to be considered stationary.')

print('')
print('k nearest neighbour distances between two countries (mean):')
for pair in pairs:
    t1 = np.array([(g.year, x) for g, x in
                   list((selected_temps[pair[0]]['AverageTemperature'] -
                         (selected_temps[pair[0]]['AverageTemperature'].shift()))
                        .to_dict().items()) if not np.isnan(x)])
    t2 = np.array([(g.year, x) for g, x in
                   list((selected_temps[pair[1]]['AverageTemperature'] -
                         (selected_temps[pair[1]]['AverageTemperature'].shift()))
                        .to_dict().items()) if not np.isnan(x)])
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(t1)
    dists, inds = neigh.kneighbors(t2)
    print('\t{} {} {}'.format(pair[0], pair[1], np.mean(dists)))
