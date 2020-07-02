#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Checken der Working Directory
get_ipython().run_line_magic('pwd', '')


# In[10]:


import os

os.chdir('C:\\Users\Yusuf Konyalicetin\Desktop\Github Airbnb\AirbnbAnalyticsBerlin(final)')


# In[2]:


#Laden der Module
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import statsmodels.api as sm
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt

#Sonstige Module und Funktionen
from time import gmtime, strftime #Für die aktuelle Zeit

import warnings


# In[3]:


#Pandas - Mehr Zeilen und Spalten anzeigen
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

#Seaborn
sns.set_style("darkgrid")
plt.matplotlib.style.use('default')

my_colors = ["windows blue", "saffron", "hot pink", "algae green", "dusty purple", "greyish", "petrol", "denim blue", "lime"]
sns.set_palette(sns.xkcd_palette(my_colors))
colors = sns.xkcd_palette(my_colors)

#Warnings
warnings.filterwarnings("ignore")


# In[5]:


def my_df_summary(data):
    '''Eigene Funktion für die Summary'''
    try:
        dat = data.copy()
        df = pd.DataFrame([dat.min(), dat.max(), dat.mean(), dat.std(), dat.isna().sum(), dat.nunique(), dat.dtypes],
        index=['Minimum', 'Maximum', 'Mittelwert', 'Stand. Abw.','#NA', '#Uniques', 'dtypes']) 
        return df
    except:
        print('Es konnte keine Summary erstellt werden.')
    return data


# In[5]:


#Alle Dateien die in Working Directory enthalten sind
get_ipython().system('dir')


# In[13]:


#Einlesen der einzelnen Datensätze
#neighbourhoods = pd.read_csv('neighbourhoods.csv')
reviews_detailed = pd.read_csv('reviews_detailed.csv')
#reviews = pd.read_csv('reviews.csv')
listings = pd.read_csv('listings.csv')
#listings_summary = pd.read_csv('listings_summary.csv')


# In[14]:


reviews_detailed.head()


# In[15]:


#len(reviews)


# In[9]:


#reviews.head()


# In[16]:


listings.head()


# In[17]:


df = pd.read_csv("listings.csv")
list_drop=["listing_url","scrape_id","last_scraped","experiences_offered",
"thumbnail_url","medium_url","picture_url","xl_picture_url","host_url",
"host_thumbnail_url","host_picture_url","minimum_minimum_nights","maximum_minimum_nights",
"minimum_maximum_nights","maximum_maximum_nights","minimum_nights_avg_ntm","maximum_nights_avg_ntm",
"calendar_last_scraped"]
[df.drop(x, axis=1, inplace=True) for x in list_drop]
df.head()


# In[18]:


df2 = pd.merge(left=reviews_detailed, right=df, how='left', left_on='listing_id', right_on='id')


# In[19]:


df2.head()


# In[20]:


df2.tail()


# In[21]:


df2


# In[22]:


df2.columns


# In[23]:


#Filtern des Datensatzes für Zeitreihenanalyse
df_TS = df2[['listing_id', 'date', 'neighbourhood', 'neighbourhood_cleansed', 'neighbourhood_group_cleansed']]


# In[24]:


df_TS.info()


# In[25]:


df2['neighbourhood_group_cleansed'].unique()


# In[26]:


df2['neighbourhood_cleansed'].unique()


# In[27]:


df2['neighbourhood'].unique()


# In[28]:


df_TS[['neighbourhood']]


# In[29]:


df_TS['date'] = pd.to_datetime(df_TS['date'])


# In[30]:


df_TS.groupby('neighbourhood_group_cleansed').count()


# In[31]:


df_TS


# In[32]:


df_TS.set_index('date', inplace=True)


# In[33]:


df_TS.groupby('date').count()


# In[34]:


df_TS.groupby('date').count().plot()


# ### Frequenzen

# In[35]:


df_ = df2.copy()

s_Dates = pd.to_datetime(df_.iloc[:,2], format='%Y-%m-%d', errors='ignore')

#Mit "Nummer" des Wochentages
df_NuDay = pd.DataFrame(s_Dates.dt.dayofweek.value_counts(dropna=False))
df_NuDay = df_NuDay.reset_index()
df_NuDay.columns=['#Tag', 'Anzahl']

#Mit "Namen" des Wochentages
df_NaDay = pd.DataFrame(s_Dates.dt.day_name().value_counts(dropna=False))
df_NaDay = df_NaDay.reset_index()
df_NaDay.columns=['Tag', 'Anzahl']


# In[36]:


df_NaDay


# In[37]:


#Oder grafisch
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = [15, 4]
fig=sns.barplot(x='Tag', y='Anzahl', data=df_NaDay)
plt.title("Wie häufig sind die einzelnen Wochentage enthalten?", size=14)
plt.xlabel("")
plt.ylabel("Häufigkeiten")
#plt.show()

plt.savefig('Verteilung_Reviews_auf_Tage.png')


# ### Durch ungleichmäßige Verteilung der Anzahl der Reviews auf die Wochentage, werden die Werte auf Wochenbasis geglättet

# In[38]:


df_ = df2.copy()

s_Dates = pd.to_datetime(df_.iloc[:,2], format='%Y-%m-%d', errors='ignore')

#Mit "Nummer" der Woche
df_NuWeek = pd.DataFrame(s_Dates.dt.to_period('W').value_counts(dropna=False))
df_NuWeek = df_NuWeek.reset_index()
df_NuWeek.columns=['#Woche', 'Anzahl']

#Mit "Namen" des Wochentages
df_NaWeek = pd.DataFrame(s_Dates.dt.to_period('W').value_counts(dropna=False))
df_NaWeek = df_NaWeek.reset_index()
df_NaWeek.columns=['Woche', 'Anzahl']


# In[39]:


df_NuWeek.sort_values('#Woche')


# In[40]:


df_NaWeek.sort_values('Woche')


# In[41]:


df_NaWeek.info()


# ### Einführung der wöchentlichen Daten

# In[42]:


df_TS = df_TS.reset_index()


# In[43]:


df_TS


# In[44]:


df_NaWeek


# In[45]:


df_TS = df_TS.resample('W', on='date').count()


# In[46]:


#Das muss Zeitreihe für Zeitreihe individuell gemacht werden!
#Schritt 0: Auswahl einer Zeitreihe
df_ = df_TS.iloc[:,0:1].copy() #Hier wähle ich den Index (Spalte 0) und die erste Datenspalte (Spalte 1) aus.

# df.iloc[:,[0,4]].copy() #So würde ich bspw. die 4. Datenspalte (Spalte 5) auswählen.


# In[47]:


df_.columns = ['Reviews_Count']


# In[48]:


df_.reset_index()


# In[43]:


#War sinnvoll für tägliche Daten
#df_ = df_.groupby('date').count().reset_index()
#df_.columns = ['Date', 'Reviews_Count']


# In[49]:


#Cleansing
#Schritt 1:
l_colnames = df_.columns.to_list()
l_colnames[0] = 'Date'
df_.columns = l_colnames

df_['Date'] = pd.to_datetime(df_['Date'], format='%Y-%m-%d', errors='ignore')
daterange = pd.date_range(start=min(df_['Date']), end=max(df_['Date']), freq='W')
df_ts = pd.DataFrame(daterange)
df_ts.columns = ['Date']
df_ts = df_ts.merge(df_, how='left', on='Date')

#Schritt 2
#print('{} fehlende Werte werden durch den zuletzt gültigen Wert ersetzt.'.format(df_ts.iloc[:,1].isna().sum()))
#df_ts = df_ts.fillna(method='ffill')
#df_ts = df_ts.set_index('Date', drop=True)


# In[50]:


df_ts


# In[51]:


df_ts = df_TS


# In[52]:


df_ts.columns = ['Date', 'listing_id', 'neighbourhood', 'neighbourhood_cleansed',
       'neighbourhood_group_cleansed']


# In[53]:


df_ts_p = df_ts.copy()


# In[54]:


df_ts_p = df_ts_p.drop(['Date', 'neighbourhood', 'neighbourhood_cleansed', 'neighbourhood_group_cleansed'], axis = 1)


# In[55]:


df_ts_p.columns = ['Reviews_Count']


# In[56]:


df_ts_p


# In[57]:


#Plotten der vollständigen Daten
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = [15, 6]
fig = df_ts_p.plot(kind='line')
plt.title('Vollständige Zeitreihe', size=18)
plt.legend(fontsize=12)
plt.ylabel('Anzahl der Reviews', size=12)
plt.legend()
plt.xlabel('Zeit')
#plt.show()


plt.savefig('vollständige_Zeitreihe.png')


# In[58]:


df = df_ts.reset_index()


# In[59]:


#Ausreißer Anzeigen - Für Zeitreihen NICHT entfernen.
#Ausreißer erkennt man in Zeitreihen an dem Verhalten der prozentualen Veränderungen zum jeweils vorherigen Wert.
get_ipython().run_line_magic('matplotlib', 'inline')
df_ = df.iloc[:,0:2].copy()
df_.iloc[:,1] = np.log(df_.iloc[:,1]) #Log-Differenzen sind die prozentualen Veränderungen
df_diff = df_.iloc[:,1].diff()

plt.rcParams['figure.figsize'] = [15, 2]

fig = sns.boxplot(data=df_diff, orient='h')

plt.title('Häufigkeiten der prozentualen Wertveränderungen', size=14)
plt.xlabel('')
plt.show()


# In[60]:


df.set_index('date')


# In[61]:


df.iloc[:,1:2]


# In[62]:


df.Date.max()


# In[63]:


df.Date.mean()


# In[64]:


df


# In[65]:


df['Date'] = df['Date'].where((df['Date'] != 0), 0.001)


# In[66]:


df['Date']


# In[67]:


df.iloc[:,0:2]


# In[68]:


df.columns = ['Date', 'Reviews_Count', 'listing_id', 'neighbourhood', 'neighbourhood_cleansed', 'neighbourhood_group_cleansed']


# In[69]:


df.iloc[:,1]


# In[70]:


i_iqr_faktor = 2
df_ = df.iloc[:,0:2].copy()
df_['Date'] = pd.to_datetime(df_['Date'], format='%Y-%m-%d', errors='ignore')
df_.iloc[:,1:] = np.log(df_.iloc[:,1:])
df_diff = df_.diff()


q25 = df_diff.iloc[:,1].quantile(0.25)
q75 = df_diff.iloc[:,1].quantile(0.75)

iqr = q75-q25

grenze_unten = q25 - (i_iqr_faktor*iqr)
grenze_oben = q75 + (i_iqr_faktor*iqr)
df_[((df_diff.iloc[:,1] < grenze_unten) | (df_diff.iloc[:,1] > grenze_oben))]




df_dates = df_[((df_diff.iloc[:,1] < grenze_unten) | (df_diff.iloc[:,1] > grenze_oben))]
df_dates = df_dates.reset_index()
df_dates = df_dates.iloc[:,0:2]

print('Bei den eingegebenen Daten und des IQR-Faktors sind Wochen, an denen die Wertveränderung < {0:.2f} oder > {1:.2f} war, auffällig.'.format(grenze_unten, grenze_oben))
print('Dies tritt an {} Tagen auf: '.format(len(df_dates)))
df_dates


# In[71]:


#Sind einzelne Jahre (Monate) besonders auffällig?
get_ipython().run_line_magic('matplotlib', 'inline')
df_dates['Jahr'] = df_dates['Date'].map(lambda x: x.strftime('%Y'))
plt.rcParams['figure.figsize'] = [15, 4]
df_dates.groupby('Jahr').size().plot(kind = 'bar')
plt.xlabel('Periode')
plt.ylabel('Anzahl')
plt.show()


# In[72]:


df_dates.groupby('Jahr').size()


# In[73]:


df_dates.groupby('Jahr').size().plot()


# In[74]:


#Zu untersuchende Daten auswählen
df_ts_clean = df_ts[(df_ts.index > '2013-12-31') & (df_ts.index < '2020-03-01')]
print('Insgesamt liegen {} zusammenhängende ähnliche Beobachtungen vor.'.format(len(df_ts_clean)))


# In[75]:


df_ts_clean.columns = ['Reviews_Count', 'listing_id', 'neighbourhood', 'neighbourhood_cleansed', 'neighbourhood_group_cleansed']


# In[76]:


df_ts_clean = df_ts_clean['Reviews_Count']


# In[77]:


df_ts_clean = pd.DataFrame(df_ts_clean)


# In[78]:


df_ts_clean


# In[79]:


#Zerteilung in Trainings- und Testdaten - Im Normalfall beginnt man mit dem Verhältnis 80:20
#Weil wir Tagesdaten haben und mit "klassischen" Zeitreihenanalyseverfahren schwerlich mehr als 60 Perioden mit hoher
#Genauigkeit schätzen kann, teilen wir 95:5.

i_split = int(0.85*len(df_ts_clean))

df_train, df_test = df_ts_clean.iloc[:i_split,:], df_ts_clean.iloc[i_split:,:]

print('Train und Test sind zusammen {} Einträge lang.'.format(len(df_train)+len(df_test)))
print('D.h., alle Forecasts müssen {} Perioden lang sein.'.format(len(df_test)))


# In[80]:


#Plot

plt.rcParams['figure.figsize'] = [15, 6]

plt.plot(df_train.index, df_train.values, label='Trainingsdaten')
plt.plot(df_test.index, df_test.values, label='Testdaten', color=colors[1])

plt.axvline(x = df_ts_clean.index[i_split], linewidth=2, color='grey', ls='--')
plt.legend(loc=2, fontsize=10)
plt.title('Anzahl der wöchentlichen Reviews für Airbnb Objekte in Berlin über die Zeit'.format(df_train.columns[0]), fontsize=14)
plt.xlabel('Zeit', fontsize=10)
plt.ylabel('Anzahl wöchentlicher Reviews', fontsize=10)
#plt.show()

plt.savefig('Einteilung_Training_Test.png')


# In[81]:


vals = np.asarray(df_train.values)
y_hat = df_test.copy()
y_hat['naiv'] = vals[-1][0]


# In[82]:


df_train.values


# In[83]:


y_hat


# In[84]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = [15, 6]

plt.plot(df_train.index, df_train.values, label='Traininsdaten')
plt.plot(df_test.index, df_test.values, label='Testdaten')
plt.plot(y_hat.index, y_hat['naiv'], label='Naiver Forecast')
plt.legend(loc='best')
plt.title("Naiver Forecast")
plt.show()


# In[85]:


#Importieren der Fehler-Schätzstatistiken aus sklearn
from sklearn.metrics import mean_squared_error
from math import sqrt #Importieren einer Wurzel-Funktion aus math


# ### Root Mean Squared Error - RMSE
# <font size="4">
# Tipp: https://en.wikipedia.org/wiki/Root-mean-square_deviation <br>
# 
# 
# \begin{align}
# \text{RMSE} \; &= \sqrt{\frac{\sum_{t=1}^T (\hat{y}_t - y_t)^2}{T}} 
# \end{align} <br>
# 
# Seltener schaut man auch einfach auch den durchschnittlichen Fehler.<br>
# 
# \begin{align*}
# \text{ME} \; &= \frac{\sum_{t=1}^T (\hat{y}_t - y_t)}{T}
# \end{align*} <br>
# 
# </font>

# In[86]:


df_test


# In[87]:


rmse = sqrt(mean_squared_error(df_test, y_hat.naiv))
me = (df_test.iloc[:,0] - y_hat['naiv']).sum() / len(df_test)
print('Für den naiven Forecast ergeben sich ein ME: {0:.4f} und ein RMSE: {1:.4f}.'.format(me,rmse))


# In[88]:


# RMSE = 0 -> a perfect fit


# <font size=4>
# Auch kann man die Güte es Forecasts an bestimmten Eigenschaften der Residuen (Schätzfehler) ablesen.<br>
# 
# \begin{align*}
# \text{Residuen} \; &= \hat{y}_t - y_t \quad = \epsilon_t
# \end{align*}
# 
# </font>

# In[89]:


residuen = (df_test.iloc[:,0] - y_hat['naiv'])
print('Die Residuen haben für diesen Forecast folgene Standardabweichung: {0:.4f}.'.format(residuen.std()))
stdres = residuen.std()


# In[90]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = [15, 12]
plt.subplot(2,1,1)
residuen.hist(bins=50, density=True)
plt.subplot(2,1,2)
plt.plot(df_test.index, residuen.values, label='Residuen', linewidth=2)
plt.legend(loc=2)
plt.title("Residuen")
plt.show()


# In[91]:


#Anlegen einer Tabelle, um später die Güte verschiedener Verfahren miteinander vergleichen zu können.
#ACHTUNG: Mit dieser Zeile wird ein leerer DataFrame erzeugt.
df_Fehler = pd.DataFrame(columns=['Methode', 'ME', 'RMSE', 'StdRes'])


# In[92]:


#Einfügen der Güte-Maße
df_Fehler = df_Fehler.append({'Methode': 'Naives Fortschreiben', 'ME': me, 'RMSE': rmse, 'StdRes': stdres},
ignore_index=True)


# In[93]:


df_Fehler


# ### Moving Average (Gleitender Mittelwert)

# In[94]:


#Mit rolling und mean() kann man den gleitenden Mittelwert ganz einfach erzeugen. 
#Mit einer Schleife und append schreibt man die Werte fort.

n = 60 #Bspw. den gleitenden Durchschnitt über alle Tage des letzten Quartals.
df_train.rolling(n).mean().iloc[-1][0]


# In[95]:


n = 60
df_mav = df_train.copy()
for i in range(len(df_test)):
    df_mav = df_mav.append({df_mav.columns[0] : df_mav.rolling(n).mean().iloc[-1][0]}, ignore_index=True)


# In[96]:


y_hat = df_test.copy()
y_hat_mav = df_mav.iloc[-len(df_test):].copy()
y_hat['mav'] = y_hat_mav.values


# In[97]:


y_hat_mav


# In[98]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = [15, 6]

plt.plot(df_train.index, df_train.values, label='Trainingsdaten')
plt.plot(df_test.index, df_test.values, label='Testdaten')
plt.plot(df_test.index, y_hat['mav'].values, label='Moving Average Forecast')

plt.title('Gleitender Durchschnitt')
plt.legend(loc='best')
plt.show()

#plt.savefig('Moving_Average.png')


# In[99]:


rmse = sqrt(mean_squared_error(df_test, y_hat.mav))
me = (df_test.iloc[:,0] - y_hat['mav']).sum() / len(df_test)
print('Für den naiven Forecast ergeben sich ein ME: {0:.4f} und ein RMSE: {1:.4f}.'.format(me,rmse))

residuen = (df_test.iloc[:,0] - y_hat['mav'])
print('Die Residuen haben für diesen Forecast folgene Standardabweichung: {0:.4f}.'.format(residuen.std()))
stdres = residuen.std()


# In[100]:


df_Fehler = df_Fehler.append({'Methode': 'Moving Average', 'ME': me, 'RMSE': rmse, 'StdRes': stdres},
ignore_index=True)


# In[101]:


df_Fehler


# In[102]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = [15, 12]
plt.suptitle('Moving Average')
plt.subplot(2,1,1)
residuen.hist(bins=50, density=True)
plt.subplot(2,1,2)
plt.plot(df_test.index, residuen.values, label='Residuen', linewidth=2)
plt.legend(loc=2)
plt.title("Residuen - Moving Average")
plt.show()


# <h3>Einfache (naive) exponentielle Glättung </h3><br>
# <font size=4>
# Für die einfache exponentielle Glättung gilt: <br><br>
# \begin{align}
# \hat{y}_{t+1} &= \hat{y}_{t} + \alpha ( y_t - \hat{y}_{t}) 
# \end{align} <br>
# 
# Weil $\alpha$ zwischen 0 und 1 liegen muss, kann man die Gleichung umschreiben: <br><br>
# \begin{align}
# \hat{y}_{t+1} &= \alpha y_t + (1-\alpha) \hat{y}_{t} 
# \end{align} <br>
# 
# Und das lässt sich schreiben als: <br><br>
# \begin{align}
# \hat{y}_{t+1} &= \alpha y_{t} + \alpha(1-\alpha) y_{t-1} + \alpha(1-\alpha)^2 y_{t-2} + \cdots + \alpha(1-\alpha)^{t-1} y_{1} + \alpha(1-\alpha)^{t} \hat{y}_{1} 
# \end{align} <br>
# </font>

# In[105]:


#Forecasts mit verschiedenen Glättungsparametern erzeugen
fit1 = SimpleExpSmoothing(df_train).fit(smoothing_level=0.25,optimized=False)
fcast1 = fit1.forecast(len(df_test)).rename(r'$\alpha=0.25$')


fit2 = SimpleExpSmoothing(df_train).fit(smoothing_level=0.50,optimized=False)
fcast2 = fit2.forecast(len(df_test)).rename(r'$\alpha=0.50$')


fit3 = SimpleExpSmoothing(df_train).fit()
fcast3 = fit3.forecast(len(df_test)).rename(r'$\alpha=%s$'%round(fit3.model.params['smoothing_level'],1))


# In[106]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = [15, 6]

fcast1.plot(legend=True, color=colors[0], ls='--')
fit1.fittedvalues.plot(color=colors[0])

fcast2.plot(legend=True, color=colors[1], ls='--')
fit2.fittedvalues.plot(color=colors[1])

fcast3.plot(legend=True, color=colors[2], ls='--')
fit3.fittedvalues.plot(color=colors[2])

plt.title('Anpassung der einfachen exponentiellen Glättung auf {}'.format(df_train.columns[0]), fontsize=14)
plt.xlabel('Zeit', fontsize=10)
plt.ylabel('Anzahl Reviews', fontsize=10)

plt.show()


# In[107]:


fit3.params


# In[108]:


rmse = sqrt(mean_squared_error(df_test, fcast1.values))
me = (df_test.iloc[:,0] - fcast1.values).sum() / len(df_test)
residuen = (df_test.iloc[:,0] - fcast1.values)
stdres = residuen.std()
print('Forecast 1 - RMSE: {0:.4f}, ME: {1:.4f}, StdRes: {2:.4f}'.format(rmse, me, stdres))


# In[109]:


df_Fehler = df_Fehler.append({'Methode': 'EinfExpGlätt 0.25', 'ME': me, 'RMSE': rmse, 'StdRes': stdres},
ignore_index=True)


# In[110]:


rmse = sqrt(mean_squared_error(df_test, fcast2.values))
me = (df_test.iloc[:,0] - fcast2.values).sum() / len(df_test)
residuen = (df_test.iloc[:,0] - fcast2.values)
stdres = residuen.std()
print('Forecast 2 - RMSE: {0:.4f}, ME: {1:.4f}, StdRes: {2:.4f}'.format(rmse, me, stdres))


# In[111]:


df_Fehler = df_Fehler.append({'Methode': 'EinfExpGlätt 0.5', 'ME': me, 'RMSE': rmse, 'StdRes': stdres},
ignore_index=True)


# In[112]:


rmse = sqrt(mean_squared_error(df_test, fcast3.values))
me = (df_test.iloc[:,0] - fcast3.values).sum() / len(df_test)
residuen = (df_test.iloc[:,0] - fcast3.values)
stdres = residuen.std()
print('Forecast 3 - RMSE: {0:.4f}, ME: {1:.4f}, StdRes: {2:.4f}'.format(rmse, me, stdres))


# In[113]:


df_Fehler = df_Fehler.append({'Methode': 'EinfExpGlätt 1', 'ME': me, 'RMSE': rmse, 'StdRes': stdres},
ignore_index=True)


# In[114]:


df_Fehler


# <h3>Exponentielle Glättung mit 'Level' und 'Trend' (Holt-Verfahren)</h3><br>
# <font size=4>
# Das Holt-Verfahren nennt man auch zweifache exponentielle Glättung:<br><br>
# 
# \begin{align*}
# \text{Level: } \; \quad \ell_t &= \alpha y_t + (1-\alpha) (\ell_{t-1} + b_{t-1}) \\ \\
# \text{Growth: } \; \quad b_t &= \beta^* (\ell_t - \ell_{t-1}) + (1-\beta^*) b_{t-1} \\ \\
# \text{Forecast: } \; \hat{y}_{t+h|t} &= \ell_t + b_t h \\ \\
# \end{align*}
# 
# Manchmal wird der Trend auch gedämpft. <br><br>
# 
# \begin{align*}
# \text{Level: } \; \quad \ell_t &= \alpha y_t + (1-\alpha) (\ell_{t-1} + \phi b_{t-1}) \\ \\
# \text{Growth: } \; \quad b_t &= \beta^* (\ell_t - \ell_{t-1}) + (1-\beta^*) b_{t-1} \\ \\
# \text{Forecast: } \; \hat{y}_{t+h|t} &= \ell_t + (\phi + \phi^2 + \cdots + \phi^h) b_t h \\ \\
# \end{align*}
# </font>

# In[115]:


#Schätzen der zweifachen exponentiellen Glättung
fit4 = Holt(df_train.iloc[:,0]).fit(smoothing_level=0.8, smoothing_slope=0.2, optimized=False)
fcast4 = fit4.forecast(len(df_test)).rename("Holt's linear trend")

fit5 = Holt(df_train.iloc[:,0], exponential=True).fit(smoothing_level=0.8, smoothing_slope=0.2, optimized=False)
fcast5 = fit5.forecast(len(df_test)).rename("Exponential trend")

fit6 = Holt(df_train.iloc[:,0], damped=True).fit()
fcast6 = fit6.forecast(len(df_test)).rename("Additive damped trend")


# In[116]:


fit6.params


# In[117]:


#Plotten der Ergebnisse

get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = [15, 6]

fcast4.plot(legend=True, color=colors[3], ls='--')
fit4.fittedvalues.plot(color=colors[3])

fcast5.plot(legend=True, color=colors[4], ls='--')
fit5.fittedvalues.plot(color=colors[4])

fcast6.plot(legend=True, color=colors[5], ls='--')
fit6.fittedvalues.plot(color=colors[5])

plt.title('Anpassung der zweifachen exponentiellen Glättung auf {}'.format(df_train.columns[0]), fontsize=14)
plt.xlabel('Zeit', fontsize=10)
plt.ylabel('Anzahl der Reviews', fontsize=10)

plt.show()


# In[118]:


rmse = sqrt(mean_squared_error(df_test, fcast4.values))
me = (df_test.iloc[:,0] - fcast4.values).sum() / len(df_test)
residuen = (df_test.iloc[:,0] - fcast4.values)
stdres = residuen.std()
print('Forecast 4 - RMSE: {0:.4f}, ME: {1:.4f}, StdRes: {2:.4f}'.format(rmse, me, stdres))

df_Fehler = df_Fehler.append({'Methode': 'Holt LT', 'ME': me, 'RMSE': rmse, 'StdRes': stdres},
ignore_index=True)

rmse = sqrt(mean_squared_error(df_test, fcast5.values))
me = (df_test.iloc[:,0] - fcast5.values).sum() / len(df_test)
residuen = (df_test.iloc[:,0] - fcast5.values)
stdres = residuen.std()
print('Forecast 5 - RMSE: {0:.4f}, ME: {1:.4f}, StdRes: {2:.4f}'.format(rmse, me, stdres))

df_Fehler = df_Fehler.append({'Methode': 'Holt ET', 'ME': me, 'RMSE': rmse, 'StdRes': stdres},
ignore_index=True)

rmse = sqrt(mean_squared_error(df_test, fcast6.values))
me = (df_test.iloc[:,0] - fcast6.values).sum() / len(df_test)
residuen = (df_test.iloc[:,0] - fcast6.values)
stdres = residuen.std()
print('Forecast 6 - RMSE: {0:.4f}, ME: {1:.4f}, StdRes: {2:.4f}'.format(rmse, me, stdres))

df_Fehler = df_Fehler.append({'Methode': 'Holt add, damped trend', 'ME': me, 'RMSE': rmse, 'StdRes': stdres},
ignore_index=True)


# In[119]:


df_Fehler


# <h3>Exponentielle Glättung mit 'Level', 'Trend' und 'Saisoneffekten' (Holt-Winters)</h3><br>
# <font size=4>
# Das Holt-Winters-Verfahren nennt man (spezielle) dreifache exponentielle Glättung:<br><br>
# 
# \begin{align*}
# \text{Level: } \; \quad \ell_t &= \alpha \frac{y_t}{s_{t-m}} + (1-\alpha) (\ell_{t-1} + b_{t-1}) \\\\
# \text{Growth: } \; \quad b_t &= \beta^* (\ell_t - \ell_{t-1}) + (1-\beta^*) b_{t-1} \\\\
# \text{Seasonal: } \; \quad s_t &= \gamma \frac{y_t}{\ell_{t-1} + b_{t-1}} + (1-\gamma) s_{t-m} \\\\
# \text{Forecast: }\; \hat{y}_{t+h|t} &= (\ell_t + b_t h ) s_{t-m+h_m^+} \\
# \end{align*}
# </font>

# In[120]:


len(df_test)


# In[121]:


len(df_train)


# In[122]:


#seasonal_periods = 2 weil Sommer und Winter Trends berücksichtigt werden sollen
#Schätzen von Holt-Winters
fit7 = ExponentialSmoothing(df_train.iloc[:,0], seasonal_periods=52, trend='add', seasonal='add').fit()
fcast7 = fit7.forecast(len(df_test)).rename("Holt-Winters Additive")

fit8 = ExponentialSmoothing(df_train.iloc[:,0], seasonal_periods=52, trend='add', seasonal='mul').fit()
fcast8 = fit8.forecast(len(df_test)).rename("Holt-Winters Multiplikativ")


# In[129]:


#Plotten der Ergebnisse

get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = [15, 6]

plt.plot(df_test.index, df_test.values, label='Testdaten')
fcast7.plot(legend=False, color=colors[1], ls='--')
fit7.fittedvalues.plot(color=colors[2])

#fcast8.plot(legend=True, color=colors[3], ls='--')
#fit8.fittedvalues.plot(color=colors[4])

plt.title('Anpassung der dreifachen exponentiellen Glättung auf {}'.format(df_train.columns[0]), fontsize=14)
plt.xlabel('Zeit', fontsize=10)
plt.ylabel('Anzahl der Reviews', fontsize=10)

#plt.show()

plt.savefig('Anpassung_dreifache_exponetielle_Glättung.png')


# In[127]:



fig, ax = plt.subplots()
lines = ax.plot(data)
ax.legend(custom_lines, ['Cold', 'Medium', 'Hot'])


# In[128]:


#Plotten der Ergebnisse

get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = [15, 6]

#fcast7.plot(legend=True, color=colors[1], ls='--')
#fit7.fittedvalues.plot(color=colors[2])

plt.plot(df_test.index, df_test.values, label='Testdaten')
fcast8.plot(legend=True, color=colors[3], ls='--')
#fit8.fittedvalues.plot(color=colors[4])
fcast7.plot(legend=True, color=colors[2], ls='--')
plt.legend(['Testdaten', 'Multiplikativ nach Holt-Winters', 'Additiv nach Holt-Winters'], loc = 'best', fontsize=14)

plt.title('Anpassung der dreifachen exponentiellen Glättung auf {}'.format(df_train.columns[0]), fontsize=16)
plt.xlabel('Zeit', fontsize=14)
plt.ylabel('Anzahl der wöchentlichen Reviews', fontsize=14)

#plt.show()

plt.savefig('Holt_Winters_prognose.png')


# In[130]:


rmse = sqrt(mean_squared_error(df_test, fcast7.values))
me = (df_test.iloc[:,0] - fcast7.values).sum() / len(df_test)
residuen = (df_test.iloc[:,0] - fcast7.values)
stdres = residuen.std()
print('Forecast 7 - RMSE: {0:.4f}, ME: {1:.4f}, StdRes: {2:.4f}'.format(rmse, me, stdres))

df_Fehler = df_Fehler.append({'Methode': 'Holt Winters add', 'ME': me, 'RMSE': rmse, 'StdRes': stdres},
ignore_index=True)

rmse = sqrt(mean_squared_error(df_test, fcast8.values))
me = (df_test.iloc[:,0] - fcast8.values).sum() / len(df_test)
residuen = (df_test.iloc[:,0] - fcast8.values)
stdres = residuen.std()
print('Forecast 8 - RMSE: {0:.4f}, ME: {1:.4f}, StdRes: {2:.4f}'.format(rmse, me, stdres))

df_Fehler = df_Fehler.append({'Methode': 'Holt Winters mult', 'ME': me, 'RMSE': rmse, 'StdRes': stdres},
ignore_index=True)


# In[131]:


df_Fehler


# ## Faktor-Dekomposition
# ### Wie bekomme ich heraus, ob und welche Saisonalität in den Daten vorliegt?

# In[132]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = [15, 8]
sm.tsa.seasonal_decompose(df_train).plot()
#plt.show()

plt.savefig('Faktor_Dekomp.png')


# ## Autoregressive Prozesse AR, ARMA und ARIMA

# <font size=4>
# Tipp: Eine Einführung in ARIMA-Modelle findet sich hier:<br>
# 
# https://towardsdatascience.com/unboxing-arima-models-1dc09d2746f8
# 
# </font>

# <h3>Autoregressive Prognose-Modelle</h3><br>
# <font size=4>
# Bei ARIMA-Modellen wird eine Zeitreihe durch einen verzögerten Term der zu erklärenden Variable (AR) und einen Moving-Average-Term (MA) erklärt. Unter bestimmten Umständen muss das Modell insgesamt als integrierte Gleichung (I; ein- oder mehrfach differenziert) optimiert werden.<br><br>
# Dafür gilt es, die optimale Anzahl an Verzögerungen (p), das optimale Momentum (q) und, ggf., das optimale Differenz-Niveau (d) gefunden werden.<br>
# 
# \begin{align*}
# \text{Forecast:} \; \quad \hat{y}_{t} &= \mu + \phi_{1}y{t-1}+\cdots+\phi_{p}y_{t-p}-\theta_{1}\epsilon_{t-1}-\cdots-\theta_{q}\epsilon_{t-q} \\\\
# \text{Für d = 0:} \; \quad y_t &= Y_t \\\\
# \text{Für d = 1:} \; \quad y_t &= Y_t - Y_{t-1} \\\\
# \text{Für d = 2:} \; \quad y_t &= (Y_t - Y_{t-1})-(Y_{t-1} - Y_{t-2})
# \end{align*}
# </font>

# In[133]:


from statsmodels.tsa.arima_model import ARIMA
warnings.filterwarnings("ignore") # specify to ignore warning messages


# In[134]:


#Eigene Auto-Arima-Funktion
#Erzeugen einer Liste von allen Parametern, die getestet werden sollen.
p = range(0, 6)
d = range(0, 3)
q = range(0, 5)
pdq = list(itertools.product(p, d, q))
pdq


# In[135]:


ts = df_train.copy()


# In[136]:


AIC = []
ARIMA_model = []
i = 0

for param in pdq:
    try:
        mod = ARIMA(ts, order=param)
        results = mod.fit()
        print() 
        print(i, ": ",'ARIMA{} - AIC:{}'.format(param, results.aic), end='\r')
        AIC.append(results.aic)
        ARIMA_model.append([param])
    except:
        continue
        i = i + 1


# In[137]:


print('Das AIC nimmt mit {} für das Modell ARIMA{} den kleinsten Wert an.'.format(min(AIC), ARIMA_model[AIC.index(min(AIC))][0]))


# In[138]:


mod = ARIMA(ts,order=ARIMA_model[AIC.index(min(AIC))][0])
fit9 = mod.fit()


# In[139]:


#Achtung, wenn d != 0, dann werden Differenzen vorhergesagt.
fcast9 = fit9.predict(start=len(ts), end=len(ts)+len(df_test)-1, dynamic=False)
fcast9.head()


# In[140]:


y_hat_ = fcast9.copy()
y_hat_[0] = y_hat_[0] + ts.iloc[-1][0]
y_hat_ = np.cumsum(y_hat_.values)
fcast9[:] = y_hat_


# In[141]:


#Plotten der Ergebnisse

get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = [15, 6]

plt.plot(df_train.index, df_train.values, label='Trainingsdaten')
plt.plot(fcast9.index, fcast9.values, label='ARIMA', color=colors[2])

plt.legend(loc=2)
plt.title('Anpassung von autoregressiven Modellen auf {}'.format(df_train.columns[0]), fontsize=14)
plt.xlabel('Zeit', fontsize=10)
plt.ylabel('Anzahl der wöchentlichen Reviews', fontsize=10)

plt.show()


# In[142]:


rmse = sqrt(mean_squared_error(df_test, fcast9.values))
me = (df_test.iloc[:,0] - fcast9.values).sum() / len(df_test)
residuen = (df_test.iloc[:,0] - fcast9.values)
stdres = residuen.std()
print('Forecast 7 - RMSE: {0:.4f}, ME: {1:.4f}, StdRes: {2:.4f}'.format(rmse, me, stdres))

df_Fehler = df_Fehler.append({'Methode': 'ARIMA(2,2,4)', 'ME': me, 'RMSE': rmse, 'StdRes': stdres},
ignore_index=True)


# In[143]:


df_Fehler


# ### Analyse nach Bezirken

# In[144]:


df2.columns


# In[145]:


# Erstellen des Datensatzes df_Bezirke
df_Bezirke = df2[['listing_id', 'neighbourhood', 'date']]


# In[146]:


df_Bezirke


# In[147]:


df_Bezirke = df_Bezirke.groupby(['neighbourhood', 'date']).count()


# In[148]:


df_Bezirke.sort_values('date')


# In[149]:


df_Bezirke = df_Bezirke.reset_index()


# In[150]:


df_Bezirke = df_Bezirke.pivot_table('listing_id', 'date', 'neighbourhood')


# In[151]:


df_Bezirke.sum().sort_values(ascending=False)[:10]


# In[152]:


top_10 = df_Bezirke.sum().sort_values(ascending=False)[:10].index
top_5 = df_Bezirke.sum().sort_values(ascending=False)[:5].index


# In[153]:


top_10


# In[154]:


df_Bezirke[top_10]


# In[155]:


df_Bezirke[top_10].plot()


# In[156]:


df_Bezirke.info()


# In[157]:


df_Bezirke = df_Bezirke.reset_index()
df_Bezirke['date'] = pd.to_datetime(df_Bezirke['date'])


# In[158]:


df_Bezirke = df_Bezirke.resample('W', on='date').sum()


# In[159]:


#sns.set_style??


# In[160]:


#sns.set_style??


# In[161]:


sns.set_style("darkgrid")


# In[163]:


df_Bezirke[top_10].plot()
plt.rcParams['figure.figsize'] = [20, 8]

plt.title('Top 10 Bezirke nach Anzahl der Reviews auf Airbnb')
plt.xlabel('Zeit')
plt.ylabel('Reviews')
#plt.show()

plt.savefig('Top10_nach_Bezirken.png')


# In[164]:


df_Bezirke_aktuell = df_Bezirke[(df_Bezirke.index > '2014-01-01')]


# In[165]:


df_Bezirke_aktuell[top_10]


# In[166]:


df_Bezirke_aktuell[top_10].plot()
plt.rcParams['figure.figsize'] = [20, 8]

plt.title('Top 10 Bezirke nach Anzahl der Reviews auf Airbnb', fontsize = 16)
plt.xlabel('Zeit')
plt.ylabel('wöchentliche Reviews', fontsize = 14)
#plt.show()

plt.savefig('Top_10.png')


# In[167]:


df_Bezirke[top_5].plot()
plt.rcParams['figure.figsize'] = [15, 6]



plt.title('Top 5 Bezirke nach Anzahl der Reviews auf Airbnb')
plt.xlabel('Zeit')
plt.ylabel('Reviews')
plt.show()


# In[168]:


plt.rcParams['figure.figsize'] = [20, 16]
fig = plt.figure()
df_Bezirke[top_10].plot(subplots=True)
plt.show()


# In[ ]:





# In[ ]:




