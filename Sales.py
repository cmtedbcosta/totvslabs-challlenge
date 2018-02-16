# -*- coding: utf-8 -*-
"""
### JSON Metadata
{
  "complemento": {
    "valorTotal": 24.9
  },
  "dets": [
    {
      "nItem": "1",
      "prod": {
        "indTot": "1",
        "qCom": 1.0,
        "uCom": "UN",
        "vProd": 3.5,
        "vUnCom": 3.5,
        "xProd": "AGUA"
      }
    },
    {
      "nItem": "2",
      "prod": {
        "indTot": "1",
        "qCom": 0.312,
        "uCom": "KG",
        "vProd": 21.4,
        "vUnCom": 68.6,
        "xProd": "BUFFET"
      }
    }
  ],
  "emit": {
    "cnpj": "01.234.567/0001-89",
    "enderEmit": {
      "fone": "1-650-933-4902",
      "xBairro": "",
      "xLgr": "650 Castro St. unit 210",
      "xMun": "Mountain View",
      "xPais": "United States",
      "uf": "CA"
    },
    "xFant": "TOTVS Labs"
  },
  "ide": {
    "dhEmi": {
      "$date": "2016-01-05T12:01:54.000Z"
    },
    "natOp": "VENDA"
  },
  "infAdic": {
    "infCpl": "Mesa 2"
  },
  "total": {
    "icmsTot": {
      "vDesc": 0.0,
      "vFrete": 0.0,
      "vOutro": 0.0,
      "vProd": 24.9,
      "vSeg": 0.0,
      "vTotTrib": 2.53,
      "vbc": 0.0,
      "vbcst": 0.0,
      "vcofins": 0.0,
      "vicms": 0.0,
      "vicmsDeson": 0.0,
      "vii": 0.0,
      "vipi": 0.0,
      "vnf": 24.9,
      "vpis": 0.0,
      "vst": 0.0
    }
  },
  "versaoDocumento": 1.0
}
"""

# Imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from pandas.io.json import json_normalize
from sklearn.svm import SVR

# Import data from JSON
df_notas = json_normalize(json.load(open('sample.txt')))

# Treats date
df_notas['date'] = pd.to_datetime(df_notas['ide.dhEmi.$date'])
df_notas.drop('ide.dhEmi.$date', axis=1, inplace=True)
df_notas['dayofweek'] = df_notas['date'].dt.dayofweek
df_notas['day'] = df_notas['date'].dt.day
df_notas['time'] = df_notas['date'].apply(lambda x: 'lunch' if x.hour < 15 else 'dinner')
df_notas['time'] = df_notas['time'].astype('category')

# Processes Items
itens = []

for index, row in df_notas.iterrows():
    for item in row['dets']:      
        prod = item['prod']
        indTot = prod['indTot']
        qCom = prod['qCom']
        uCom = prod['uCom']
        vProd = prod['vProd']
        vUnCom = prod['vUnCom']
        xProd = prod['xProd']
        itens.append([index,indTot,qCom,uCom,vProd,vUnCom,xProd])

df_itens = pd.DataFrame(itens,columns=['index_nf','indTot','qCom','uCom','vProd','vUnCom','xProd'])

# drop orignial column after processing items
df_notas.drop('dets', axis=1, inplace=True)

# Explore through data
df_notas['complemento.valorTotal'].describe()
df_itens['vUnCom'].describe()

# Plot which day of week sells the most
# Monday=0 ... Sunday=6
dayofweek_data = df_notas.groupby('dayofweek')
dayofweek_data.count().plot(y='complemento.valorTotal', legend=False)
plt.xlabel('Day of Week')
plt.ylabel('Receipt count')
plt.grid(True)

# Plot wich time sells the  most
data = pd.concat([df_notas['complemento.valorTotal'], df_notas['time']], axis=1)
fig = sns.boxplot(x='time', y="complemento.valorTotal", data=data)
fig.axis(ymin=0, ymax=175)
plt.xlabel('Time')
plt.ylabel('Total value')
plt.show()

# Plot wich table sells the most
table_data = df_notas.groupby('infAdic.infCpl')
table_data.count().plot(y='complemento.valorTotal', legend=False)
plt.xlabel('Tables')
plt.ylabel('Receipt count')
plt.grid(True)

# Plot sales by day
day_data = df_notas.groupby('day')
day_data.count().plot(y='complemento.valorTotal', legend=False)
#day_data.sort_values('day')
plt.xlabel('Day')
plt.ylabel('Receipt count')
plt.grid(True)

# Histogram
pd.DataFrame.hist(df_notas[['complemento.valorTotal']], bins=5)
plt.xlabel('Total Value')
plt.ylabel('Frequency')
plt.show()

# Plot which products sell the most
itens_data = df_itens.groupby('xProd')
itens_data['vUnCom'].describe()
itens_data.sum().plot(y='vUnCom', legend=False)
plt.xlabel('Products')
plt.ylabel('Value')
plt.grid(True)

# Histogram
pd.DataFrame.hist(df_itens[['vUnCom']], bins=5)
plt.xlabel('Total Value')
plt.ylabel('Frequency')
plt.show()

# Check for missing values
df_notas.isnull().all().all()
df_itens.isnull().all().all()

# Drop insignificant and empty columns after data analysis
df_notas.drop('emit.cnpj', axis=1, inplace=True) #all the same value
df_notas.drop('emit.enderEmit.fone', axis=1, inplace=True) #all the same value
df_notas.drop('emit.enderEmit.xBairro', axis=1, inplace=True) #empty
df_notas.drop('emit.enderEmit.xLgr', axis=1, inplace=True) #all the same value
df_notas.drop('emit.enderEmit.xPais', axis=1, inplace=True) #all the same value
df_notas.drop('emit.enderEmit.uf', axis=1, inplace=True) #all the same value
df_notas.drop('emit.enderEmit.xMun', axis=1, inplace=True) #all the same value
df_notas.drop('emit.xFant', axis=1, inplace=True) #all the same value
df_notas.drop('ide.natOp', axis=1, inplace=True) #all the same value
df_notas.drop('total.icmsTot.vFrete', axis=1, inplace=True) #all the same value
df_notas.drop('total.icmsTot.vOutro', axis=1, inplace=True) #all the same value
df_notas.drop('total.icmsTot.vSeg', axis=1, inplace=True) #all the same value
df_notas.drop('total.icmsTot.vbc', axis=1, inplace=True) #all the same value
df_notas.drop('total.icmsTot.vbcst', axis=1, inplace=True) #all the same value
df_notas.drop('total.icmsTot.vcofins', axis=1, inplace=True) #all the same value
df_notas.drop('total.icmsTot.vicms', axis=1, inplace=True) #all the same value
df_notas.drop('total.icmsTot.vicmsDeson', axis=1, inplace=True) #all the same value
df_notas.drop('total.icmsTot.vii', axis=1, inplace=True) #all the same value
df_notas.drop('total.icmsTot.vipi', axis=1, inplace=True) #all the same value
df_notas.drop('total.icmsTot.vpis', axis=1, inplace=True) #all the same value
df_notas.drop('total.icmsTot.vst', axis=1, inplace=True) #all the same value
df_notas.drop('versaoDocumento', axis=1, inplace=True) #all the same value
df_notas.drop('versaoDocumento.$numberLong', axis=1, inplace=True) #all the same value
df_itens.drop('indTot', axis=1, inplace=True) #all the same value

# Prepare data for processing
day_data = df_notas.groupby('day')
data = day_data.aggregate(np.sum)
data['valor'] = data['complemento.valorTotal']
data.loc[10] = 0 # adding sunday to data
data.loc[17] = 0 # adding sunday to data
data['day'] = data.index
data = data.sort_values('day')

# Split X and Y
X = np.array(data['day'].values.reshape(-1, 1))
y = np.array(data['valor'].values.reshape(-1, 1)).ravel()

# Set prediction X
X_predict = np.array(range(24,31)).reshape(-1, 1)

# Run SVR algorithm
svr_rbf = SVR(kernel='rbf', C=1e8, gamma=0.1)
rbf = svr_rbf.fit(X, y)
y_rbf = rbf.predict(X)
y_predict = rbf.predict(X_predict)

# Check algorithm accuracy and do hyperparameter tunning
lw = 2
plt.scatter(X, y, color='darkorange', label='data')
plt.plot(X, y_rbf, color='navy', lw=lw, label='RBF model')
plt.plot(X_predict, y_predict, color='red', lw=lw, label='Prediction')
plt.xlabel('data')
plt.ylabel('target')
plt.title('Support Vector Regression')
plt.legend()
plt.show()

# Manually set the prediction to 0, since day 24 is a sunday
y_predict[0] = 0

# Forecasted value for sales
print('Forecasted value: $ %.2f' % y_predict.sum())
