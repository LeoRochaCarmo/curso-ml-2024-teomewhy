#%%

import pandas as pd

df = pd.read_parquet('../data/dados_clones.parquet')

df.head()

#%%
# Como podemos descobrir onde está o problema?
# (Estatística Descritiva)

df.groupby('Status ')[['Estatura(cm)', 'Massa(em kilos)']].mean()

#%%

df['status_bool'] = df['Status '] == 'Apto'
df

#%%

# Taxa de aptos pela distância ombro a ombro
df.groupby('Distância Ombro a ombro')['status_bool'].mean()

#%%

# Taxa de aptos pelo tamanho do crânio	
df.groupby('Tamanho do crânio')['status_bool'].mean()

#%%

# Taxa de aptos pelo tamanho dos pés
df.groupby('Tamanho dos pés')['status_bool'].mean()

#%%

# Taxa de aptos pelo general Jedi
df.groupby('General Jedi encarregado')['status_bool'].mean()


#%%

# Determinando quais serão as features
features = [
    'Massa(em kilos)',
    'Estatura(cm)', 
    'Distância Ombro a ombro', 
    'Tamanho do crânio', 
    'Tamanho dos pés'
]

# Determinando apenas as features categóricas (variáveis qualitativas)
cat_features = [
    'Distância Ombro a ombro', 
    'Tamanho do crânio', 
    'Tamanho dos pés'
]

X = df[features]
y = df['Status ']

#%%

# Transformando variáveis categóricas em numéricas
from feature_engine import encoding

onehot = encoding.OneHotEncoder(variables=cat_features)
onehot.fit(X)

X = onehot.transform(X)
X

# %%
# Como podemos fazer a máquina aprender?
# (Machine Learning)

from sklearn import tree
import matplotlib.pyplot as plt

arvore = tree.DecisionTreeClassifier(max_depth=3)
arvore.fit(X,y)

plt.figure(dpi=300)
tree.plot_tree(arvore, 
               class_names=arvore.classes_, 
               feature_names=X.columns, 
               filled=True)

plt.show()

# Conclusão Final:
'''
Um lote de clones apresenta defeito de fabricaçãoem
determinadas medidas (massa e estatura).
'''

# Ações necessárias:
'''
Evitar a compra de clones dentro das faixas indicadas.
Caso necessário, distribuir entre outros generais.
'''