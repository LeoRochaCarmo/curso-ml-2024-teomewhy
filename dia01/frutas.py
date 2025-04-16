# %%
import pandas as pd

df = pd.read_excel("../data/dados_frutas.xlsx")
df

# %%
## Como aplicar o método do slide para descobrir a fruta?
filtro_redonda = df['Arredondada'] == 1
filtro_suculenta = df['Suculenta'] == 1
filtro_vermelha = df['Vermelha'] == 1
filtro_doce = df['Doce'] == 1
df[filtro_redonda & filtro_suculenta & filtro_vermelha & filtro_doce]

# %%
## Como podemos fazer a máquina aprender?

from sklearn import tree

features = ['Arredondada', 'Suculenta', 'Vermelha', 'Doce']	
target = 'Fruta'

x = df[features]
y = df[target]

#%%

arvore = tree.DecisionTreeClassifier(random_state=42)
arvore.fit(x, y)

#%%
import matplotlib.pyplot as plt

plt.figure(dpi=200)

tree.plot_tree(arvore, 
               class_names=arvore.classes_, 
               feature_names=features, 
               filled=True)

plt.show()

#%%

# ['Arredondada', 'Suculenta', 'Vermelha', 'Doce']

# retorna apenas o nome de UMA fruta (por empate retorna o de ordem alfabética)
arvore.predict([[0,1,1,1]])

#%%

# retorna uma lista com a probabilidade de cada fruta
probas = arvore.predict_proba([[1,1,1,1]])[0]
pd.Series(probas, index=arvore.classes_)

#%%
