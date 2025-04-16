#%%
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt

df = pd.read_excel("../data/dados_cerveja.xlsx")
df

# %%
## Como podemos fazer a máquina aprender?

features = ['temperatura', 'copo', 'espuma', 'cor']
target = 'classe'

x = df[features]
y = df[target]

# transformação das variáveis em string para 0 e 1 (jeito didático)
x = x.replace(
    {
        'mud': 1, 'pint': 0,
        'sim': 1, 'não': 0,
        'escura': 1, 'clara': 0,
    }
)

# random_state -> garante que todos vão ver a mesma árvore
arvore = tree.DecisionTreeClassifier(random_state=42)
arvore.fit(x,y)

plt.figure(dpi=300)
tree.plot_tree(arvore, class_names=arvore.classes_, feature_names=features, filled=True)

plt.show()

#%%

# probalidades de ser: temperatura -1, copo mud, espuma não e cor escura
probas = arvore.predict_proba([[-1, 1, 0, 1]])[0]
pd.Series(probas, index=arvore.classes_)
