#%%

import pandas as pd
from sklearn import model_selection

df = pd.read_csv('../data/dados_pontos.csv', sep=';')
df

#%%

# Seperação entre treino e teste

features = df.columns[3:-1]
target = 'flActive'

X_train, X_test, y_train, y_test = model_selection.train_test_split(df[features], 
                                                                    df[target],
                                                                    random_state=42,
                                                                    test_size=0.2,
                                                                    # garente números mais próximos
                                                                    stratify=df[target])

print('Tx Resposta Treino:', y_train.mean())
print('Tx Resposta Teste:', y_test.mean())

#%%

# Checagem nos dados

X_train.isna().sum().T

#%%

# Base de teste é APENAS PARA TESTE
# Base de treino é para INSIGHTS

input_avgRecorrencia = X_train['avgRecorrencia'].max()
X_train['avgRecorrencia'] = X_train['avgRecorrencia'].fillna(input_avgRecorrencia)
X_test['avgRecorrencia'] = X_test['avgRecorrencia'].fillna(input_avgRecorrencia)

#%%

# Árvore de Decisão

from sklearn import tree, metrics

# Aqui a gente treina
arvore = tree.DecisionTreeClassifier(max_depth=6,
                                     min_samples_leaf=50, 
                                     random_state=42)
arvore.fit(X_train, y_train)

# Aqui a gente prevê na própria base
tree_pred_train = arvore.predict(X_train)
acc_tree_train = metrics.accuracy_score(y_train, tree_pred_train)
print('Árvore Train ACC:', acc_tree_train)

# Aqui a gente prevê na base de teste
tree_pred_test = arvore.predict(X_test)
acc_tree_test = metrics.accuracy_score(y_test, tree_pred_test)
print('Árvore Test ACC:', acc_tree_test)

print('-' * 37)

# Aqui a gente prevê na própria base
tree_proba_train = arvore.predict_proba(X_train)[:,1]
tree_roc_train = metrics.roc_auc_score(y_train, tree_proba_train)
print('Árvore Train AUC:', tree_roc_train)

# Aqui a gente prevê na base de teste
tree_proba_test = arvore.predict_proba(X_test)[:,1]
tree_roc_test = metrics.roc_auc_score(y_test, tree_proba_test)
print('Árvore Test AUC:', tree_roc_test)

#%%

print('Taxa de pessoas que não voltaram:', 1 - y_test.mean())

#%%
