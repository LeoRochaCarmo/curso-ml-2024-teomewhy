#%%

import pandas as pd

df = pd.read_excel('..\\data\\dados_cerveja_nota.xlsx')
df

#%%

df['Aprovado'] = df['nota'] >= 5
df

features = ['cerveja']
target = 'Aprovado'

#%%

# REGRESSÃO

from sklearn import linear_model

reg = linear_model.LogisticRegression(penalty=None, 
                                      fit_intercept=True)

# aqui o modelo aprende
reg.fit(df[features], df[target])

# aqui o modelo prevê
reg_predict = reg.predict(df[features])

#%%

from sklearn import metrics

# comparando valor verdadeiro e previsão do modelo
reg_acc = metrics.accuracy_score(df[target], reg_predict)
print(f'Acurácia Reg Log: {reg_acc}')

reg_precision = metrics.precision_score(df[target], reg_predict)
print(f'Precisão Reg Log: {reg_precision}')

reg_recall = metrics.recall_score(df[target], reg_predict)
print(f'Recall Reg Log: {reg_recall}')

# matriz de confusão
reg_conf = metrics.confusion_matrix(df[target], reg_predict)
reg_conf = pd.DataFrame(reg_conf,
                        index=['False', 'True'],
                        columns=['False', 'True'])
reg_conf

#%%

# REGRESSÃO

from sklearn import tree

arvore = tree.DecisionTreeClassifier(max_depth=2)

# aqui o modelo aprende
arvore.fit(df[features], df[target])

# aqui o modelo prevê
arvore_predict = arvore.predict(df[features])

arvore_acc = metrics.accuracy_score(df[target], arvore_predict)
print(f'Acurácia Árvore: {arvore_acc}')

arvore_precision = metrics.precision_score(df[target], arvore_predict)
print(f'Precisão Árvore: {arvore_precision}')

arvore_recall = metrics.recall_score(df[target], arvore_predict)
print(f'Recall Árvore: {arvore_recall}')

arvore_conf = metrics.confusion_matrix(df[target], arvore_predict)
arvore_conf = pd.DataFrame(arvore_conf,
                        index=['False', 'True'],
                        columns=['False', 'True'])
arvore_conf

#%%

# NAIVE BAYES

from sklearn import naive_bayes

nb = naive_bayes.GaussianNB()

# aqui o modelo aprende
nb.fit(df[features], df[target])

# aqui o modelo prevê
nb_predict = nb.predict(df[features])

nb_acc = metrics.accuracy_score(df[target], nb_predict)
print(f'Acurácia Naive Bayes: {nb_acc}')

nb_precision = metrics.precision_score(df[target], nb_predict)
print(f'Precisão Naive Bayes: {nb_precision}')

nb_recall = metrics.recall_score(df[target], nb_predict)
print(f'Recall Naive Bayes: {nb_recall}')


nb_conf = metrics.confusion_matrix(df[target], nb_predict)
nb_conf = pd.DataFrame(nb_conf,
                        index=['False', 'True'],
                        columns=['False', 'True'])
nb_conf

#%% 
