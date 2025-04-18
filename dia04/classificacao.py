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

# REGRESSÃO LOGÍSTICA

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

# ÁRVORE DE DECISÃO

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

# No sklearn, o ponto de corte (threshold) padrão é na metade (0.5)

nb_proba = nb.predict_proba(df[features])[:,1]
threshold = 0.5
nb_predict = nb_proba > threshold

print(f'Ponte de corte igual a {threshold}:')
print('-' * 40)

nb_acc = metrics.accuracy_score(df[target], nb_predict)
print(f'Acurácia Naive Bayes: {nb_acc}')

nb_precision = metrics.precision_score(df[target], nb_predict)
print(f'Precisão Naive Bayes: {nb_precision}')

nb_recall = metrics.recall_score(df[target], nb_predict)
print(f'Recall Naive Bayes: {nb_recall}')

threshold = 0.8
nb_predict = nb_proba > threshold

print(f'\nPonte de corte igual a {threshold}:')
print('-' * 40)

nb_acc = metrics.accuracy_score(df[target], nb_predict)
print(f'Acurácia Naive Bayes: {nb_acc}')

nb_precision = metrics.precision_score(df[target], nb_predict)
print(f'Precisão Naive Bayes: {nb_precision}')

nb_recall = metrics.recall_score(df[target], nb_predict)
print(f'Recall Naive Bayes: {nb_recall}')

# Conclusão 1: depende do ponto de corte para eu escolher o melhor modelo.
# Conclusão 2: quem define o valor da probabilidade no ponto de corte é onde você entrega mais dinheiro para o negócio.

#%%

df['prob_nb'] = nb_proba
df

#%%

# Plotando a curva ROC

import matplotlib.pyplot as plt

fpr, tpr, thresholds = metrics.roc_curve(df[target], nb_proba)

plt.figure(dpi=150)
plt.title('Curva ROC: TPR vs FPR')
plt.plot(fpr, tpr)
plt.grid(True)
plt.plot([0,1], [0,1], '--')
plt.ylabel('Taxa de Verdadeiros Positivos (TPR / Recall / Sensibilidade)', size=9)
plt.xlabel('Taxa de Falsos Positivos (FPR / 1 - especificidade)', size=9)

# cálculo da área abaixo da curva (Area Under Curve)
modelo_precisao = round(metrics.roc_auc_score(df[target], nb_proba), 3)
plt.text(0.862, 0.065, 
         s=f'Precisão do modelo:\n{modelo_precisao}%', 
         backgroundcolor= '0.85', 
         horizontalalignment='center', 
         fontsize=8)
plt.show()

#%%

