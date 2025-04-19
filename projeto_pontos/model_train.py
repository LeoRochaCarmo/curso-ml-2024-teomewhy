#%%

import pandas as pd

from sklearn import metrics
from sklearn import model_selection
from sklearn import pipeline

from sklearn import tree
from sklearn import linear_model
from sklearn import ensemble
from sklearn import naive_bayes

from feature_engine import imputation 

#%%

df = pd.read_csv('../data/dados_pontos.csv', sep=';')
df

#%%

# Definição das features e target
features = df.columns.tolist()[3:-1]
target = 'flActive'

#%%

# Divisão dos dados em treino e teste
# stratify -> Isso é MUITO importante quando o target tem classes desbalanceadas
X_train, X_test, y_train, y_test = model_selection.train_test_split(df[features],
                                                                    df[target],
                                                                    test_size=0.2,
                                                                    random_state=42,
                                                                    stratify=df[target])

print('Tx. Resposta Treino:', y_train.mean())
print('Tx. Resposta Teste:', y_test.mean())

#%%

# Etapas de imputação de valores faltantes
features_imput_0 = [
    'qtdeRecencia',
    'freqDias',
    'freqTransacoes',
    'qtdListaPresença',
    'qtdChatMessage',
    'qtdTrocaPontos',
    'qtdResgatarPonei',
    'qtdPresençaStreak',
    'pctListaPresença',
    'pctChatMessage',
    'pctTrocaPontos',
    'pctResgatarPonei',
    'pctPresençaStreak',
    'qtdePontosGanhos',
    'qtdePontosGastos',
    'qtdePontosSaldo'
]

imputacao_0 = imputation.ArbitraryNumberImputer(variables=features_imput_0, arbitrary_number=0)

max_avgRecorrencia = X_train['avgRecorrencia'].max()
imputacao_max = imputation.ArbitraryNumberImputer(variables='avgRecorrencia',
                                                  arbitrary_number=max_avgRecorrencia)

#%%

# ÁRVORE DE DECISÃO 

model = tree.DecisionTreeClassifier(max_depth=4, min_samples_leaf=50, random_state=42)

meu_pipeline = pipeline.Pipeline([
    ('input_0', imputacao_0),
    ('input_max', imputacao_max),
    ('model', model),
    ])

meu_pipeline.fit(X_train, y_train)

y_train_predict = meu_pipeline.predict(X_train)
y_train_proba = meu_pipeline.predict_proba(X_train)[:,1]
y_test_predict = meu_pipeline.predict(X_test)
y_test_proba = meu_pipeline.predict_proba(X_test)[:,1]

acc_train = metrics.accuracy_score(y_train, y_train_predict)
acc_test = metrics.accuracy_score(y_test, y_test_predict)
print('Acurácia base train', acc_train)
print('Acurácia base test', acc_test)

auc_train = metrics.roc_auc_score(y_train, y_train_proba)
auc_test = metrics.roc_auc_score(y_test, y_test_proba)
print('\nAUC base train', auc_train)
print('AUC base test', auc_test)

# Acurácia base train 0.8109619686800895
# Acurácia base test 0.8008948545861297

# AUC base train 0.8531284015204619
# AUC base test 0.8380512447094162

#%%

# RANDOM FOREST

model = ensemble.RandomForestClassifier(random_state=42)

params = {
    'n_estimators': [100, 150, 250, 500],
    'min_samples_leaf': [10, 20, 30, 50, 100],
}

# Grid Search -> encontra automaticamente a melhor combinação de hiperparâmetros para o modelo.
# Faz validação cruzada
grid = model_selection.GridSearchCV(model, param_grid=params, 
                                    n_jobs=-1, 
                                    scoring='roc_auc')

meu_pipeline = pipeline.Pipeline([
    ('input_0', imputacao_0),
    ('input_max', imputacao_max),
    ('model', grid),
    ])

meu_pipeline.fit(X_train, y_train)

y_train_predict = meu_pipeline.predict(X_train)
y_train_proba = meu_pipeline.predict_proba(X_train)[:,1]
y_test_predict = meu_pipeline.predict(X_test)
y_test_proba = meu_pipeline.predict_proba(X_test)[:,1]

acc_train = metrics.accuracy_score(y_train, y_train_predict)
acc_test = metrics.accuracy_score(y_test, y_test_predict)
print('Acurácia base train', acc_train)
print('Acurácia base test', acc_test)

auc_train = metrics.roc_auc_score(y_train, y_train_proba)
auc_test = metrics.roc_auc_score(y_test, y_test_proba)
print('\nAUC base train', auc_train)
print('AUC base test', auc_test)

# %%
