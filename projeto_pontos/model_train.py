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

features = df.columns.tolist()[3:-1]
target = 'flActive'

#%%

X_train, X_test, y_train, y_test = model_selection.train_test_split(df[features],
                                                                    df[target],
                                                                    test_size=0.2,
                                                                    random_state=42,
                                                                    stratify=df[target])

print('Tx. Resposta Treino:', y_train.mean())
print('Tx. Resposta Teste:', y_test.mean())

#%%

max_avgRecorrencia = X_train['avgRecorrencia'].max()
imputacao_max = imputation.ArbitraryNumberImputer(variables='avgRecorrencia',
                                                  arbitrary_number=max_avgRecorrencia)

model = tree.DecisionTreeClassifier(max_depth=4, min_samples_leaf=50)

meu_pipeline = pipeline.Pipeline([
    ('input_max', imputacao_max),
    ('model', model),
    ])

meu_pipeline.fit(X_train, y_train)

#%%

y_test_predict = meu_pipeline.predict(X_test)
y_test_predict