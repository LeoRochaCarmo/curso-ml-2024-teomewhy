#%%

import pandas as pd
import numpy as np
from tqdm import tqdm

#%%

df = pd.read_csv('../pib-sampler-main/data/pop_uf_pib.csv')
df.head()

#%%

# Amostragem Estratificada

# Selecionar quantidade de pessoas proporcial ao tamanho do estado
# em relação ao país todo

# Quanto que cada estado representa do total da população no Brasil
df_peso = df.groupby('uf')['pib_pessoa'].count()
df_peso = df_peso / df.shape[0]
df_peso = df_peso.reset_index().rename(columns={'pib_pessoa': 'peso'})

medias_estratificadas = []
for i in tqdm(range(100)):
    medias_ufs = []
    for uf in df['uf'].unique():
        peso = df_peso[df_peso['uf'] == uf]['peso'].iloc[0]
        data = df[df['uf'] == uf]
        amostra = np.random.choice(data['pib_pessoa'], size=max(int(100*peso), 1))
        media_uf = np.mean(amostra)
        media_uf_penalizada = media_uf * peso
        medias_ufs.append(media_uf_penalizada)
    
    medias_estratificadas.append(np.sum(medias_ufs))

print(f'Média das médias Estrat.: {np.mean(medias_estratificadas):.2f}')
print(f'Desvio das médias Estrat.: {np.std(medias_estratificadas):.2f}')

# %%