#%%

import pandas as pd
import numpy as np

#%%

df = pd.read_csv('../pib-sampler-main/data/pop_uf_pib.csv')
df.head()

#%%

media_pib_nacional = df['pib_pessoa'].mean()
print(f'PIB per capit Pop: R${media_pib_nacional:,.2f}')

#%%

# Amonstragem Aleatória Simples

# Cada uma das 2 milhões vão ter a mesma probabilidade de
# serem escolhidas por acaso

# Simulação gerando médias de 100 amostras aleatórias
medias_simples = []
for i in range(100):
    amostra_simples = np.random.choice(df['pib_pessoa'], size=100)
    media_simples = amostra_simples.mean()
    medias_simples.append(media_simples)

print(f'Média das médias: {np.mean(medias_simples):.2f}')
print(f'Desvio das médias: {np.std(medias_simples):.2f}')

# %%

