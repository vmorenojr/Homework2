# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%% [markdown]
# ### Introduction to Data Science - 2019.2
# 
# ### Valter Moreno
#%% [markdown]
# ### Homework 2: Predicting Schools Performance
#%% [markdown]
# ### Data
#%% [markdown]
# The data for this project was provided in four csv files containing information on shools in São Paulo metropolitan region. They included characteristics of the students and classes, academic perfomance metrics, and results in the national evaluation exam (ENEM).

#%%
import numpy as np
import pandas as pd

#%%
# Reading the data into dataframes

enem = pd.read_csv('Data/ENEM2015.csv', header=0, names=['cd_escola','participantes','enem'], encoding='utf-8')
censo = pd.read_csv('Data/ESC2013_RMSP_CEM.csv', encoding='utf-8')
rendimento = pd.read_csv('Data/RendimentoEscolar2000-2015.csv', encoding='utf-8')
escolas = pd.read_csv('Data/DadosEscolares1996-2015.csv', encoding='utf-8')


#%%
enem.head()

#%%
censo.head()

#%%
rendimento.head()

#%%
escolas.head()

#%% [markdown]
# ### Check if the school numbers match in all files

#%% 
cd_enem = enem.cd_escola.unique()
cd_censo = censo.CODESC.unique()
cd_rendimento = rendimento.CODMEC.unique()
cd_escolas = escolas.CODMEC.unique()

print('cd_enem:', len(cd_enem))
print('cd_censo:', len(cd_censo))
print('cd_rendimento:', len(cd_rendimento))
print('cd_escolas:', len(cd_escolas))

type(cd_enem)

#%%
len(cd_enem) - sum(np.in1d(cd_enem, cd_censo)) 

#%%
