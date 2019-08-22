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
import seaborn as sb

pd.set_option('display.float_format', lambda x: '%.2f' % x)

#%%
# Reading the data into dataframes

enem = pd.read_csv('Data/ENEM2015.csv', header=0, names=['CD_ESCOLA','PARTICIPANTES','ENEM'], encoding='utf-8')
rendimento = pd.read_csv('Data/RendimentoEscolar2000-2015.csv', encoding='utf-8')
escolas = pd.read_csv('Data/DadosEscolares1996-2015.csv', encoding='utf-8')

censo = pd.read_csv('Data/ESC2013_RMSP_CEM.csv', encoding='utf-8')
censo.replace(['999,9', '999,99', '9999,99'], '', inplace=True)

#%%
enem.head()

#%%
enem.dtypes

#%%
censo.head()

#%%
print('Columns of the string type:')
print(censo.select_dtypes('object').columns.to_list())
print()
print('Columns of numeric type:')
print(censo.select_dtypes('number').columns.to_list())

#%%
rendimento.head()

#%%
print('Columns of the string type:')
print(rendimento.select_dtypes('object').columns.to_list())
print()
print('Columns of numeric type:')
print(rendimento.select_dtypes('number').columns.to_list())

#%% [markdown]
# The last column of the dataframe is ill-formatted. 
# Its contents will be fixed and stored as integers.

#%%
cols = list(rendimento.columns)
del cols[-1]
cols.append('EJATOT')
rendimento.columns = cols
rendimento.EJATOT = pd.to_numeric(rendimento.EJATOT.str.replace(',', ''))

rendimento.head()

#%%
escolas.head()

#%%
print('Columns of the string type:')
print(escolas.select_dtypes('object').columns.to_list())
print()
print('Columns of numeric type:')
print(escolas.select_dtypes('number').columns.to_list())

#%% [markdown]
# ### Checking if the number of schools match in all files

#%% 
cd_enem = enem.CD_ESCOLA.unique()
cd_censo = censo.CODESC.unique()
cd_rendimento = rendimento.CODMEC.unique()
cd_escolas = escolas.CODMEC.unique()

print('Number of records in each dataframe:')
print('cd_enem:', len(cd_enem))
print('cd_censo:', len(cd_censo))
print('cd_rendimento:', len(cd_rendimento))
print('cd_escolas:', len(cd_escolas), '\n')

print('Number of schools in the ENEM dataframe:')
print('cd_enem:', len(cd_enem))
print('cd_censo:', np.in1d(cd_censo, cd_enem).sum())
print('cd_rendimento:', np.in1d(cd_rendimento, cd_enem).sum())
print('cd_escolas:', np.in1d(cd_escolas, cd_enem).sum())

#%% [markdown]
# Many of the schools that are in the ENEM2015.csv file are not in 
# in the other datasets. Schools that are not in the ENEM file
# will be dropped then from the other dataframes. Only schools that 
# are listed in the four dataframes will be used in the analysis.

#%%
cd_enem = set(cd_enem)
cd_censo = set(cd_censo)
cd_rendimento = set(cd_rendimento)
cd_escolas = set(cd_escolas)

cods = cd_enem.intersection(cd_censo, cd_rendimento, cd_escolas)

enem = enem[enem.CD_ESCOLA.isin(cods)]
censo = censo[censo.CODESC.isin(cods)]
rendimento = rendimento[rendimento.CODMEC.isin(cods)]
escolas = escolas[escolas.CODMEC.isin(cods)]

#%% [markdown]
# Dataframes 'rendimento' and 'escolas' contain longitudinal data. 
# I will get the number of schools for which there is data for each
# year in the time horizon.

#%%
def long_count(df, df_name):
      print('Dataframe:', df_name)
      for year in np.sort(df.ANO.unique()):
            print(year, ': ',
                  df[(df.ANO == year) & (df.CODMEC.isin(cods))].shape[0])
      print('\n')

long_count(rendimento, 'Rendimento')
long_count(escolas, 'Escolas')

#%% [markdown]
# I decided to retain years for which there is data for at least 500 schools. This is
# necessary to allow meus to train and test the model with samples of reasonable size. 
# Thus, I will be able to keep only the records for 2015 in 'rendimentos', and for 
# 2012, 2013, 2014 and 2015 in 'escolas'.

#%%

rendimento = rendimento[(rendimento.ANO == 2015) & 
                        (rendimento.CODMEC.isin(cods))]

rendimento[(rendimento.ANO == 2015) & (rendimento.CODMEC.isin(cods))].shape

escolas = escolas[(escolas.ANO.isin([2012, 2013, 2014, 2015])) &
                  (escolas.CODMEC.isin(cods))]

print('Number of remaining records in each dataframe:')
print('Enem:', enem.shape[0])
print('Censo:', censo.shape[0])
print('Rendimento:', rendimento.shape[0])
print('Escolas:', escolas.shape[0], '\n')

censo.to_csv('Data/censo.csv', index=False)
rendimento.to_csv('Data/rendimento.csv', index=False)
escolas.to_csv('Data/escolas.csv', index=False)

#%% [markdown]
# ## Missing values
# I will performa a quick inspection of the dataframes for missing values.

#%%

def miss_data(df):
    cols = []
    values = []
    for col in df.columns:
        miss = df[col].isnull().sum()
        if miss > 0:
            cols.append(col)
            values.append(miss)
    missing = {'Columns': cols, 'Missing': values}
    return missing 

#%%
censo_miss = pd.DataFrame(miss_data(censo))
censo_miss['Percent missing'] = censo_miss.Missing/censo.shape[0]
print('Columns with missing values in dataframe Censo:', '\n',
      censo_miss.sort_values(by='Missing', ascending=False))

#%%    
rendimento_miss = pd.DataFrame(miss_data(rendimento))
rendimento_miss['Percent missing'] = rendimento_miss.Missing/rendimento.shape[0]
print('Columns with missing values in dataframe Rendimento:', '\n',
      rendimento_miss.sort_values(by='Missing', ascending=False))

#%%
escolas_miss = pd.DataFrame(miss_data(escolas))
escolas_miss['Percent missing'] = escolas_miss.Missing/escolas.shape[0]
print('Columns with missing values in dataframe Escolas:', '\n',
      escolas_miss.sort_values(by='Missing',ascending=False))

#%%
censo_miss.to_csv('Data/censo_miss.csv', index=False)
rendimento_miss.to_csv('Data/rendimento_miss.csv', index=False)
escolas_miss.to_csv('Data/escolas_miss.csv', index=False)

#%% [markdown]
# Alghough the 'rendimento' dataframe has no missing values, 'censo' and 
# 'escolas' do.

# Instead of inputing the missing values, I will drop the records with missing 
# values for any of the features. Nevertheless, given the small size of the sample
# (505 schools), I want to retain as much data as possible. 

# I start by discarding columns with more than 10% of missing data. Before,
# I do this, I will merge the 'enem', 'censo', and 'rendimento' dataframes, as
# they have data only for one year. Because the 'escolas' dataframe has longitudinal data,
# it will be treated separetely.

#%%
schools = enem.merge(censo, left_on = 'CD_ESCOLA', right_on='CODESC')
schools = schools.merge(rendimento, left_on = 'CD_ESCOLA', right_on='CODMEC')
schools.drop(['CODESC', 'CODMEC', 'ANO', 'DEP', 'NOME', 'SETEDU',
              'DISTRITO', 'SUBPREF', 'DIRET', 'ZONA',],
             axis=1, inplace=True)

#%%
def drop_miss(df, df_miss, percent):
      drop_cols = df_miss[df_miss['Percent missing'] >= percent].Columns
      df.drop(drop_cols, axis=1, inplace=True)
      return df

#%%
schools_miss = pd.DataFrame(miss_data(schools))
schools_miss['Percent missing'] = schools_miss.Missing/schools.shape[0]

schools = drop_miss(schools, schools_miss, .10)

#%%
escolas = escolas[escolas.CODMEC.isin(schools.CD_ESCOLA)]

escolas_miss = pd.DataFrame(miss_data(escolas))
escolas_miss['Percent missing'] = escolas_miss.Missing/escolas.shape[0]
escolas = drop_miss(escolas, escolas_miss, .10)

#%% [markdown]
# Here are the remaining columns with missing values:

#%%
schools_miss = pd.DataFrame(miss_data(schools))
schools_miss['Percent missing'] = schools_miss.Missing/schools.shape[0]
print('Columns with missing values in dataframe Schools:', '\n',
      schools_miss.sort_values(by='Missing', ascending=False))
print()
print('Sorted by name:', '\n',
      schools_miss.sort_values(by='Columns'))

#%%    
escolas_miss = pd.DataFrame(miss_data(escolas))
escolas_miss['Percent missing'] = escolas_miss.Missing/escolas.shape[0]
print('Columns with missing values in dataframe Escolas:', '\n',
      escolas_miss.sort_values(by='Missing', ascending=False))
print()
print('Sorted by name:', '\n',
      escolas_miss.sort_values(by='Columns'))

#%% [markdown]
# Based on the descriptions of the variables in the data dictionary,
# features that seem less relevant to predict the school's result
# in ENEM and that have missing values will be removed from the 
# dataframes.

#%% [markdown]
# The following columns will be dropped from the 'schools' dataframe:
#     - AP9EF_09
#     - AB1EM_10
#     - APR3EM_11
#     - AP9EF_11
#     - AP9EF_10
#     - AP3EM_12

#%%
schools.drop(['AP9EF_09', 'AB1EM_10', 'APR3EM_11',
              'AP9EF_11', 'AP9EF_10', 'AP3EM_12'],
             axis =1, inplace=True)

#%% [markdown]
# The following columns will be dropped from the 'escolas' dataframe:
#     - 0A3
#     - 0A4
#     - 4A6
#     - 5A6
#     - >6 
#     - TotalEdInf
#     - MENOR3
#     - CLE9F1S
#     - ALE9F1S
#     - CLE9F2S
#     - ALE9F2S
#     - CLE9F5S
#     - ALE9F5S
#     - CLE9F3S
#     - ALE9F3S
#     - CLE9F4S
#     - ALE9F4S
#     - CLE9F9S
#     - ALE9F9S
#     - ALE9F8S
#     - CLE9F8S
#     - CLE9F6S
#     - ALE9F6S
#     - CLE9F7S
#     - ALE9F7S

#%%
escolas.drop(escolas_miss[escolas_miss['Percent missing'] > 0.01]['Columns'], 
             axis=1, inplace=True)

#%% [markdown]
# The 'escolas' dataframe is in the long format. It will be converted to the 
# wide format before the rows with missing values are removed.

#%%
escolas.drop(['TIPOESC', 'NOME', 'DEP', 'SETEDU', 'DISTRITO',
              'SUBPREF', 'CORED', 'ZONA'], axis=1, inplace=True)

escolas_long = pd.melt(escolas, id_vars=['CODMEC','ANO'], 
                       var_name='Vars', value_name='Values')

escolas_long['NewVar'] = escolas_long['Vars'] + '_' + escolas_long['ANO'].map(str)
      
escolas_long.drop(['ANO', 'Vars'], axis=1, inplace=True)

escolas = escolas_long.pivot(index='CODMEC', columns='NewVar',
                             values='Values').reset_index()

#%% [markdown]
# I will remove rows with missing values from both dataframes, and
# merge them into a new 'schools' dataframe. The new dataframe will be recorded 
# to the 'Schools.csv' file.

#%%
schools.dropna(inplace=True)
escolas.dropna(inplace=True)

schools = schools.merge(escolas, left_on='CD_ESCOLA', right_on='CODMEC')
schools.head()

#%% [markdown]
# The final dataset contains data for 436 schools.

# I will now drop columns that are not relevant to the analysis
# and set the type of categorical features to string.

#%%
schools.drop(['ID', 'LONGITUDE', 'LATITUDE', 'CODESCTX', 'NOMEESC',
              'NOMEMUN', 'NOMDIST', 'COD_DEP',
              'TIP_DEP', 'BAIRRO', 'CEP', 'END_ESC', 'NUM_ESC',
              'DDD', 'TELEFONE', 'LOCALIZA'], 
             axis=1, inplace=True)

#%%
schools.CD_ESCOLA = [str(x) for x in schools.CD_ESCOLA]
schools.CODMUN = [str(x) for x in schools.CODMUN]
schools.CODDIST = [str(int(x)) for x in schools.CODDIST]

#%% [markdown]
# The next step is to check the feature for redundant information. To do this
# I will generate the correlations between variables and print them in a 
# heatmap.

#%%
sb.heatmap(schools.corr(), cmap='viridis', center=0)

#%% [markdown]
# The chart shows several variables with very high absolute correlations, 
# suggesting the information they contain is redundant. I will list the
# variables with correlations equal to 1.0 or -1.0.

#%%

corr_1 = schools.select_dtypes('number').corr()

col_list = []

for col in corr_1.columns:
      if (sum(corr_1[col] == 1.0) >= 2) | (sum(corr_1[col] == -1.0) >= 1):
            col_list.append(col)

corr_1 = schools[col_list].corr()
corr_1

#%%
while True:
      cols_drop = list(corr_1.index[(corr_1.iloc[:, 0] == 1) | 
                                    (corr_1.iloc[:, 0] == -1)])
      schools.drop(cols_drop[1:len(cols_drop)], 
                   axis=1, inplace=True)
      corr_1.drop(cols_drop, 
                  axis=1, inplace=True)
      corr_1.drop(cols_drop, 
                  axis=0, inplace=True)
      if corr_1.shape[0] == 0:
            break

#%% [markdown] 
# In addition, many correlations were not generated. This happens when
# the variances of at least one of the variables is zero. Features with
# variances equal to zero do not contain information relevant 
# to the analysis and will be dropped.

#%%
std_0 = schools.select_dtypes('number').std()
cols_std_0 = list(std_0[std_0 == 0].index)

print('Columns with variance equal to 0.0:')
print(cols_std_0)

schools.drop(cols_std_0, axis=1, inplace=True)

#%% [markdown]
# Finally, the dataframe will be recorded to the 'Schools.csv' file.

#%%
print('The final dataframe has', schools.shape[1],
      'columns and', schools.shape[0], 'rows.')

#%%
schools.describe()

#%%
schools.to_csv('Data/Schools.csv', index=False)

#%%
