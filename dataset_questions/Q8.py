import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ds = pd.read_csv('academic_survival_longitudinal.csv')

## **Pergunta 8 - Qual é a divisão de gênero entre os alunos que são a primeira geração da família na faculdade e entre os que não são?**

mascara_primeira_geracao = (ds['First_Generation'] == 1) & (ds['Semester'] == 1)
mascara_nao_primeira_geracao = (ds['First_Generation'] == 0) & (ds['Semester'] == 1)

ds_primeira_geracao = ds[mascara_primeira_geracao]
ds_nao_primeira_geracao = ds[mascara_nao_primeira_geracao]

qtdFemininoPG = len(ds_primeira_geracao[ds_primeira_geracao['Gender'].isin(['Female', 'female', 'F'])])
qtdMasculinoPG = len(ds_primeira_geracao[ds_primeira_geracao['Gender'].isin(['Male', 'male', 'M'])])
qtdOutrosPG = len(ds_primeira_geracao) - qtdFemininoPG - qtdMasculinoPG

qtdFemininoNPG = len(ds_nao_primeira_geracao[ds_nao_primeira_geracao['Gender'].isin(['Female', 'female', 'F'])])
qtdMasculinoNPG = len(ds_nao_primeira_geracao[ds_nao_primeira_geracao['Gender'].isin(['Male', 'male', 'M'])])
qtdOutrosNPG = len(ds_nao_primeira_geracao) - qtdFemininoNPG - qtdMasculinoNPG

plt.subplot(1, 2, 1)
plt.pie(x=[qtdFemininoPG, qtdMasculinoPG, qtdOutrosPG],
        labels=None,
        autopct='%1.1f%%')
plt.title('Primeira Geração')

plt.subplot(1, 2, 2)
plt.pie(x=[qtdFemininoNPG, qtdMasculinoNPG, qtdOutrosNPG],
        labels=None,
        autopct='%1.1f%%')
plt.title('Não Primeira Geração')

plt.suptitle('Gênero dos Alunos por Geração na Faculdade')
plt.legend(['Feminino', 'Masculino', 'Outro ou prefere \nnão dizer'], loc="lower center", bbox_to_anchor=(-0.1, -0.3), ncol=3)

plt.show()

