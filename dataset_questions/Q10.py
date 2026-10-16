import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ds = pd.read_csv('academic_survival_longitudinal.csv')

## **Pergunta 10 — Os alunos bolsistas abandonam menos o curso, mesmo tendo a renda familiar menor?**

def classificaBolsa(x):
    if x == 1:
        return 'Bolsista'
    else:
        return 'Não bolsista'

def faixaRenda(x):
    if x <= 25000:
        return 'Até 25k'
    elif x <= 40000:
        return '25-40k'
    elif x <= 55000:
        return '40-55k'
    elif x <= 75000:
        return '55-75k'
    elif x <= 100000:
        return '75-100k'
    else:
        return '100k+'

ds_bolsa = ds.dropna(subset=['Family_Income']).copy()
ds_bolsa['Bolsa'] = ds_bolsa['Scholarship'].apply(classificaBolsa)
ds_bolsa['Faixa de Renda'] = ds_bolsa['Family_Income'].apply(faixaRenda)

abandonoSemestre = ds_bolsa.groupby(['Bolsa', 'Semester'])['Target_Dropout_Next_Sem'].mean() * 100
abandonoHoras = (ds_bolsa.groupby(['Bolsa', ds_bolsa['Work_Hours'].round()])['Target_Dropout_Next_Sem'].mean() * 100).reset_index()
abandonoRenda = (ds_bolsa.groupby(['Bolsa', 'Faixa de Renda'])['Target_Dropout_Next_Sem'].mean() * 100).unstack()
abandonoRenda = abandonoRenda[['Até 25k', '25-40k', '40-55k', '55-75k', '75-100k', '100k+']]

np.random.seed(10)
amostra = ds_bolsa[ds_bolsa['Family_Income'] <= 150000].sample(3000)
amostraBolsista = amostra[amostra['Scholarship'] == 1]
amostraNaoBolsista = amostra[amostra['Scholarship'] == 0]

print(abandonoSemestre.unstack().round(1))
print(abandonoRenda.round(1))

plt.figure(figsize = (14, 10))

plt.subplot(2, 2, 1)
plt.plot(abandonoSemestre['Bolsista'], 'o-g', linewidth=2, label='Bolsista')
plt.plot(abandonoSemestre['Não bolsista'], 's--', color='gray', linewidth=2, label='Não bolsista')
plt.xlabel('Semestre')
plt.ylabel('% que abandona no semestre seguinte')
plt.title('Abandono ao longo dos semestres')
plt.legend()

plt.subplot(2, 2, 2)
sns.regplot(data=amostraNaoBolsista, x='Family_Income', y='Financial_Stress', color='gray', scatter_kws={'alpha': 0.2}, logx=True, label='Não bolsista')
sns.regplot(data=amostraBolsista, x='Family_Income', y='Financial_Stress', color='green', scatter_kws={'alpha': 0.4}, logx=True, label='Bolsista')
plt.xlabel('Renda familiar')
plt.ylabel('Estresse financeiro')
plt.title('Estresse financeiro pela renda familiar')
plt.legend()

plt.subplot(2, 2, 3)
sns.scatterplot(data=abandonoHoras, x='Work_Hours', y='Target_Dropout_Next_Sem', hue='Bolsa', palette=['green', 'gray'])
plt.xlabel('Horas de trabalho por semana')
plt.ylabel('% que abandona no semestre seguinte')
plt.title('Abandono pelas horas de trabalho')

plt.subplot(2, 2, 4)
sns.heatmap(abandonoRenda, annot=True, fmt='.1f', cmap='Reds')
plt.xlabel('Renda familiar')
plt.xticks(rotation=0)
plt.ylabel('')
plt.title('% de abandono por faixa de renda')

plt.suptitle('Os alunos bolsistas abandonam menos o curso, mesmo tendo renda familiar menor?')
plt.tight_layout()
plt.show()