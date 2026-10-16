import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ds = pd.read_csv('academic_survival_longitudinal.csv')

## *Os alunos que planejam trancar o curso no semestre que vem têm mais estresse financeiro do que os que não planejam?*

def classificaTrancamento(x):
    if x == 1:
        return 'Planejam trancar'
    else:
        return 'Nao planejam trancar'

ds['Trancamento'] = ds['Target_Dropout_Next_Sem'].apply(classificaTrancamento)

print(ds.groupby('Trancamento')['Financial_Stress'].describe())

sns.boxplot(
    data = ds,
    x = 'Trancamento',
    y = 'Financial_Stress'
)
plt.xlabel('')
plt.ylabel('Estresse financeiro')
plt.title('Estresse financeiro de quem planeja ou nao trancar o curso')
plt.show()