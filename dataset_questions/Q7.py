import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ds = pd.read_csv('academic_survival_longitudinal.csv')

## **Quantos alunos que reprovaram trabalham e estão entre os 25% de menor renda familiar?**
baixaRenda = ds[ds['Semester'] == 1]['Family_Income'].quantile(0.25)

mascara_reprovou = ds['Failed_Courses'] > 0
mascara_trabalha = ds['Work_Hours'] > 0
mascara_baixa_renda = ds['Family_Income'] < baixaRenda

qtdReprovaram = len(np.unique(ds[mascara_reprovou]['Student_ID']))
qtdReprovaramTrabalham = len(np.unique(ds[mascara_reprovou & mascara_trabalha]['Student_ID']))
qtdReprovaramBaixaRenda = len(np.unique(ds[mascara_reprovou & mascara_baixa_renda]['Student_ID']))
qtdReprovaramBaixaRendaTrabalham = len(np.unique(ds[mascara_reprovou & mascara_baixa_renda & mascara_trabalha]['Student_ID']))

print('Limite de baixa renda:', baixaRenda)
print('Reprovaram:', qtdReprovaram)
print('Reprovaram e trabalham:', qtdReprovaramTrabalham)
print('Reprovaram e sao de baixa renda:', qtdReprovaramBaixaRenda)
print('Reprovaram, sao de baixa renda e trabalham:', qtdReprovaramBaixaRendaTrabalham)

grupos = ['Reprovaram', 'Reprovaram\ne trabalham', 'Reprovaram e são\nde baixa renda', 'Reprovaram, são de\nbaixa renda e trabalham']
quantidades = [qtdReprovaram, qtdReprovaramTrabalham, qtdReprovaramBaixaRenda, qtdReprovaramBaixaRendaTrabalham]

plt.figure(figsize=(10, 6))
plt.bar(grupos, quantidades, color=['steelblue', 'orange', 'seagreen', 'tomato'])
plt.ylabel('Quantidade de alunos')
plt.title('Alunos que reprovaram por situação de trabalho e renda')
plt.show()