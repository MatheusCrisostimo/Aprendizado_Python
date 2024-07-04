import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import rcParams

# Configurar parâmetros para gráficos
rcParams.update({'figure.autolayout': True})

# Dados fornecidos
data = {
    'ultima_dia_semana': ['25/02/2024', '03/03/2024', '10/03/2024', '17/03/2024', '24/03/2024', '31/03/2024', '07/04/2024', '14/04/2024'],
    'vendas_semana': [10, 10, 9, 4, 6, 13, 15, 9],
    'fichas_semana': [40, 108, 93, 69, 56, 54, 51, 43],
    'internas': [22, 58, 62, 54, 42, 33, 31, 22],
    'externas': [18, 50, 31, 15, 14, 21, 20, 21],
    'taxa_conv_internas': [45, 17, 15, 7, 14, 39, 48, 41],
    'taxa_conv_externas': [56, 20, 29, 27, 43, 62, 75, 43],
    'taxa_conversao_semana': [25, 9, 10, 6, 11, 24, 29, 21]
}

# Criar DataFrame
df = pd.DataFrame(data)

# Converter datas
df['ultima_dia_semana'] = pd.to_datetime(df['ultima_dia_semana'], format='%d/%m/%Y')

# Análise Descritiva
descriptive_stats = df.describe()

# ANOVA para taxa de conversão semanal
model = ols('taxa_conversao_semana ~ vendas_semana + fichas_semana + internas + externas', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)

# Teste de Tukey (se necessário)
tukey = pairwise_tukeyhsd(endog=df['taxa_conversao_semana'], groups=df['vendas_semana'], alpha=0.05)

# Função para simular a taxa de conversão
def simulate_conversion(base_rate, n_simulations=1000):
    np.random.seed(42)
    simulated_conversions = np.random.normal(loc=base_rate, scale=5, size=n_simulations)
    return simulated_conversions

# Simular para diferentes períodos (valores médios fictícios)
base_rate_30 = 20  # valor fictício para simulação
base_rate_60 = 25  # valor fictício para simulação
base_rate_90 = 15  # valor fictício para simulação

sim_30 = simulate_conversion(base_rate_30)
sim_60 = simulate_conversion(base_rate_60)
sim_90 = simulate_conversion(base_rate_90)

# Calcular médias e intervalos de confiança
mean_30 = np.mean(sim_30)
mean_60 = np.mean(sim_60)
mean_90 = np.mean(sim_90)

ci_30 = np.percentile(sim_30, [2.5, 97.5])
ci_60 = np.percentile(sim_60, [2.5, 97.5])
ci_90 = np.percentile(sim_90, [2.5, 97.5])

# Função de custo (minimizar o negativo da taxa de conversão média)
def cost_function(period):
    period = np.round(period)  # Arredondar para o inteiro mais próximo
    if period == 30:
        simulated_data = simulate_conversion(base_rate_30)
    elif period == 60:
        simulated_data = simulate_conversion(base_rate_60)
    elif period == 90:
        simulated_data = simulate_conversion(base_rate_90)
    else:
        raise ValueError("Period must be 30, 60, or 90")
    return -np.mean(simulated_data)

# Executar a otimização
result = minimize(cost_function, x0=60, bounds=[(30, 90)], method='L-BFGS-B')
optimal_period = np.round(result.x[0])

# Configurar o PDF para salvar os gráficos
pdf_path = 'analise_pre_lancamento.pdf'
pp = PdfPages(pdf_path)

# Adicionar explicações e seções textuais ao PDF
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')
text = """
Explicação sobre as Metodologias Utilizadas:
Neste relatório, utilizamos Análise de Variância (ANOVA) para avaliar as diferenças entre as taxas de conversão semanais
baseadas em diferentes fatores, como vendas semanais e fichas cadastradas. Além disso, empregamos simulações para prever
taxas de conversão em diferentes períodos de pré-lançamento (30, 60 e 90 dias) e uma otimização para determinar o período
ideal de pré-lançamento.

Resumo:
Analisamos dados históricos de vendas e fichas cadastradas para entender o impacto no sucesso do lançamento imobiliário.
Aplicamos ANOVA para identificar variáveis significativas e utilizamos o teste de Tukey para comparações múltiplas. 
Simulações ajudaram a prever taxas de conversão para diferentes períodos de pré-lançamento.

Métricas:
- Média e Intervalo de Confiança (IC 95%) das taxas de conversão simuladas.
- ANOVA para identificar variáveis significativas.
- Teste de Tukey para comparações múltiplas.

Conclusão:
O período de pré-lançamento ideal, determinado pela otimização, é de {optimal_period} dias. Este período maximiza a taxa
de conversão média, sugerindo que uma preparação adequada durante este tempo é crucial para o sucesso do lançamento.
"""
ax.text(0, 1, text, fontsize=12, verticalalignment='top')
pp.savefig(fig)

# Gráfico de Barras: Fichas Internas e Externas por Semana
plt.figure(figsize=(10, 6))
df.plot(x='ultima_dia_semana', y=['internas', 'externas'], kind='bar', stacked=True)
plt.xlabel('Último Dia da Semana')
plt.ylabel('Número de Fichas')
plt.title('Fichas Internas e Externas por Semana')
plt.legend(title='Tipo de Ficha')
plt.grid(True)
plt.tight_layout()
pp.savefig()  # Salvar gráfico no PDF
plt.show()

# Gráfico de Dispersão: Vendas Semanais vs Taxa de Conversão
plt.figure(figsize=(10, 6))
plt.scatter(df['vendas_semana'], df['taxa_conversao_semana'])
plt.xlabel('Vendas Semanais')
plt.ylabel('Taxa de Conversão (%)')
plt.title('Relação entre Vendas Semanais e Taxa de Conversão')
plt.grid(True)
plt.tight_layout()
pp.savefig()  # Salvar gráfico no PDF
plt.show()

# Boxplot: Distribuição das Taxas de Conversão
plt.figure(figsize=(10, 6))
sns.boxplot(data=df[['taxa_conv_internas', 'taxa_conv_externas']])
plt.xlabel('Tipo de Taxa de Conversão')
plt.ylabel('Taxa de Conversão (%)')
plt.title('Distribuição das Taxas de Conversão Internas e Externas')
plt.grid(True)
plt.tight_layout()
pp.savefig()  # Salvar gráfico no PDF
plt.show()

# Histograma: Distribuição das Taxas de Conversão Simuladas
plt.figure(figsize=(10, 6))
sns.histplot(sim_30, kde=True, label='30 dias')
sns.histplot(sim_60, kde=True, label='60 dias')
sns.histplot(sim_90, kde=True, label='90 dias')
plt.xlabel('Taxa de Conversão Simulada (%)')
plt.ylabel('Frequência')
plt.title('Distribuição das Taxas de Conversão Simuladas')
plt.legend()
plt.grid(True)
plt.tight_layout()
pp.savefig()  # Salvar gráfico no PDF
plt.show()

# Gráfico de Linhas: Taxas de Conversão Internas e Externas ao Longo do Tempo
plt.figure(figsize=(10, 6))
plt.plot(df['ultima_dia_semana'], df['taxa_conv_internas'], label='Taxa de Conversão Internas', marker='o')
plt.plot(df['ultima_dia_semana'], df['taxa_conv_externas'], label='Taxa de Conversão Externas', marker='o')
plt.xlabel('Último Dia da Semana')
plt.ylabel('Taxa de Conversão (%)')
plt.title('Taxas de Conversão Internas e Externas ao Longo do Tempo')
plt.legend()
plt.grid(True)
plt.tight_layout()
pp.savefig()  # Salvar gráfico no PDF
plt.show()

# Heatmap: Correlação entre Variáveis
plt.figure(figsize=(10, 6))
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Mapa de Calor da Correlação entre Variáveis')
plt.tight_layout()
pp.savefig()  # Salvar gráfico no PDF
plt.show()

# Fechar o PDF
pp.close()

# Salvar os dados e resultados em um arquivo Excel
output_file = 'analise_pre_lancamento.xlsx'
with pd.ExcelWriter(output_file) as writer:
    df.to_excel(writer, sheet_name='Dados Originais')
    descriptive_stats.to_excel(writer, sheet_name='Estatísticas Descritivas')
    anova_table.to_excel(writer, sheet_name='ANOVA')
    pd.DataFrame(data=tukey.summary()).to_excel(writer, sheet_name='Tukey')

print(f'Relatório salvo em: {pdf_path}')
print(f'Dados salvos em: {output_file}')