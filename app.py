# save this as app.py
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import t
import streamlit as st

# Carregamento dos Dados
df = pd.read_csv("immunization_master_data.csv")

# Pré-processamento dos Dados
siglas_estados = ['AC','AL','AM','AP','BA','CE','DF','ES','GO','MA','MG','MS','MT',
                  'PA','PB','PE','PI','PR','RJ','RN','RO','RR','RS','SC','SE','SP','TO']
estado_para_regiao = {
    'AC': 'Norte', 'AP': 'Norte', 'AM': 'Norte', 'PA': 'Norte', 'RO': 'Norte', 'RR': 'Norte', 'TO': 'Norte',
    'AL': 'Nordeste', 'BA': 'Nordeste', 'CE': 'Nordeste', 'MA': 'Nordeste', 'PB': 'Nordeste',
    'PE': 'Nordeste', 'PI': 'Nordeste', 'RN': 'Nordeste', 'SE': 'Nordeste',
    'DF': 'Centro-Oeste', 'GO': 'Centro-Oeste', 'MT': 'Centro-Oeste', 'MS': 'Centro-Oeste',
    'ES': 'Sudeste', 'MG': 'Sudeste', 'RJ': 'Sudeste', 'SP': 'Sudeste',
    'PR': 'Sul', 'RS': 'Sul', 'SC': 'Sul'
}
df = df[df['LOCAL_NAME'].isin(siglas_estados)].copy()
df['Região'] = df['LOCAL_NAME'].map(estado_para_regiao)

vacinas_basicas = [
    'FL_U1_BCG', 'FL_U1_POLIO', 'FL_U1_DTP', 'FL_U1_HepB',
    'FL_U1_Hib', 'FL_Y1_DTP', 'FL_Y1_POLIO', 'FL_Y1_MMR1'
]

df = df[
    (df['INDICATOR'].isin(vacinas_basicas)) &
    (df['PC_COVERAGE'].notna())
]

df_agrupado = df.groupby(['LOCAL_NAME', 'Região', 'YEAR']).agg({
    'PC_COVERAGE': 'mean',
    'MHDI_I': 'mean',
    'MHDI_E': 'mean'
}).reset_index()

st.sidebar.title("Controle")
ano_selecionado = st.sidebar.slider("Ano", int(df_agrupado['YEAR'].min()), int(df_agrupado['YEAR'].max()), 2021)
mostrar_regressao = st.sidebar.checkbox("Mostrar linha de regressão", value=True)
mostrar_rotulos = st.sidebar.checkbox("Mostrar nomes dos estados", value=False)

variavel_idh_legenda = {'MHDI_I': 'MHDI_I (renda)', 'MHDI_E': 'MHDI_E (educação)'}
variavel_idh = st.sidebar.selectbox("Escolher variável de IDH", options=list(variavel_idh_legenda.keys()), index=0)



df_ano = df_agrupado[df_agrupado['YEAR'] == ano_selecionado]

melhores_estados = df_ano.nlargest(3, 'PC_COVERAGE')[['LOCAL_NAME', 'PC_COVERAGE']]
piores_estados = df_ano.nsmallest(3, 'PC_COVERAGE')[['LOCAL_NAME', 'PC_COVERAGE']]

st.sidebar.markdown("#### Estados com maior cobertura")
st.sidebar.dataframe(melhores_estados.rename(columns={'LOCAL_NAME': 'Estado', 'PC_COVERAGE': 'Cobertura'}), hide_index=True)

st.sidebar.markdown("#### Estados com menor cobertura")
st.sidebar.dataframe(piores_estados.rename(columns={'LOCAL_NAME': 'Estado', 'PC_COVERAGE': 'Cobertura'}), hide_index=True)

st.sidebar.markdown(":small[Coberturas maiores que 100% são comuns quando a população-alvo está subestimada ou crianças de outros estados se vacinam ali.]")

if len(df_ano[variavel_idh]) >= 3 and len(df_ano['PC_COVERAGE']) >= 3:
    correlacao_matrix = np.corrcoef(df_ano[variavel_idh], df_ano['PC_COVERAGE'])
    correlacao = correlacao_matrix[0, 1]
    n = len(df_ano)
    if correlacao**2 < 1:
        t_obs = correlacao * np.sqrt((n - 2) / (1 - correlacao**2))
        p_valor = 2 * (1 - t.cdf(np.abs(t_obs), df=n - 2))
        subtitulo = f"Coeficiente de correlação: {correlacao:.3f} | P-valor: {p_valor:.3f}"
    elif correlacao == 1 or correlacao == -1:
        p_valor = 0.0
        subtitulo = f"Coeficiente de correlação: {correlacao:.3f} | P-valor: {p_valor:.3f} (Correlação Perfeita)"
    else:
        p_valor = np.nan
        subtitulo = f"Coeficiente de correlação: {correlacao:.3f} | P-valor: N/A"
else:
    correlacao = np.nan
    p_valor = np.nan
    subtitulo = "Dados insuficientes para calcular correlação (mínimo 3 pontos)"

fig = px.scatter(
    df_ano,
    x=variavel_idh,
    y='PC_COVERAGE',
    color='Região',
    hover_data=['LOCAL_NAME'],
    text='LOCAL_NAME' if mostrar_rotulos else None,
    labels={
        variavel_idh: variavel_idh_legenda[variavel_idh],
        'PC_COVERAGE': 'Cobertura De Vacinação Infantil Média (%)'
    },
    title=f'Cobertura De Vacinação Infantil Média vs {variavel_idh_legenda[variavel_idh]} — {ano_selecionado}<br>{subtitulo}',
    height=600
)

if mostrar_rotulos:
    fig.update_traces(textposition='top center')

if mostrar_regressao and not df_ano.empty and len(df_ano[variavel_idh]) >= 2:
    coeficientes = np.polyfit(df_ano[variavel_idh], df_ano['PC_COVERAGE'], 1)
    valores_x = np.linspace(df_ano[variavel_idh].min(), df_ano[variavel_idh].max(), 100)
    valores_y = coeficientes[0] * valores_x + coeficientes[1]
    fig.add_scatter(x=valores_x, y=valores_y, mode='lines', name='Regressão Linear')

st.plotly_chart(fig, use_container_width=True)



st.markdown("<h3 style='font-size: 20px;'>Variações Regionais na Cobertura De Vacinação Infantil</h3><br>"
"<p style='font-size: 18px;'>Disparidades geográficas significativas na cobertura de vacinação infantil existem em todo o Brasil, com taxas mais baixas frequentemente observadas nas regiões Norte e Nordeste em comparação com o Sul e Sudeste. Essas disparidades regionais frequentemente se correlacionam com variações no desenvolvimento socioeconômico e no acesso à infraestrutura de saúde. Essas disparidades regionais persistentes provavelmente contribuem para a relação não linear, pois áreas com MHDI_I (renda) mais baixo podem consistentemente exibir menor cobertura vacinal devido a barreiras sistêmicas. A distribuição desigual de recursos, infraestrutura e profissionais de saúde em todo o Brasil significa que, mesmo com melhorias no MHDI em nível nacional, certas regiões podem continuar a ficar para trás na cobertura vacinal, levando a uma relação geral complexa.</p>"
"<h3 style='font-size: 20px;'>Correlação entre MHDI (IDH Municipal), Status Socioeconômico e Vacinação</h3><br>"
"<p style='font-size: 18px;'>Municípios com MHDI_I (renda) mais baixo e anos esperados de escolaridade consistentemente apresentam taxas de cobertura de vacinação mais baixas. Grupos socioeconômicos mais baixos frequentemente apresentam as maiores taxas de não vacinação. A associação entre MHDI_I e cobertura vacinal pode ser modificada por fatores como a cobertura da atenção primária à saúde. A descoberta consistente de taxas de vacinação mais baixas em áreas com MHDI_I mais baixo e entre grupos socioeconômicos mais baixos sugere fortemente uma correlação positiva entre desenvolvimento humano e cobertura vacinal, mas a modificação por outros fatores indica uma relação não linear onde o desenvolvimento socioeconômico sozinho não garante alta cobertura. Embora um MHDI_I (renda) mais alto geralmente implique melhor acesso à educação, saúde e recursos, o que deve facilitar a vacinação, a influência de fatores como a eficácia da atenção primária e contextos locais específicos pode criar variações nessa relação.</p>"
"<h3 style='font-size: 20px;'>Desafios Enfrentados por Populações Vulneráveis</h3><br>"
"<p style='font-size: 18px;'>Alcançar populações vulneráveis, como comunidades indígenas, aquelas em áreas remotas e famílias de baixa renda, requer estratégias específicas que abordem suas barreiras únicas de acesso. Fatores como distância das unidades de saúde, falta de transporte e restrições financeiras são barreiras significativas. Os desafios específicos enfrentados pelas populações vulneráveis provavelmente contribuem para a não linearidade, criando focos de menor cobertura vacinal mesmo em regiões com MHDI_I (renda) médio relativamente mais alto, pois essas populações podem ser desproporcionalmente afetadas por barreiras de acesso. Embora uma região possa apresentar melhorias gerais no MHDI_I (renda), comunidades marginalizadas dentro dessa região ainda podem enfrentar obstáculos significativos para acessar serviços de saúde como a vacinação, levando a uma relação mais complexa e não uniforme entre desenvolvimento humano e cobertura vacinal.</p>"
"<br>"
"<h3 style='font-size: 20px;'>Análise Temporal da Correlação e P-valor:</h3>"
"<p style='font-size: 18px;'><strong>1996-1999:</strong> Há uma tendência de aumento na correlação positiva entre a Cobertura De Vacinação Infantil Média e o MHDI_I, com o P-valor diminuindo e a Correlação de Pearson aumentando.</p>"
"<p style='font-size: 18px;'><strong>2000-2001:</strong> A correlação permanece positiva em 2000, mas diminui significativamente em 2001, com o P-valor aumentando substancialmente.</p>"
"<p style='font-size: 18px;'><strong>2002-2005:</strong> A correlação torna-se fraca e não estatisticamente significativa, com P-valores elevados, indicando uma relação linear fraca ou inexistente entre as variáveis.</p>"
"<p style='font-size: 18px;'><strong>2006-2011:</strong> Observa-se uma tendência de correlação negativa, sugerindo que a Cobertura De Vacinação Infantil diminui à medida que o MHDI_I aumenta, embora a significância estatística varie.</p>"
"<p style='font-size: 18px;'><strong>2016-2021:</strong> A correlação torna-se positiva novamente, indicando uma tendência de aumento na Cobertura De Vacinação Infantil com o aumento do MHDI_I, com alguns anos mostrando significância estatística (2016, 2020 e 2021).</p>"
"<br>"
"<h3 style='font-size: 20px;'>Interpretação das Transições Suaves nos Gráficos:</h3>"
"<p style='font-size: 18px;'>As transições suaves entre os gráficos e suas linhas de correlação ao longo dos anos fornecem algumas informações importantes sobre a relação entre a Cobertura De Vacinação Infantil Média e o MHDI_I:</p>"
"<p style='font-size: 18px;'><strong>Continuidade e Gradualismo:</strong> As mudanças graduais sugerem que a relação entre as variáveis não sofreu alterações abruptas de um ano para o outro. Em vez disso, as transformações ocorreram de forma mais orgânica e progressiva, refletindo possivelmente mudanças sociais, econômicas e de saúde que se acumularam ao longo do tempo.</p>"
"<p style='font-size: 18px;'><strong>Estabilidade Relativa:</strong> A suavidade nas transições indica que, embora a força e a direção da correlação possam mudar, a relação geral entre as variáveis mantém certa estabilidade. Isso pode sugerir que fatores subjacentes (como infraestrutura de saúde, políticas públicas, desenvolvimento socioeconômico) influenciam ambas as variáveis de maneira consistente ao longo dos anos.</p>"
"<p style='font-size: 18px;'><strong>Tendências de Longo Prazo:</strong> As transições suaves facilitam a identificação de tendências de longo prazo. Por exemplo, podemos observar períodos de aumento ou diminuição gradual na correlação positiva ou negativa, em vez de flutuações aleatórias. Isso ajuda a distinguir mudanças significativas de ruídos ou variações de curto prazo.</p>"
"<p style='font-size: 18px;'><strong>Possíveis Fatores Causais:</strong> A gradualidade das mudanças pode sugerir que fatores causais que afetam a Cobertura De Vacinação Infantil Média e o MHDI_I também evoluem lentamente. Isso pode incluir mudanças demográficas, melhorias na educação, expansão dos serviços de saúde, ou transformações nas políticas de saúde pública.</p>"
"<p style='font-size: 18px;'><strong>Necessidade de Análise Longitudinal:</strong> As transições suaves reforçam a importância de uma análise longitudinal para entender completamente a relação entre as variáveis. Analisar apenas um ou dois anos pode fornecer uma imagem incompleta ou enganosa, enquanto observar a evolução ao longo de vários anos revela padrões e tendências mais robustos.</p>"
"<br>"
"<h2 style='font-size: 24px;'>Conclusão</h2><br>"
"<p style='font-size: 18px;'>A relação entre desenvolvimento humano e cobertura vacinal no Brasil é intrinsecamente complexa e não linear, moldada por uma miríade de determinantes históricos, sociais, culturais, políticos e econômicos interconectados. A trajetória do Programa Nacional de Imunizações, desde sua criação até os desafios contemporâneos, reflete a influência desses múltiplos fatores. A histórica cultura de aceitação de vacinas e a expansão do acesso por meio do SUS e da ESF estabeleceram uma base sólida, mas o surgimento da hesitação vacinal, impulsionado pela desinformação e influências socioculturais, juntamente com a instabilidade política e as crises econômicas, introduziu não linearidades significativas. As disparidades espaciais e socioeconômicas persistem, indicando que o desenvolvimento humano, embora importante, não é o único determinante da cobertura vacinal. Uma abordagem holística é essencial para compreender e enfrentar os desafios na manutenção de altas taxas de cobertura vacinal em todas as camadas da população brasileira.</p>",
unsafe_allow_html=True)