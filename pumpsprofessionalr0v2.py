# app.py
import streamlit as st
import pandas as pd
import math
import time
import numpy as np
from scipy.optimize import root
import graphviz
import matplotlib.pyplot as plt
import io

# Configura a página e o Matplotlib
st.set_page_config(layout="wide", page_title="Análise de Redes Hidráulicas")
plt.style.use('seaborn-v0_8-whitegrid')

# --- BIBLIOTECAS DE DADOS ---
MATERIAIS = {
    "Aço Carbono (novo)": 0.046, "Aço Carbono (pouco uso)": 0.1, "Aço Carbono (enferrujado)": 0.2,
    "Aço Inox": 0.002, "Ferro Fundido": 0.26, "PVC / Plástico": 0.0015, "Concreto": 0.5
}
K_FACTORS = {
    "Entrada de Borda Viva": 0.5, "Entrada Levemente Arredondada": 0.2, "Entrada Bem Arredondada": 0.04,
    "Saída de Tubulação": 1.0, "Válvula Gaveta (Totalmente Aberta)": 0.2, "Válvula Gaveta (1/2 Aberta)": 5.6,
    "Válvula Globo (Totalmente Aberta)": 10.0, "Válvula de Retenção (Tipo Portinhola)": 2.5,
    "Cotovelo 90° (Raio Longo)": 0.6, "Cotovelo 90° (Raio Curto)": 0.9, "Cotovelo 45°": 0.4,
    "Curva de Retorno 180°": 2.2, "Tê (Fluxo Direto)": 0.6, "Tê (Fluxo Lateral)": 1.8,
}
FLUIDOS = { "Água a 20°C": {"rho": 998.2, "nu": 1.004e-6}, "Etanol a 20°C": {"rho": 789.0, "nu": 1.51e-6} }

# --- FUNÇÕES DE CÁLCULO HIDRÁULICO ---
# (Todas as funções de cálculo que já desenvolvemos)
def calcular_perda_serie(lista_trechos, vazao_m3h, fluido_selecionado):
    perda_total = 0
    for trecho in lista_trechos:
        perdas = calcular_perdas_trecho(trecho, vazao_m3h, fluido_selecionado)
        perda_total += perdas["principal"] + perdas["localizada"]
    return perda_total

def calcular_perdas_trecho(trecho, vazao_m3h, fluido_selecionado):
    if vazao_m3h < 0: vazao_m3h = 0
    rugosidade_mm = MATERIAIS[trecho["material"]]
    vazao_m3s, diametro_m = vazao_m3h / 3600, trecho["diametro"] / 1000
    nu = FLUIDOS[fluido_selecionado]["nu"]
    if diametro_m <= 0: return {"principal": 1e12, "localizada": 0, "velocidade": 0}
    area = (math.pi * diametro_m**2) / 4
    velocidade = vazao_m3s / area if area > 0 else 0
    reynolds = (velocidade * diametro_m) / nu if nu > 0 else 0
    fator_atrito = 0
    if reynolds > 4000:
        rugosidade_m = rugosidade_mm / 1000
        if diametro_m <= 0: return {"principal": 1e12, "localizada": 0, "velocidade": 0}
        log_term = math.log10((rugosidade_m / (3.7 * diametro_m)) + (5.74 / reynolds**0.9))
        fator_atrito = 0.25 / (log_term**2)
    elif reynolds > 0:
        fator_atrito = 64 / reynolds
    perda_principal = fator_atrito * (trecho["comprimento"] / diametro_m) * (velocidade**2 / (2 * 9.81))
    k_total_trecho = sum(ac["k"] * ac["quantidade"] for ac in trecho["acessorios"])
    perda_localizada = k_total_trecho * (velocidade**2 / (2 * 9.81))
    return {"principal": perda_principal, "localizada": perda_localizada, "velocidade": velocidade}

def calcular_perdas_paralelo(ramais, vazao_total_m3h, fluido_selecionado):
    num_ramais = len(ramais)
    if num_ramais < 2: return 0, {}
    lista_ramais = list(ramais.values())
    def equacoes_perda(vazoes_parciais_m3h):
        vazao_ultimo_ramal = vazao_total_m3h - sum(vazoes_parciais_m3h)
        if vazao_ultimo_ramal < -0.01: return [1e12] * (num_ramais - 1)
        todas_vazoes = np.append(vazoes_parciais_m3h, vazao_ultimo_ramal)
        perdas = [calcular_perda_serie(ramal, vazao, fluido_selecionado) for ramal, vazao in zip(lista_ramais, todas_vazoes)]
        erros = [perdas[i] - perdas[-1] for i in range(num_ramais - 1)]
        return erros
    chute_inicial = np.full(num_ramais - 1, vazao_total_m3h / num_ramais)
    solucao = root(equacoes_perda, chute_inicial, method='hybr', options={'xtol': 1e-8})
    if not solucao.success: return -1, {}
    vazoes_finais = np.append(solucao.x, vazao_total_m3h - sum(solucao.x))
    perda_final_paralelo = calcular_perda_serie(lista_ramais[0], vazoes_finais[0], fluido_selecionado)
    distribuicao_vazao = {nome_ramal: vazao for nome_ramal, vazao in zip(ramais.keys(), vazoes_finais)}
    return perda_final_paralelo, distribuicao_vazao

def calcular_analise_energetica(vazao_m3h, h_man, eficiencia_bomba_percent, eficiencia_motor_percent, horas_dia, custo_kwh, fluido_selecionado):
    rho = FLUIDOS[fluido_selecionado]["rho"]
    ef_bomba = eficiencia_bomba_percent / 100
    ef_motor = eficiencia_motor_percent / 100
    potencia_eletrica_kW = (vazao_m3h / 3600 * rho * 9.81 * h_man) / (ef_bomba * ef_motor) / 1000 if ef_bomba * ef_motor > 0 else 0
    custo_anual = potencia_eletrica_kW * horas_dia * 30 * 12 * custo_kwh
    return {"potencia_eletrica_kW": potencia_eletrica_kW, "custo_anual": custo_anual}

def criar_funcao_curva(df_curva, col_x, col_y, grau=2):
    df_curva[col_x] = pd.to_numeric(df_curva[col_x], errors='coerce')
    df_curva[col_y] = pd.to_numeric(df_curva[col_y], errors='coerce')
    df_curva = df_curva.dropna(subset=[col_x, col_y])
    if len(df_curva) < grau + 1: return None
    coeficientes = np.polyfit(df_curva[col_x], df_curva[col_y], grau)
    return np.poly1d(coeficientes)

def encontrar_ponto_operacao(sistema, h_geometrica, fluido, func_curva_bomba):
    def curva_sistema(vazao_m3h):
        if vazao_m3h < 0: return h_geometrica
        perda_total = 0
        perda_total += calcular_perda_serie(sistema['antes'], vazao_m3h, fluido)
        perda_par, _ = calcular_perdas_paralelo(sistema['paralelo'], vazao_m3h, fluido)
        if perda_par == -1: return 1e12
        perda_total += perda_par
        perda_total += calcular_perda_serie(sistema['depois'], vazao_m3h, fluido)
        return h_geometrica + perda_total
    def erro(vazao_m3h):
        if vazao_m3h < 0: return 1e12
        return func_curva_bomba(vazao_m3h) - curva_sistema(vazao_m3h)
    solucao = root(erro, 50.0, method='hybr', options={'xtol': 1e-8})
    if solucao.success and solucao.x[0] > 1e-3:
        vazao_op = solucao.x[0]
        altura_op = func_curva_bomba(vazao_op)
        return vazao_op, altura_op, curva_sistema
    else:
        return None, None, curva_sistema

# --- INICIALIZAÇÃO DO ESTADO DA SESSÃO ---
# Define valores padrão para as chaves se elas não existirem
if 'cenario_ativo' not in st.session_state:
    st.session_state.cenario_ativo = False
if 'trechos_antes' not in st.session_state:
    st.session_state.trechos_antes = []
if 'trechos_depois' not in st.session_state:
    st.session_state.trechos_depois = []
if 'ramais_paralelos' not in st.session_state:
    st.session_state.ramais_paralelos = {}
# ... e assim por diante para todas as variáveis que você precisa ...

# --- PLACEHOLDERS: FUNÇÕES DE INTERAÇÃO COM BANCO DE DADOS ---
# !! IMPORTANTE !! Substitua estas funções pela sua lógica real de backend.
def db_listar_projetos(usuario):
    # Exemplo: SELECT DISTINCT nome_projeto FROM projetos WHERE user_id = ?
    return ["Projeto Exemplo A", "Projeto Usina Z"]

def db_listar_cenarios(projeto):
    # Exemplo: SELECT nome_cenario FROM cenarios WHERE nome_projeto = ?
    if projeto == "Projeto Exemplo A":
        return ["Cenário Base", "Cenário Otimizado"]
    return ["Cenário Padrão"]

def db_carregar_cenario(projeto, cenario):
    # Exemplo: SELECT dados_json FROM cenarios WHERE nome_projeto = ? AND nome_cenario = ?
    # Retorna um dicionário com os dados salvos.
    st.warning("Função de carregar dados ainda não implementada. Usando dados de exemplo.")
    return {
        'parametros': {
            'h_geometrica': 25.0,
            'fluido_selecionado': 'Água a 20°C',
            'rend_motor': 90,
            'horas_por_dia': 8.0,
            'tarifa_energia': 0.75
        },
        'sistema': {
            'antes': [{'id': time.time(), 'comprimento': 100.0, 'diametro': 150.0, 'material': 'Aço Carbono (novo)', 'acessorios': []}],
            'paralelo': {},
            'depois': [{'id': time.time()+1, 'comprimento': 200.0, 'diametro': 150.0, 'material': 'Aço Carbono (novo)', 'acessorios': [{'nome': 'Válvula Gaveta (Totalmente Aberta)', 'k': 0.2, 'quantidade': 2}]}]
        },
        'curvas': {
            'altura': pd.DataFrame([{"Vazão (m³/h)": 0, "Altura (m)": 40}, {"Vazão (m³/h)": 50, "Altura (m)": 35}, {"Vazão (m³/h)": 100, "Altura (m)": 25}]),
            'eficiencia': pd.DataFrame([{"Vazão (m³/h)": 0, "Eficiência (%)": 0}, {"Vazão (m³/h)": 50, "Eficiência (%)": 70}, {"Vazão (m³/h)": 100, "Eficiência (%)": 65}])
        }
    }

# --- FUNÇÕES DE CALLBACK DA INTERFACE ---
def carregar_cenario_callback():
    try:
        projeto = st.session_state['projeto_selecionado']
        cenario = st.session_state['cenario_selecionado']
        if not projeto or not cenario:
            st.warning("Por favor, selecione um projeto e um cenário.")
            return

        dados = db_carregar_cenario(projeto, cenario)

        # Popular o session_state
        st.session_state.trechos_antes = dados['sistema']['antes']
        st.session_state.trechos_depois = dados['sistema']['depois']
        st.session_state.ramais_paralelos = dados['sistema']['paralelo']
        st.session_state.curva_altura_df = dados['curvas']['altura']
        st.session_state.curva_eficiencia_df = dados['curvas']['eficiencia']
        
        # Atribuir valores aos widgets da sidebar a partir dos dados carregados
        st.session_state.widget_h_geometrica = dados['parametros']['h_geometrica']
        st.session_state.widget_fluido = dados['parametros']['fluido_selecionado']
        # ... carregar outros parâmetros ...
        
        st.session_state.cenario_ativo = True
        st.success(f"Cenário '{cenario}' carregado com sucesso!")
    except Exception as e:
        st.session_state.cenario_ativo = False
        st.error(f"Não foi possível carregar o cenário. Erro: {e}")

# --- RENDERIZAÇÃO DA INTERFACE (SIDEBAR) ---
# (Assumindo que o login já foi feito e o nome do usuário está em st.session_state['name'])
# Você deve integrar seu código de autenticação aqui.
st.session_state['name'] = "Pedro IA" # Placeholder para o nome do usuário

with st.sidebar:
    st.write(f"Bem-vindo(a), **{st.session_state['name']}**!")
    st.divider()
    
    st.title("🚀 Gestão de Projetos e Cenários")
    
    projetos = db_listar_projetos(st.session_state['name'])
    st.selectbox("Selecione o Projeto", options=projetos, key='projeto_selecionado', index=None)

    if st.session_state.projeto_selecionado:
        cenarios = db_listar_cenarios(st.session_state.projeto_selecionado)
        st.selectbox("Selecione o Cenário", options=cenarios, key='cenario_selecionado', index=None)

    st.button("Carregar Cenário", on_click=carregar_cenario_callback, use_container_width=True)
    # st.button("Deletar Cenário", on_click=deletar_cenario_callback, use_container_width=True) # Adicionar no futuro
    
    st.divider()
    st.text_input("Nome do Projeto", key="save_nome_projeto", placeholder="Nome do projeto existente ou novo")
    st.text_input("Nome do Cenário", key="save_nome_cenario", placeholder="Nome para salvar o cenário atual")
    # st.button("Salvar Cenário", on_click=salvar_cenario_callback, use_container_width=True) # Adicionar no futuro

# --- LÓGICA PRINCIPAL E EXIBIÇÃO ---

st.title("💧 Análise de Redes de Bombeamento com Curva de Bomba")

# "Portão" principal: só executa se um cenário estiver ativo
if not st.session_state.get('cenario_ativo', False):
    st.info("👋 Bem-vindo! Para começar, selecione e carregue um projeto e cenário na barra lateral.")
    st.markdown("---")
    st.write("Ainda não tem um projeto? A funcionalidade de 'Salvar Novo Cenário' será implementada em breve.")

else:
    # Se um cenário está ativo, a aplicação principal é executada.
    try:
        # --- Renderização da Interface Principal (Inputs) ---
        # Usamos chaves diferentes para os widgets para não conflitarem com o estado carregado
        h_geometrica = st.number_input("Altura Geométrica (m)", value=st.session_state.get('widget_h_geometrica', 15.0), key="input_h_geo")
        fluido_selecionado = st.selectbox("Selecione o Fluido", list(FLUIDOS.keys()), index=list(FLUIDOS.keys()).index(st.session_state.get('widget_fluido', 'Água a 20°C')), key="input_fluido")
        # Adicionar aqui os outros inputs como eficiência, horas, etc.
        st.divider()

        # --- Início dos Cálculos ---
        func_curva_bomba = criar_funcao_curva(st.session_state.curva_altura_df, "Vazão (m³/h)", "Altura (m)")
        func_curva_eficiencia = criar_funcao_curva(st.session_state.curva_eficiencia_df, "Vazão (m³/h)", "Eficiência (%)")
        
        if func_curva_bomba is None or func_curva_eficiencia is None:
            st.warning("Dados da curva da bomba insuficientes no cenário carregado.")
            st.stop()
        
        shutoff_head = func_curva_bomba(0)
        if shutoff_head < h_geometrica:
            st.error(f"**Bomba Incompatível:** A altura máxima da bomba ({shutoff_head:.2f} m) é menor que a Altura Geométrica ({h_geometrica:.2f} m).")
            st.stop()

        sistema_atual = {
            'antes': st.session_state.trechos_antes,
            'paralelo': st.session_state.ramais_paralelos,
            'depois': st.session_state.trechos_depois
        }
        
        vazao_op, altura_op, func_curva_sistema = encontrar_ponto_operacao(sistema_atual, h_geometrica, fluido_selecionado, func_curva_bomba)

        if vazao_op is not None and altura_op is not None:
            # --- Exibição de Resultados e Gráficos ---
            st.header("📊 Resultados no Ponto de Operação")
            # ... Colocar aqui toda a lógica de exibição de métricas, gráficos, etc. que já tínhamos ...
            # Exemplo:
            # eficiencia_op = func_curva_eficiencia(vazao_op)
            # resultados_energia = calcular_analise_energetica(...)
            # st.metric("Vazão de Operação", f"{vazao_op:.2f} m³/h")
            # ... etc ...
            st.success("Cálculo realizado com sucesso! (Implementar exibição completa dos resultados aqui)")
            st.write(f"Ponto de Operação Encontrado: Vazão = {vazao_op:.2f} m³/h, Altura = {altura_op:.2f} m")

        else:
            st.error("Não foi possível encontrar um ponto de operação válido com os dados do cenário carregado.")
            # Aqui pode entrar o gráfico mostrando as curvas que não se cruzam, como fizemos antes.

    except Exception as e:
        st.error(f"Ocorreu um erro inesperado durante a execução do cálculo. Detalhe: {e}")
