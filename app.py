import pulp
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import uuid

def tratar_erro_arquivo(arquivo, erro, diretorio):
    """
    Trata erros de I/O para arquivos, fornecendo mensagem padronizada.

    Args:
        arquivo (str): Nome do arquivo.
        erro (Exception): Exceção capturada.
        diretorio (str): Diretório onde o arquivo seria salvo.

    Raises:
        IOError: Repropaga o erro com mensagem clara.
    """
    mensagem = f"Erro ao acessar {arquivo}: {erro}. Verifique permissões no diretório '{diretorio}'."
    raise IOError(mensagem)

def gerar_dados_solares(N, M, seed=42):
    """
    Gera dados fictícios para 12 meses para um consórcio de N residências.

    Args:
        N (int): Número de residências no consórcio.
        M (int): Número de unidades solares disponíveis.
        seed (int): Semente para reprodutibilidade dos dados aleatórios.

    Returns:
        dict: Dados estruturados incluindo consumo, irradiação, especificações de painéis, etc.
    """
    # Define semente para geração de dados consistentes
    np.random.seed(seed)

    # Define fatores sazonais de consumo para cada mês
    sazonalidade = [1.2, 1.2, 1.1, 1.0, 0.9, 0.9, 0.9, 1.0, 1.0, 1.1, 1.2, 1.2]
    # Gera consumo mensal de energia para cada residência (kWh)
    E_cons = [[np.random.uniform(150, 300) * sazonalidade[t] for t in range(12)] for _ in range(N)]

    # Define valores de irradiação solar mensal (kWh/kW/mês)
    irrad = [152.0, 148.0, 145.0, 140.0, 138.0, 135.0, 137.0, 140.0, 145.0, 150.0, 152.0, 154.0]
    # Atribui irradiação a cada unidade solar
    k = [irrad for _ in range(M)]

    # Define potências (kW) e custos de instalação (R$) para cada unidade
    P = [2, 3, 4, 5, 6, 7, 8, 9, 10, 10]
    c = [9000, 13500, 18000, 22500, 27000, 31500, 36000, 40500, 45000, 45000]
    # Define fatores de perda de energia por unidade
    l = [0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.045, 0.04, 0.035]

    # Define perdas de distribuição para cada residência (fixo em 10%)
    l_dist = [0.1] * N

    # Valida os dados de entrada para garantir consistência física
    if any(p <= 0 for p in P): raise ValueError(f"Potências inválidas: {P}")
    if any(c <= 0 for c in c): raise ValueError(f"Custos inválidos: {c}")
    if any(l < 0 or l > 1 for l in l): raise ValueError(f"Perdas inválidas: {l}")
    if any(k_jt <= 0 for k_j in k for k_jt in k_j): raise ValueError(f"Irradiação inválida: {k}")
    if any(e <= 0 for e_i in E_cons for e in e_i): raise ValueError(f"Consumo inválido: {E_cons}")

    return {
        'E_cons': E_cons,
        'P': P,
        'c': c,
        'l': l,
        'l_dist': l_dist,
        'k': k
    }

def configurar_modelo_solar(config, data):
    """
    Configura o modelo MILP para geração compartilhada de energia solar, seguindo as regras da ANEEL.

    Args:
        config (dict): Parâmetros de configuração (N, M, orçamento, tarifa, etc.).
        data (dict): Dados de entrada (consumo, potência, custos, etc.).

    Returns:
        tuple: Modelo PuLP e variáveis de decisão.
    """
    # Inicializa o modelo MILP para maximizar a economia
    prob = pulp.LpProblem("Consorcio_Solar_12Meses", pulp.LpMaximize)
    T = 12

    # Define variáveis de decisão
    x = pulp.LpVariable.dicts("x", range(config['M']), cat="Binary")  # Instalação de unidade (0 ou 1)
    E_eff = pulp.LpVariable.dicts("E_eff", range(T), lowBound=0)  # Energia efetiva gerada
    m = pulp.LpVariable.dicts("m", [(i, t) for i in range(config['N']) for t in range(T)], lowBound=0)  # Energia solar usada
    c = pulp.LpVariable.dicts("c", [(i, t) for i in range(config['N']) for t in range(T)], lowBound=0)  # Créditos gerados
    r = pulp.LpVariable.dicts("r", [(i, t) for i in range(config['N']) for t in range(T)], lowBound=0)  # Energia da rede
    u = pulp.LpVariable.dicts("u", [(i, t) for i in range(config['N']) for t in range(T)], lowBound=0)  # Créditos usados
    C = pulp.LpVariable.dicts("C", [(i, t) for i in range(config['N']) for t in range(T+1)], lowBound=0)  # Estoque de créditos
    a = pulp.LpVariable.dicts("a", [(i, t) for i in range(config['N']) for t in range(T)], lowBound=0, upBound=1)  # Fração de alocação
    E_alloc = pulp.LpVariable.dicts("E_alloc", [(i, t) for i in range(config['N']) for t in range(T)], lowBound=0)  # Energia alocada

    # Calcula custos operacionais mensais e de garantia
    custo_op_mensal = pulp.lpSum(0.01 * data['c'][j] / 12 * x[j] for j in range(config['M']))
    custo_garantia = config['custo_garantia_por_kw'] * pulp.lpSum(data['P'][j] * x[j] for j in range(config['M']))

    # Define a função objetivo: maximizar economia menos custos
    prob += (
        config['r'] * pulp.lpSum(m[i, t] + u[i, t] for i in range(config['N']) for t in range(T)) -
        custo_op_mensal * T -
        config['taxa_disponibilidade'] * config['N'] * T +
        config['gamma'] * pulp.lpSum(c[i, t] for i in range(config['N']) for t in range(T)) -
        config['r_rede'] * pulp.lpSum(c[i, t] for i in range(config['N']) for t in range(T)) -
        custo_garantia
    )

    # Adiciona restrições para cada mês
    for t in range(T):
        # Energia efetiva gerada após perdas
        prob += E_eff[t] == pulp.lpSum(x[j] * (1 - data['l'][j]) * data['k'][j][t] * data['P'][j]
                                      for j in range(config['M'])), f"Energia_Efetiva_{t}"
        # Limita geração a 1,5x o consumo total
        prob += E_eff[t] <= 1.5 * sum(data['E_cons'][i][t] for i in range(config['N'])), f"Limite_Geracao_{t}"
        # Garante geração mínima (alpha * consumo total)
        prob += E_eff[t] >= config['alpha'] * sum(data['E_cons'][i][t] for i in range(config['N'])), f"Demanda_Minima_{t}"
        # Soma das frações de alocação é 1
        prob += pulp.lpSum(a[i, t] for i in range(config['N'])) == 1, f"Alocacao_Total_{t}"
        # Energia alocada total igual à energia efetiva
        prob += pulp.lpSum(E_alloc[i, t] for i in range(config['N'])) == E_eff[t], f"Alocacao_Energia_{t}"

        # Restrições específicas por residência
        for i in range(config['N']):
            prob += E_alloc[i, t] <= a[i, t] * 1e6, f"E_alloc_Upper_{i}_{t}"
            prob += E_alloc[i, t] >= 0, f"E_alloc_Lower_{i}_{t}"
            prob += E_alloc[i, t] <= E_eff[t], f"E_alloc_Max_{i}_{t}"
            prob += m[i, t] <= data['E_cons'][i][t], f"Limite_Consumo_{i}_{t}"
            prob += m[i, t] <= (1 - data['l_dist'][i]) * E_alloc[i, t], f"Limite_Atribuicao_{i}_{t}"
            prob += c[i, t] == (1 - data['l_dist'][i]) * E_alloc[i, t] - m[i, t], f"Creditos_{i}_{t}"
            prob += data['E_cons'][i][t] == m[i, t] + r[i, t] + u[i, t], f"Demanda_Coberta_{i}_{t}"
            prob += C[i, t+1] == C[i, t] + c[i, t] - u[i, t], f"Estoque_Creditos_{i}_{t}"
            prob += u[i, t] <= C[i, t], f"Limite_Uso_Creditos_{i}_{t}"
            prob += a[i, t] == 1.0 / config['N'], f"Alocacao_Equitativa_{i}_{t}"
            # Garante créditos iguais entre residências
            for j in range(i + 1, config['N']):
                prob += c[i, t] == c[j, t], f"Creditos_Iguais_{i}_{j}_{t}"

    # Restrições de orçamento e limite de unidades
    prob += pulp.lpSum(data['c'][j] * x[j] for j in range(config['M'])) <= config['B'], "Orcamento"
    prob += pulp.lpSum(x[j] for j in range(config['M'])) <= config['max_units'], "Limite_Unidades"

    # Inicializa estoque de créditos como zero
    for i in range(config['N']):
        prob += C[i, 0] == 0, f"Estoque_Inicial_{i}"

    return prob, x, E_eff, m, c, r, u, C, a, E_alloc

def salvar_grafico(caminho, dpi=300, bbox_inches='tight'):
    """
    Salva um gráfico com configurações padrão.

    Args:
        caminho (str): Caminho onde o gráfico será salvo.
        dpi (int): Resolução do gráfico.
        bbox_inches (str): Ajuste de margens.
    """
    plt.tight_layout()
    plt.savefig(caminho, dpi=dpi, bbox_inches=bbox_inches)
    plt.close()

def resolver_e_salvar(prob, x, E_eff, m, c, r, u, C, a, E_alloc, config, data, run_sensitivity=True, run_id=None):
    """
    Resolve o modelo MILP, salva resultados e gera visualizações.

    Args:
        prob: Modelo PuLP.
        x, E_eff, ...: Variáveis de decisão.
        config (dict): Parâmetros de configuração.
        data (dict): Dados de entrada.
        run_sensitivity (bool): Se True, realiza análise de sensibilidade.
        run_id (str): Identificador único para a execução.

    Returns:
        dict: Resultados incluindo status, economia, métricas de energia, etc.
    """
    # Gera um ID único para a execução, se não fornecido
    if run_id is None:
        run_id = str(uuid.uuid4())

    print(f"Iniciando execução {run_id} com tarifa={config['r']}")

    # Cria diretório para relatórios
    relatorios_dir = f"relatorios_{run_id}"
    try:
        os.makedirs(relatorios_dir, exist_ok=True)
    except OSError as e:
        raise OSError(f"Erro ao criar diretório 'relatorios': {e}. Verifique permissões.")

    # Resolve o modelo MILP
    try:
        prob.solve(pulp.PULP_CBC_CMD(timeLimit=300, msg=1))
    except Exception as e:
        print(f"Erro no solver (execução {run_id}): {e}")
        raise

    # Obtém status do solver
    status = pulp.LpStatus[prob.status]
    resultados = {
        'status': status,
        'run_id': run_id,
        'tarifa': config['r'],
        'residencias': [],
        'totais': {
            'economia': [0]*12,
            'creditos': [0]*12,
            'creditos_usados': [0]*12,
            'energia_gerada': [0]*12,
            'energia_consumida': [0]*12,
            'uso_rede': [0]*12,
            'custo_operacional': [0]*12
        }
    }

    if status == "Optimal":
        # Calcula custo total de instalação e unidades instaladas
        total_custo_instalacao = sum(data['c'][j] * pulp.value(x[j]) for j in range(config['M']))
        unidades_instaladas = [j for j in range(config['M']) if pulp.value(x[j]) == 1]
        potencia_total = sum(data['P'][j] for j in unidades_instaladas)

        # Armazena dados diagnósticos para análise
        diagnostico = {
            'l_dist': data['l_dist'],
            'E_cons': data['E_cons'],
            'a': [[pulp.value(a[i, t]) or 0 for t in range(12)] for i in range(config['N'])],
            'm': [[pulp.value(m[i, t]) or 0 for t in range(12)] for i in range(config['N'])]
        }

        # Pré-calcula consumo total por mês
        total_consumo_por_mes = [sum(data['E_cons'][i][t] for i in range(config['N'])) for t in range(12)]

        # Processa resultados para cada mês
        for t in range(12):
            total_energia_gerada = pulp.value(E_eff[t]) or 0
            total_consumo = total_consumo_por_mes[t]
            total_creditos = 0
            total_creditos_usados = 0
            total_economia = 0
            total_uso_rede = 0
            custo_op_mensal = sum(0.01 * data['c'][j] / 12 * pulp.value(x[j]) for j in range(config['M']))

            # Coleta resultados por residência
            for i in range(config['N']):
                energia_solar = pulp.value(m[i, t]) or 0
                energia_rede = pulp.value(r[i, t]) or 0
                creditos = pulp.value(c[i, t]) or 0
                creditos_usados = pulp.value(u[i, t]) or 0
                alocacao = pulp.value(a[i, t]) or 0
                economia = config['r'] * (energia_solar + creditos_usados) - config['taxa_disponibilidade']
                if t == 0:
                    resultados['residencias'].append({
                        'consumo': [], 'energia_solar': [], 'energia_rede': [],
                        'creditos': [], 'economia': [], 'creditos_usados': [], 'alocacao': []
                    })
                resultados['residencias'][i]['consumo'].append(data['E_cons'][i][t])
                resultados['residencias'][i]['energia_solar'].append(energia_solar)
                resultados['residencias'][i]['energia_rede'].append(energia_rede)
                resultados['residencias'][i]['creditos'].append(creditos)
                resultados['residencias'][i]['economia'].append(economia)
                resultados['residencias'][i]['creditos_usados'].append(creditos_usados)
                resultados['residencias'][i]['alocacao'].append(alocacao)
                total_creditos += creditos
                total_creditos_usados += creditos_usados
                total_economia += economia
                total_uso_rede += energia_rede

            # Armazena totais mensais
            resultados['totais']['energia_gerada'][t] = total_energia_gerada
            resultados['totais']['energia_consumida'][t] = total_consumo
            resultados['totais']['uso_rede'][t] = total_uso_rede
            resultados['totais']['creditos'][t] = total_creditos
            resultados['totais']['creditos_usados'][t] = total_creditos_usados
            resultados['totais']['economia'][t] = total_economia
            resultados['totais']['custo_operacional'][t] = custo_op_mensal

        # Armazena metadados da execução
        resultados['unidades_instaladas'] = unidades_instaladas
        resultados['total_custo_instalacao'] = total_custo_instalacao
        resultados['potencia_total'] = potencia_total

        # Define rótulos dos meses
        meses = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']

        try:
            # Gráfico: Desempenho do Consórcio Solar
            plt.figure(figsize=(12, 6))
            plt.plot(meses, resultados['totais']['energia_gerada'], label='Energia Gerada (kWh)', marker='o', color='blue')
            plt.plot(meses, resultados['totais']['energia_consumida'], label='Energia Consumida (kWh)', marker='x', color='green')
            plt.plot(meses, resultados['totais']['creditos_usados'], label='Créditos Usados (kWh)', marker='s', color='orange')
            ax1 = plt.gca()
            ax2 = ax1.twinx()
            ax2.plot(meses, resultados['totais']['economia'], label='Economia (R$)', marker='^', color='purple')
            ax1.set_xlabel('Mês')
            ax1.set_ylabel('Energia (kWh)', color='blue')
            ax2.set_ylabel('Economia (R$)', color='purple')
            ax1.set_title('Desempenho do Consórcio Solar')
            ax1.legend(loc='upper left', bbox_to_anchor=(0, 1))
            ax2.legend(loc='upper right', bbox_to_anchor=(1, 1))
            ax1.grid(True, linestyle='--', alpha=0.7)
            ax1.set_ylim(0, max(max(resultados['totais']['energia_gerada']), max(resultados['totais']['energia_consumida'])) * 1.1)
            ax2.set_ylim(0, max(resultados['totais']['economia']) * 1.1)
            salvar_grafico(os.path.join(relatorios_dir, f"desempenho_consorcio_{run_id}.png"))

            # Gráfico: Energia Gerada vs. Consumida
            plt.figure(figsize=(10, 6))
            plt.plot(meses, resultados['totais']['energia_gerada'], label='Energia Gerada (kWh)', marker='o', color='blue')
            plt.plot(meses, resultados['totais']['energia_consumida'], label='Energia Consumida (kWh)', marker='x', color='green')
            plt.xlabel('Mês')
            plt.ylabel('Energia (kWh)')
            plt.title('Energia Gerada vs. Consumida')
            plt.legend(loc='best')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.ylim(0, max(max(resultados['totais']['energia_gerada']), max(resultados['totais']['energia_consumida'])) * 1.1)
            salvar_grafico(os.path.join(relatorios_dir, f"energia_gerada_vs_consumida_{run_id}.png"))

            # Gráfico: Composição da Economia Mensal
            economia_solar = [sum(resultados['residencias'][i]['energia_solar'][t] * config['r'] for i in range(config['N'])) for t in range(12)]
            economia_creditos = [sum(resultados['residencias'][i]['creditos_usados'][t] * config['r'] for i in range(config['N'])) for t in range(12)]
            economia_total = [resultados['totais']['economia'][t] for t in range(12)]
            plt.figure(figsize=(10, 6))
            plt.stackplot(meses, economia_solar, economia_creditos, labels=['Economia Solar', 'Economia Créditos'], colors=['#1f77b4', '#ff7f0e'])
            plt.plot(meses, economia_total, label='Economia Total (R$)', color='black', linestyle='--', linewidth=2)
            plt.xlabel('Mês')
            plt.ylabel('Economia (R$)')
            plt.title('Composição da Economia Mensal')
            plt.legend(loc='upper left')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.ylim(0, max(economia_total) * 1.1)
            salvar_grafico(os.path.join(relatorios_dir, f"composicao_economia_{run_id}.png"))

            # Gráfico: Créditos Gerados por Residência
            creditos_medios = [resultados['residencias'][0]['creditos'][t] for t in range(12)]
            plt.figure(figsize=(10, 6))
            plt.plot(meses, creditos_medios, label='Créditos por Residência (kWh)', marker='o', color='teal')
            plt.xlabel('Mês')
            plt.ylabel('Créditos Gerados (kWh)')
            plt.title('Créditos Gerados por Residência')
            plt.legend(loc='best')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.ylim(0, max(creditos_medios + [0]) * 1.1 or 1)
            salvar_grafico(os.path.join(relatorios_dir, f"creditos_por_casa_{run_id}.png"))

            # Gráfico: Dependência da Rede Elétrica
            plt.figure(figsize=(10, 6))
            plt.bar(meses, resultados['totais']['uso_rede'], color='orange', edgecolor='black')
            plt.xlabel('Mês')
            plt.ylabel('Uso da Rede (kWh)')
            plt.title('Dependência da Rede Elétrica')
            plt.grid(True, axis='y', linestyle='--', alpha=0.7)
            plt.ylim(0, max(resultados['totais']['uso_rede']) * 1.1)
            salvar_grafico(os.path.join(relatorios_dir, f"uso_rede_{run_id}.png"))

            # Gráfico: Energia Gerada vs. Irradiação Solar
            irrad = data['k'][0]
            irrad_norm = [i / max(irrad) * max(resultados['totais']['energia_gerada']) for i in irrad]
            plt.figure(figsize=(10, 6))
            plt.plot(meses, resultados['totais']['energia_gerada'], label='Energia Gerada (kWh)', marker='o', color='blue')
            plt.plot(meses, irrad_norm, label='Irradiação Normalizada (kWh/kW/mês)', marker='x', color='red')
            plt.xlabel('Mês')
            plt.ylabel('Energia (kWh)')
            plt.title('Energia Gerada vs. Irradiação Solar')
            plt.legend(loc='best')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.ylim(0, max(max(resultados['totais']['energia_gerada']), max(irrad_norm)) * 1.1)
            salvar_grafico(os.path.join(relatorios_dir, f"geracao_vs_irrad_{run_id}.png"))

            # Gráfico: Distribuição da Economia por Residência
            economia_por_casa = [[resultados['residencias'][i]['economia'][t] for t in range(12)] for i in range(config['N'])]
            plt.figure(figsize=(12, 6))
            plt.boxplot(economia_por_casa, tick_labels=[f'Casa {i+1}' for i in range(config['N'])], patch_artist=True, boxprops=dict(facecolor='lightblue'))
            plt.xlabel('Residência')
            plt.ylabel('Economia (R$)')
            plt.title('Distribuição da Economia por Residência')
            plt.grid(True, axis='y', linestyle='--', alpha=0.7)
            plt.xticks(rotation=45)
            salvar_grafico(os.path.join(relatorios_dir, f"economia_por_casa_{run_id}.png"))

            # Gráfico: Estoque de Créditos Acumulados
            estoque_creditos = [0] * 13
            for t in range(12):
                estoque_creditos[t+1] = estoque_creditos[t] + resultados['totais']['creditos'][t] - resultados['totais']['creditos_usados'][t]
            plt.figure(figsize=(10, 6))
            plt.plot(meses, estoque_creditos[1:], label='Estoque de Créditos (kWh)', marker='o', color='purple')
            plt.xlabel('Mês')
            plt.ylabel('Créditos (kWh)')
            plt.title('Estoque de Créditos Acumulados')
            plt.legend(loc='best')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.ylim(0, max(estoque_creditos[1:] + [0]) * 1.1 or 1)
            salvar_grafico(os.path.join(relatorios_dir, f"estoque_creditos_{run_id}.png"))

            # Gráfico: Consumo por Residência
            plt.figure(figsize=(12, 6))
            for i in range(config['N']):
                plt.plot(meses, resultados['residencias'][i]['consumo'], label=f'Casa {i+1}', alpha=0.5)
            plt.xlabel('Mês')
            plt.ylabel('Consumo (kWh)')
            plt.title('Consumo por Residência')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.ylim(0, max([max(resultados['residencias'][i]['consumo']) for i in range(config['N'])]) * 1.1)
            salvar_grafico(os.path.join(relatorios_dir, f"consumo_por_casa_{run_id}.png"))

            # Gráfico: Uso de Rede por Residência
            plt.figure(figsize=(12, 6))
            for i in range(config['N']):
                plt.plot(meses, resultados['residencias'][i]['energia_rede'], label=f'Casa {i+1}', alpha=0.5)
            plt.xlabel('Mês')
            plt.ylabel('Uso da Rede (kWh)')
            plt.title('Uso de Rede por Residência')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.ylim(0, max([max(resultados['residencias'][i]['energia_rede']) for i in range(config['N'])]) * 1.1)
            salvar_grafico(os.path.join(relatorios_dir, f"uso_rede_por_casa_{run_id}.png"))

            # Gráfico: Créditos Usados por Residência
            plt.figure(figsize=(12, 6))
            for i in range(config['N']):
                plt.plot(meses, resultados['residencias'][i]['creditos_usados'], label=f'Casa {i+1}', alpha=0.5)
            plt.xlabel('Mês')
            plt.ylabel('Créditos Usados (kWh)')
            plt.title('Créditos Usados por Residência')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.ylim(0, max([max(resultados['residencias'][i]['creditos_usados']) for i in range(config['N'])]) * 1.1 or 1)
            salvar_grafico(os.path.join(relatorios_dir, f"creditos_usados_por_casa_{run_id}.png"))

            # Gráfico: Validação da Cobertura de Demanda
            plt.figure(figsize=(12, 6))
            for i in range(config['N']):
                consumo = resultados['residencias'][i]['consumo']
                energia_solar = resultados['residencias'][i]['energia_solar']
                creditos_usados = resultados['residencias'][i]['creditos_usados']
                energia_rede = resultados['residencias'][i]['energia_rede']
                plt.plot(meses, consumo, label=f'Consumo Casa {i+1}', linestyle='-', alpha=0.3)
                plt.plot(meses, [energia_solar[t] + creditos_usados[t] + energia_rede[t] for t in range(12)],
                         label=f'Soma Casa {i+1}', linestyle='--', alpha=0.3)
            plt.xlabel('Mês')
            plt.ylabel('Energia (kWh)')
            plt.title('Validação: Consumo = Energia Solar + Créditos Usados + Rede')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.ylim(0, max([max(resultados['residencias'][i]['consumo']) for i in range(config['N'])]) * 1.1)
            salvar_grafico(os.path.join(relatorios_dir, f"validacao_demanda_{run_id}.png"))

            # Gráfico: Evolução Mensal da Economia Líquida
            plt.figure(figsize=(8, 6))
            plt.plot(meses, resultados['totais']['economia'], marker='o', linestyle='-', color='navy', linewidth=2, markersize=8,
                     label=f'Total Anual: R${sum(resultados["totais"]["economia"]):,.2f}')
            plt.title('Evolução Mensal da Economia Líquida', fontsize=14, pad=10)
            plt.xlabel('Mês', fontsize=12)
            plt.ylabel('Economia Líquida (R$)', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(fontsize=10)
            plt.xticks(rotation=45)
            salvar_grafico(os.path.join(relatorios_dir, f"economia_mensal_{run_id}.png"))

            # Realiza análise de sensibilidade, se habilitada
            if run_sensitivity:
                sens = analise_sensibilidade(config, data, 'r', [0.80, 0.85, 0.90], run_id)
                plt.figure(figsize=(8, 6))
                plt.plot([s['value'] for s in sens], [s['economia'] for s in sens], marker='o', color='purple')
                plt.xlabel('Tarifa (R$/kWh)')
                plt.ylabel('Economia Anual (R$)')
                plt.title('Sensibilidade à Tarifa')
                plt.grid(True, linestyle='--', alpha=0.7)
                salvar_grafico(os.path.join(relatorios_dir, f"sensibilidade_tarifa_{run_id}.png"))

                # Salva resultados da análise de sensibilidade
                try:
                    with open(os.path.join(relatorios_dir, f"sensibilidade_{run_id}.csv"), "w", newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(["Tarifa_R$_por_kWh", "Economia_Anual_BRL"])
                        for s in sens:
                            writer.writerow([f"{s['value']:.2f}", f"{s['economia']:.2f}"])
                except IOError as e:
                    tratar_erro_arquivo(f"sensibilidade_{run_id}.csv", e, relatorios_dir)

            # Salva CSV de créditos por residência
            try:
                with open(os.path.join(relatorios_dir, f"creditos_por_casa_{run_id}.csv"), "w", newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["Mês"] + [f"Casa {i+1}" for i in range(config['N'])])
                    for t in range(12):
                        writer.writerow([t+1] + [f"{resultados['residencias'][i]['creditos'][t]:.2f}"
                                                for i in range(config['N'])])
            except IOError as e:
                tratar_erro_arquivo(f"creditos_por_casa_{run_id}.csv", e, relatorios_dir)

            # Salva CSV de diagnóstico
            try:
                with open(os.path.join(relatorios_dir, f"diagnostico_{run_id}.csv"), "w", newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["Casa", "l_dist"] + [f"E_cons_Mês_{t+1}" for t in range(12)] +
                                    [f"a_Mês_{t+1}" for t in range(12)] + [f"m_Mês_{t+1}" for t in range(12)])
                    for i in range(config['N']):
                        writer.writerow([f"Casa {i+1}", f"{diagnostico['l_dist'][i]:.4f}"] +
                                        [f"{diagnostico['E_cons'][i][t]:.2f}" for t in range(12)] +
                                        [f"{diagnostico['a'][i][t]:.4f}" for t in range(12)] +
                                        [f"{diagnostico['m'][i][t]:.2f}" for t in range(12)])
            except IOError as e:
                tratar_erro_arquivo(f"diagnostico_{run_id}.csv", e, relatorios_dir)

            # Salva CSV de relatório por residência
            try:
                with open(os.path.join(relatorios_dir, f"relatorio_por_casa_{run_id}.csv"), "w", newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["Casa", "Mês", "Consumo_kWh", "Uso_Rede_kWh", "Créditos_Gerados_kWh", "Créditos_Usados_kWh"])
                    for i in range(config['N']):
                        for t in range(12):
                            writer.writerow([f"Casa {i+1}", t+1,
                                            f"{resultados['residencias'][i]['consumo'][t]:.2f}",
                                            f"{resultados['residencias'][i]['energia_rede'][t]:.2f}",
                                            f"{resultados['residencias'][i]['creditos'][t]:.2f}",
                                            f"{resultados['residencias'][i]['creditos_usados'][t]:.2f}"])
            except IOError as e:
                tratar_erro_arquivo(f"relatorio_por_casa_{run_id}.csv", e, relatorios_dir)

            # Salva CSV de resultados mensais
            try:
                with open(os.path.join(relatorios_dir, f"resultados_mensais_{run_id}.csv"), "w", newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["Mês", "Energia_Gerada_kWh", "Energia_Consumida_kWh", "Uso_Rede_kWh",
                                     "Créditos_Gerados_kWh", "Créditos_Usados_kWh", "Economia_BRL"])
                    for t in range(12):
                        writer.writerow([t+1,
                                        f"{resultados['totais']['energia_gerada'][t]:.2f}",
                                        f"{resultados['totais']['energia_consumida'][t]:.2f}",
                                        f"{resultados['totais']['uso_rede'][t]:.2f}",
                                        f"{resultados['totais']['creditos'][t]:.2f}",
                                        f"{resultados['totais']['creditos_usados'][t]:.2f}",
                                        f"{resultados['totais']['economia'][t]:.2f}"])
                    writer.writerow(["Total",
                                     f"{sum(resultados['totais']['energia_gerada']):.2f}",
                                     f"{sum(resultados['totais']['energia_consumida']):.2f}",
                                     f"{sum(resultados['totais']['uso_rede']):.2f}",
                                     f"{sum(resultados['totais']['creditos']):.2f}",
                                     f"{sum(resultados['totais']['creditos_usados']):.2f}",
                                     f"{sum(resultados['totais']['economia']):.2f}"])
            except IOError as e:
                tratar_erro_arquivo(f"resultados_mensais_{run_id}.csv", e, relatorios_dir)

            # Salva resultados em JSON
            try:
                with open(os.path.join(relatorios_dir, f"resultados_consorcio_12meses_{run_id}.json"), "w") as f:
                    json.dump(resultados, f, indent=2)
            except IOError as e:
                tratar_erro_arquivo(f"resultados_consorcio_12meses_{run_id}.json", e, relatorios_dir)

        except Exception as e:
            print(f"Erro ao gerar saídas (execução {run_id}): {e}. Verifique permissões no diretório '{relatorios_dir}'.")
            raise

    return resultados

def analise_sensibilidade(config, data, param, values, parent_run_id):
    """
    Realiza análise de sensibilidade para um parâmetro especificado.

    Args:
        config (dict): Parâmetros de configuração.
        data (dict): Dados de entrada.
        param (str): Parâmetro a variar (ex.: 'r' para tarifa).
        values (list): Valores a testar.
        parent_run_id (str): ID da execução principal.

    Returns:
        list: Resultados com economia anual para cada valor do parâmetro.
    """
    resultados = []
    original = config[param]
    # Testa cada valor, pulando a configuração atual
    for v in values:
        if v == config[param]:
            continue
        config[param] = v
        prob, x, E_eff, m, c, r, u, C, a, E_alloc = configurar_modelo_solar(config, data)
        res = resolver_e_salvar(prob, x, E_eff, m, c, r, u, C, a, E_alloc, config, data,
                               run_sensitivity=False, run_id=f"{parent_run_id}_sens_{v}")
        if res['status'] == "Optimal":
            resultados.append({
                'value': v,
                'economia': sum(res['totais']['economia'])
            })
        config[param] = original
    return resultados

def principal():
    """
    Executa a otimização do consórcio solar.
    """
    # Carrega configuração do arquivo JSON
    try:
        with open('config.json') as f:
            config = json.load(f)
    except IOError as e:
        print(f"Erro ao carregar config.json: {e}. Verifique se o arquivo existe no diretório atual.")
        raise

    # Valida chaves obrigatórias no config
    chaves_obrigatorias = ['N', 'M', 'B', 'r', 'r_rede', 'gamma', 'taxa_disponibilidade',
                           'custo_garantia_por_kw', 'alpha', 'max_units', 'seed']
    chaves_faltando = [chave for chave in chaves_obrigatorias if chave not in config]
    if chaves_faltando:
        raise KeyError(f"Chaves faltando em config.json: {chaves_faltando}")

    # Gera dados fictícios
    data = gerar_dados_solares(config['N'], config['M'], config['seed'])
    # Configura o modelo MILP
    prob, x, E_eff, m, c, r, u, C, a, E_alloc = configurar_modelo_solar(config, data)
    # Gera um ID único para a execução
    run_id = str(uuid.uuid4())
    # Resolve e salva resultados
    resultados = resolver_e_salvar(prob, x, E_eff, m, c, r, u, C, a, E_alloc, config, data,
                                 run_sensitivity=False, run_id=run_id)

    # Exibe resumo se a solução for ótima
    if resultados['status'] == "Optimal":
        print(f"Status: {resultados['status']}")
        print(f"Unidades Instaladas: {resultados['unidades_instaladas']}")
        print(f"Potência Total: {resultados['potencia_total']:.2f} kW")
        print(f"Custo Total Instalação: R${resultados['total_custo_instalacao']:.2f}")
        print("Meses com créditos usados:")
        for t in range(12):
            total_usado = resultados['totais']['creditos_usados'][t]
            if total_usado > 0:
                print(f"Mês {t+1}: {total_usado:.2f} kWh")
        print("\nResultados Mensais:")
        for t in range(12):
            print(f"Mês {t+1}: Energia Gerada={resultados['totais']['energia_gerada'][t]:.2f} kWh, "
                  f"Energia Consumida={resultados['totais']['energia_consumida'][t]:.2f} kWh, "
                  f"Uso da Rede={resultados['totais']['uso_rede'][t]:.2f} kWh, "
                  f"Créditos Gerados={resultados['totais']['creditos'][t]:.2f} kWh, "
                  f"Créditos Usados={resultados['totais']['creditos_usados'][t]:.2f} kWh, "
                  f"Economia=R${resultados['totais']['economia'][t]:.2f}")
    else:
        print(f"Status: {resultados['status']}. Solução não encontrada.")

if __name__ == "__main__":
    principal()