import numpy as np
from pulp import LpProblem, LpVariable, LpMaximize, lpSum, LpContinuous, PULP_CBC_CMD

def cenario_3_multiobjetivo(params):
    """
    Otimiza alocação solar em geração compartilhada conforme a legislação brasileira.

    Notas:
    - A Tarifa Social de Energia Elétrica garante descontos tarifários para famílias de baixa renda, mas não prioridade na alocação de energia solar.
    - Neste modelo, todas as residências são tratadas de forma equitativa, respeitando o princípio de distribuição proporcional previsto na Lei nº 14.300/2022.
    - A alocação de energia prioriza o uso eficiente da geração solar, minimiza o uso da rede elétrica e respeita os limites de injeção.

    Resumo dos objetivos:
    - Distribuir a energia solar gerada de forma justa entre os participantes.
    - Minimizar o consumo de energia da rede pública.
    - Respeitar os limites operacionais do sistema de compensação.
    """
    # Parâmetros
    num_residencias = params['num_residencias']
    num_horas = params['num_horas']
    demanda_residencias = params['demanda_residencias']
    geracao_solar = params['geracao_solar']
    eficiencia_rede = params['eficiencia_rede']
    demanda_base_ajustada = params['demanda_base_ajustada']
    demandas_minimas_residencias = params['demandas_minimas_residencias']
    limite_injecao_rede = params['limite_injecao_rede']
    tarifas_residencias = params['tarifas_residencias']
    pesos_objetivos_c3 = [0.7, 0.05, 0.1, 0.15]  # Mantidos pesos para diferentes objetivos

    # Pesos iguais: nenhuma diferenciação entre Tarifa Social e demais
    pesos_prioridade = [1.0 for _ in range(num_residencias)]
    print(f"Pesos de prioridade (sem distinção Tarifa Social): {[round(p, 2) for p in pesos_prioridade]}")

    # Criar modelo
    model = LpProblem("Cenario_3", LpMaximize)

    # Variáveis
    x = LpVariable.dicts("x", ((i, t) for i in range(num_residencias) for t in range(num_horas)), lowBound=0, cat=LpContinuous)
    u = LpVariable.dicts("u", ((i, t) for i in range(num_residencias) for t in range(num_horas)), lowBound=0, cat=LpContinuous)
    z = LpVariable.dicts("z", (t for t in range(num_horas)), lowBound=0, cat=LpContinuous)
    v = LpVariable.dicts("v", (i for i in range(num_residencias)), lowBound=0, cat=LpContinuous)
    creditos = LpVariable.dicts("creditos", (t for t in range(num_horas)), lowBound=0, cat=LpContinuous)

    # Objetivos normalizados
    Z_1 = lpSum(pesos_prioridade[i] * x[i, t] * demanda_residencias[i][t] for i in range(num_residencias) for t in range(num_horas)) / np.sum(demanda_residencias)
    Z_2 = lpSum(v[i] for i in range(num_residencias)) / (num_residencias * np.mean(demanda_residencias))
    Z_3 = lpSum(u[i, t] for i in range(num_residencias) for t in range(num_horas)) / np.sum(demanda_residencias)
    Z_4 = lpSum(tarifas_residencias[i][t] * u[i, t] for i in range(num_residencias) for t in range(num_horas)) / np.sum(tarifas_residencias * demanda_residencias)
    model += pesos_objetivos_c3[0] * Z_1 - pesos_objetivos_c3[1] * Z_2 - pesos_objetivos_c3[2] * Z_3 - pesos_objetivos_c3[3] * Z_4

    # Restrições
    for t in range(num_horas):
        model += lpSum(x[i, t] * demanda_residencias[i][t] for i in range(num_residencias)) <= eficiencia_rede[t] * geracao_solar[t]
        model += lpSum(x[i, t] * demanda_residencias[i][t] for i in range(num_residencias)) <= 0.99 * geracao_solar[t]
        model += lpSum(x[i, t] * demanda_residencias[i][t] for i in range(num_residencias)) + z[t] <= eficiencia_rede[t] * geracao_solar[t]
        model += z[t] <= limite_injecao_rede[t]
        model += creditos[t] == z[t] * 0.9
        if 15 <= t <= 18:
            model += lpSum(u[i, t] for i in range(num_residencias)) <= 0.1 * sum(demanda_residencias[i][t] for i in range(num_residencias))

    for i in range(num_residencias):
        model += lpSum(x[i, t] * demanda_residencias[i][t] for t in range(num_horas)) >= min(demanda_base_ajustada, demandas_minimas_residencias[i]) * 0.5
        for t in range(num_horas):
            credito_por_casa = sum(creditos[s] for s in range(t)) / num_residencias
            model += x[i, t] * demanda_residencias[i][t] + u[i, t] >= demanda_residencias[i][t] - credito_por_casa

    # Equidade
    total_alocado = lpSum(x[i, t] * demanda_residencias[i][t] for i in range(num_residencias) for t in range(num_horas))
    media_alocado = total_alocado / num_residencias
    for i in range(num_residencias):
        model += lpSum(x[i, t] * demanda_residencias[i][t] for t in range(num_horas)) - media_alocado <= v[i]
        model += media_alocado - lpSum(x[i, t] * demanda_residencias[i][t] for t in range(num_horas)) <= v[i]
        model += v[i] >= 0.1 * media_alocado

    # Resolver
    model.solve(PULP_CBC_CMD(msg=1, timeLimit=600))

    # Verificar status
    print(f"Status do solver (Cenário 3): {model.status}")
    if model.status != 1:
        print("Erro: Solver não encontrou solução ótima no Cenário 3")
        raise ValueError("Solver falhou para o Cenário 3")

    # Saídas
    aloc = np.zeros((num_residencias, num_horas))
    rede = np.zeros((num_residencias, num_horas))
    excedente = np.zeros(num_horas)

    for i in range(num_residencias):
        for t in range(num_horas):
            aloc[i, t] = max(0, x[i, t].varValue or 0) * demanda_residencias[i][t]
            rede[i, t] = max(0, u[i, t].varValue or 0)
    for t in range(num_horas):
        excedente[t] = max(0, z[t].varValue or 0)

    # Depuração
    print(f"Resumo da alocação (Cenário 3):")
    for i in range(num_residencias):
        energia_alocada = sum(aloc[i, t] for t in range(num_horas))
        print(f"Residência {i+1}: {energia_alocada:.2f} kWh")
    print(f"Energia total alocada: {np.sum(aloc):.2f} kWh")
    print(f"Uso da rede: {np.sum(rede):.2f} kWh")
    print(f"Desvio padrão (equidade): {np.std([sum(aloc[i, t] for t in range(num_horas)) for i in range(num_residencias)]):.2f}")
    print(f"Custo total da rede: {sum(tarifas_residencias[i][t] * rede[i][t] for i in range(num_residencias) for t in range(num_horas)):.2f} R$")

    return aloc, rede, excedente
