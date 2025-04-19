import numpy as np

def cenario_1_proporcional(params):
    """
    Cenário 1: Alocação proporcional ao consumo (método conforme diretrizes da ANEEL).

    Neste cenário, a energia solar gerada é alocada proporcionalmente à demanda de cada residência
    em cada hora, considerando perdas na rede elétrica (eficiência) e respeitando o limite da geração solar.

    Qualquer demanda não atendida pela energia solar é complementada com energia da rede.
    O excedente de energia solar (caso haja) é registrado para eventual compensação futura.

    Args:
        params: Dicionário de parâmetros contendo:
            - num_residencias (int): Número de residências participantes.
            - num_horas (int): Número de horas simuladas.
            - demanda_residencias (ndarray NxT): Matriz de demandas das residências.
            - geracao_solar (ndarray T): Vetor de geração solar por hora.
            - eficiencia_rede (ndarray T): Vetor de eficiência da rede por hora.

    Returns:
        aloc (ndarray NxT): Energia solar alocada por residência e hora (kWh).
        rede (ndarray NxT): Energia consumida da rede elétrica por residência e hora (kWh).
        excedente (ndarray T): Energia solar excedente injetada na rede por hora (kWh).
    """

    # Extração dos parâmetros
    num_residencias = params['num_residencias']
    num_horas = params['num_horas']
    demanda_residencias = params['demanda_residencias']
    geracao_solar = params['geracao_solar']
    eficiencia_rede = params['eficiencia_rede']

    # Inicialização das matrizes de resultados
    aloc = np.zeros((num_residencias, num_horas))  # Energia solar alocada
    rede = np.zeros((num_residencias, num_horas))  # Energia consumida da rede
    excedente = np.zeros(num_horas)                # Energia solar excedente

    # Processamento hora a hora
    for t in range(num_horas):
        # Soma da demanda de todas as residências naquela hora
        total_demanda = demanda_residencias[:, t].sum()

        if total_demanda > 0:
            # Calcula a proporção da demanda de cada residência
            proporcao = demanda_residencias[:, t] / total_demanda

            # Energia disponível após considerar a eficiência da rede
            energia_disponivel = geracao_solar[t] * eficiencia_rede[t]

            # Energia efetivamente alocada proporcionalmente, limitada pela geração ou demanda
            energia_a_alocar = min(energia_disponivel, total_demanda)
            aloc[:, t] = proporcao * energia_a_alocar

            # Se houver sobra de energia após atender toda a demanda
            excedente[t] = max(0, energia_disponivel - total_demanda)

        # Energia que precisará ser suprida pela rede elétrica
        rede[:, t] = np.maximum(0, demanda_residencias[:, t] - aloc[:, t])

    return aloc, rede, excedente
