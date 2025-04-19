import numpy as np

def configurar_parametros():
    params = {}
    params['num_residencias'] = 10
    params['num_horas'] = 24
    
    # Demanda: ~4 kWh/dia (TS), ~6 kWh/dia (demais)
    params['demanda_residencias'] = np.zeros((10, 24))
    for i in range(10):
        for t in range(24):
            if i % 2 == 0:  # TS
                if 0 <= t < 12:  # Manhã
                    params['demanda_residencias'][i][t] = 0.13
                elif 12 <= t < 18:  # Tarde
                    params['demanda_residencias'][i][t] = 0.22
                else:  # Noite
                    params['demanda_residencias'][i][t] = 0.30
            else:  # Demais
                if 0 <= t < 12:
                    params['demanda_residencias'][i][t] = 0.20
                elif 12 <= t < 18:
                    params['demanda_residencias'][i][t] = 0.28
                else:
                    params['demanda_residencias'][i][t] = 0.40
    
    # Geração solar: ~57 kWh/dia
    params['geracao_solar'] = np.array([0.0] * 6 + [3.0, 6.0, 9.0, 10.5, 10.5, 9.0, 6.0, 3.0] + [0.0] * 10)
    
    # Eficiência da rede
    params['eficiencia_rede'] = np.array([0.95 if t < 15 or t > 18 else 0.90 for t in range(24)])
    
    params['demanda_base_ajustada'] = 1.0
    params['demandas_minimas_residencias'] = [1.0] * 10
    
    # Limite de injeção
    params['limite_injecao_rede'] = [30.0] * 24
    
    # Tarifas: com impostos e Tarifa Social
    params['tarifas_residencias'] = np.zeros((10, 24))
    for i in range(10):
        for t in range(24):
            base = 0.67 if t < 15 or t > 18 else 1.11
            base *= 1.25  # ICMS/taxas
            if i % 2 == 0:  # TS
                params['tarifas_residencias'][i][t] = base * 0.6  # 40% desconto
            else:
                params['tarifas_residencias'][i][t] = base
    
    params['pesos_objetivos_c3'] = [0.7, 0.05, 0.1, 0.15]
    return params