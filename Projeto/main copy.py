import os
import pandas as pd
from config import configurar_parametros
from cenario3 import cenario_3_multiobjetivo
from analise import analisar_resultados
from visualizacao import plotar_graficos

def exportar_resultados(analise, aloc, params):
    """Exporta resultados para CSVs."""
    num_residencias = params['num_residencias']
    num_horas = params['num_horas']
    
    # Comparação geral
    df_comparacao = pd.DataFrame([analise])
    df_comparacao.to_csv("saidas/comparacao_cenario3_alagoas.csv", index=False)
    
    # Alocação horária
    df_aloc = pd.DataFrame(
        aloc,
        columns=[f"Hora {t}" for t in range(num_horas)],
        index=[f"Unidade {i+1}{' (TS)' if i % 2 == 0 else ''}" for i in range(num_residencias)]
    )
    df_aloc.to_csv("saidas/alocacao_horaria_cenario3_alagoas.csv")

def main():
    """Executa o Cenário 3 com prioridade para Tarifa Social."""
    os.makedirs("saidas", exist_ok=True)
    params = configurar_parametros()
    
    print("\nExecutando Cenário 3 com prioridade para Tarifa Social...")
    try:
        aloc, rede, excedente = cenario_3_multiobjetivo(params)
        print("\nAnalisando resultados...")
        analise = analisar_resultados(aloc, rede, params, "Cenário 3 [Alagoas, Tarifa Social]")
        print("Exportando resultados...")
        exportar_resultados(analise, aloc, params)
        print("Gerando gráficos...")
        plotar_graficos([analise], [aloc], [rede], params)
        print("Execução concluída! Resultados salvos em 'saidas/'.")
    except Exception as e:
        print(f"Erro ao executar Cenário 3: {e}")

if __name__ == "__main__":
    main()