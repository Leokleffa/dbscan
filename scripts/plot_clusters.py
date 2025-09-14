import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np

def plotar_clusters(caminho_arquivo_csv, caminho_arquivo_saida=None):
    """
    Lê um arquivo CSV com coordenadas de pontos e seus clusters e gera um gráfico de dispersão.

    Args:
        caminho_arquivo_csv (str): O caminho para o arquivo CSV de entrada.
        caminho_arquivo_saida (str, optional): O caminho para salvar a imagem do gráfico.
    """
    try:
        dados = pd.read_csv(caminho_arquivo_csv)
    except FileNotFoundError:
        print(f"Erro: O arquivo '{caminho_arquivo_csv}' não foi encontrado.")
        return
    except Exception as e:
        print(f"Ocorreu um erro ao ler o arquivo: {e}")
        return

    # Mude para um backend não interativo explicitamente para evitar problemas
    plt.switch_backend('Agg')

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    clusters_unicos = sorted(dados['cluster_id'].unique())
    cores = plt.get_cmap('tab10')

    for i, cluster_id in enumerate(clusters_unicos):
        pontos_do_cluster = dados[dados['cluster_id'] == cluster_id]

        if cluster_id == -1:
            # Plotando pontos de ruído (sem edgecolor)
            ax.scatter(pontos_do_cluster['x'],
                       pontos_do_cluster['y'],
                       color='gray',
                       label='Ruído (Noise)',
                       marker='x',
                       s=30, # Tamanho do marcador
                       alpha=0.8)
        else:
            # Plotando clusters reais (com edgecolor)
            ax.scatter(pontos_do_cluster['x'],
                       pontos_do_cluster['y'],
                       color=cores(cluster_id - 1),
                       label=f'Cluster {cluster_id}',
                       marker='o',
                       s=50, # Tamanho do marcador
                       alpha=0.8,
                       edgecolors='k', # A borda preta só é aplicada aqui
                       linewidths=0.5)

    ax.set_title('Visualização dos Clusters - DBSCAN', fontsize=16)
    ax.set_xlabel('Coordenada X', fontsize=12)
    ax.set_ylabel('Coordenada Y', fontsize=12)
    ax.legend(title='Clusters', loc='best', frameon=True, shadow=True)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    if caminho_arquivo_saida:
        try:
            plt.savefig(caminho_arquivo_saida, dpi=300)
            print(f"Gráfico salvo com sucesso em '{caminho_arquivo_saida}'")
        except Exception as e:
            print(f"Erro ao salvar o gráfico: {e}")
    else:
        # Se nenhum arquivo de saída for especificado, avisar o usuário.
        print("Aviso: Nenhum arquivo de saída especificado. O gráfico não foi salvo.")
        print("Para salvar, forneça um segundo argumento: python plot_clusters.py entrada.csv saida.png")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python plot_clusters.py <arquivo_entrada.csv> [arquivo_saida.png]")
        sys.exit(1)

    arquivo_entrada = sys.argv[1]
    arquivo_saida = None
    if len(sys.argv) > 2:
        arquivo_saida = sys.argv[2]
    
    plotar_clusters(arquivo_entrada, arquivo_saida)