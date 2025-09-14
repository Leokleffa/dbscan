import numpy as np
import pandas as pd
import sys
from sklearn.datasets import make_blobs

def gerar_dados_clusters(n_samples=1000, n_features=2, centers=5, cluster_std=0.6, random_state=42, output_filename="pontos_entrada.csv"):
    """
    Gera um dataset com múltiplos clusters usando make_blobs do scikit-learn.

    Args:
        n_samples (int): O número total de pontos a serem gerados.
        n_features (int): O número de dimensões (coordenadas) para cada ponto.
        centers (int): O número de clusters a serem gerados.
        cluster_std (float): O desvio padrão dos clusters.
        random_state (int): Semente para reprodutibilidade.
        output_filename (str): Nome do arquivo CSV de saída.
    """
    # Gera os dados
    X, y = make_blobs(n_samples=n_samples,
                      n_features=n_features,
                      centers=centers,
                      cluster_std=cluster_std,
                      random_state=random_state)

    # Cria um DataFrame do pandas
    df = pd.DataFrame(X, columns=['x', 'y'])

    # Salva em um arquivo CSV sem cabeçalho e com vírgula como separador
    df.to_csv(output_filename, header=False, index=False, sep=',')

    print(f"Dataset com {n_samples} pontos e {centers} clusters salvo em '{output_filename}'")

if __name__ == "__main__":
    # Exemplo de uso: python generate_data.py 10000 5 data_10k.csv
    if len(sys.argv) < 3:
        print("Uso: python generate_data.py <numero_de_pontos> <numero_de_clusters> [arquivo_saida.csv]")
        sys.exit(1)

    try:
        num_pontos = int(sys.argv[1])
        num_clusters = int(sys.argv[2])
    except ValueError:
        print("Erro: O número de pontos e de clusters devem ser inteiros.")
        sys.exit(1)

    arquivo_saida = "pontos_entrada.csv"
    if len(sys.argv) > 3:
        arquivo_saida = sys.argv[3]

    gerar_dados_clusters(n_samples=num_pontos, centers=num_clusters, output_filename=arquivo_saida)