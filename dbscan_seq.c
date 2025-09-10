#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>

// ============================================================================
//ESTRUTURAS DE DADOS E CONSTANTES
// ============================================================================

// Parâmetros do DBSCAN
#define EPSILON 1.0  // Raio da vizinhança
#define MIN_POINTS 3 // Mínimo de pontos para formar uma região densa

// Identificadores de cluster
#define UNCLASSIFIED 0
#define NOISE -1

// Estrutura para um ponto 2D
typedef struct {
    double x;
    double y;
    int cluster_id; // ID do cluster ao qual o ponto pertence
} Point;

// Estrutura para o dataset
typedef struct {
    Point* points;
    int num_points;
} Dataset;

// ============================================================================
// FUNÇÕES AUXILIARES
// ============================================================================

// Calcula a distância euclidiana entre dois pontos
double euclidean_distance(Point p1, Point p2) {
    return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
}

// Encontra os índices de todos os pontos na vizinhança de 'point_idx' (dentro do raio EPSILON)
// Retorna um array de índices e armazena o tamanho em 'num_neighbors'
int* regionQuery(int point_idx, const Dataset* data, int* num_neighbors) {
    *num_neighbors = 0;
    // Aloca memória inicial para os vizinhos. Pode ser necessário realocar.
    int capacity = 10;
    int* neighbors = (int*)malloc(capacity * sizeof(int));

    for (int i = 0; i < data->num_points; ++i) {
        if (euclidean_distance(data->points[point_idx], data->points[i]) <= EPSILON) {
            // Se o array de vizinhos está cheio, dobra a capacidade
            if (*num_neighbors >= capacity) {
                capacity *= 2;
                neighbors = (int*)realloc(neighbors, capacity * sizeof(int));
            }
            neighbors[*num_neighbors] = i;
            (*num_neighbors)++;
        }
    }
    return neighbors;
}


// ============================================================================
// FUNÇÕES PRINCIPAIS DO ALGORITMO
// ============================================================================

// Expande um cluster a partir de um ponto central
void expandCluster(int point_idx, int* neighbors, int num_neighbors, int cluster_id, Dataset* data) {
    // Atribui o ID do cluster ao ponto central inicial
    data->points[point_idx].cluster_id = cluster_id;

    // Cria uma "fila" (usando o array de vizinhos) para processar
    for (int i = 0; i < num_neighbors; ++i) {
        int current_point_idx = neighbors[i];

        // Se o ponto vizinho não foi classificado ou é ruído, ele agora faz parte deste cluster
        if (data->points[current_point_idx].cluster_id == UNCLASSIFIED || data->points[current_point_idx].cluster_id == NOISE) {
            data->points[current_point_idx].cluster_id = cluster_id;

            // Tenta expandir mais ainda a partir deste novo ponto
            int new_num_neighbors = 0;
            int* new_neighbors = regionQuery(current_point_idx, data, &new_num_neighbors);

            // Se este vizinho também é um ponto central, adiciona seus vizinhos à fila
            if (new_num_neighbors >= MIN_POINTS) {
                // Aumenta o array original de vizinhos para adicionar os novos
                neighbors = (int*)realloc(neighbors, (num_neighbors + new_num_neighbors) * sizeof(int));
                for(int j = 0; j < new_num_neighbors; j++) {
                    // Adiciona apenas se o ponto já não estiver na lista para evitar processamento redundante
                    bool already_in_list = false;
                    for (int k = 0; k < num_neighbors; k++) {
                        if (neighbors[k] == new_neighbors[j]) {
                            already_in_list = true;
                            break;
                        }
                    }
                    if (!already_in_list) {
                         neighbors[num_neighbors] = new_neighbors[j];
                         num_neighbors++;
                    }
                }
            }
            free(new_neighbors); // Libera a memória do query da vizinhança atual
        }
    }
}


// Função principal do DBSCAN
void dbscan(Dataset* data) {
    int cluster_id = 1; // Começamos com o cluster 1

    // Itera sobre todos os pontos do dataset
    for (int i = 0; i < data->num_points; ++i) {
        // Pula pontos que já foram classificados
        if (data->points[i].cluster_id != UNCLASSIFIED) {
            continue;
        }

        // Encontra os vizinhos do ponto atual
        int num_neighbors = 0;
        int* neighbors = regionQuery(i, data, &num_neighbors);

        // Se não há vizinhos suficientes, marca como ruído (pode ser alterado para ponto de borda depois)
        if (num_neighbors < MIN_POINTS) {
            data->points[i].cluster_id = NOISE;
            free(neighbors);
            continue;
        }

        // Expansão do cluster.
        expandCluster(i, neighbors, num_neighbors, cluster_id, data);
        free(neighbors); // Libera a memória após a expansão

        // Prepara para o próximo cluster
        cluster_id++;
    }
}


// ============================================================================
// MAIN
// ============================================================================
int main(int argc, char *argv[]) {
    // ----------------------------------------------------
    // 1. VALIDAR ARGUMENTOS DE LINHA DE COMANDO
    // ----------------------------------------------------
    if (argc != 3) {
        fprintf(stderr, "Uso: %s <arquivo_entrada.csv> <arquivo_saida.csv>\n", argv[0]);
        return 1;
    }
    const char* input_filename = argv[1];
    const char* output_filename = argv[2];

    // ----------------------------------------------------
    // 2. LER ARQUIVO DE ENTRADA
    // ----------------------------------------------------
    FILE* infile = fopen(input_filename, "r");
    if (!infile) {
        perror("Erro ao abrir o arquivo de entrada");
        return 1;
    }

    // Primeira passada: contar o número de pontos
    int num_points = 0;
    char buffer[1024];
    while (fgets(buffer, sizeof(buffer), infile)) {
        num_points++;
    }

    // Alocar memória para o dataset
    Dataset data;
    data.num_points = num_points;
    data.points = (Point*)malloc(num_points * sizeof(Point));
    if (!data.points) {
        perror("Falha ao alocar memória para o dataset");
        fclose(infile);
        return 1;
    }

    // Segunda passada: ler os dados dos pontos
    rewind(infile); // Volta para o início do arquivo
    for (int i = 0; i < num_points; ++i) {
        if (fscanf(infile, "%lf,%lf", &data.points[i].x, &data.points[i].y) != 2) {
            fprintf(stderr, "Erro ao ler a linha %d do arquivo de entrada.\n", i + 1);
            free(data.points);
            fclose(infile);
            return 1;
        }
        data.points[i].cluster_id = UNCLASSIFIED;
    }
    fclose(infile);

    printf("Iniciando DBSCAN Sequencial em C...\n");
    printf("Parâmetros: Epsilon = %.2f, MinPoints = %d\n", EPSILON, MIN_POINTS);
    printf("Total de pontos lidos do arquivo '%s': %d\n\n", input_filename, data.num_points);

    // ----------------------------------------------------
    // 3. EXECUTAR O ALGORITMO DBSCAN
    // ----------------------------------------------------
    dbscan(&data);

    // ----------------------------------------------------
    // 4. ESCREVER RESULTADOS NO ARQUIVO DE SAÍDA
    // ----------------------------------------------------
    FILE* outfile = fopen(output_filename, "w");
    if (!outfile) {
        perror("Erro ao abrir o arquivo de saída");
        free(data.points);
        return 1;
    }

    // Escrever cabeçalho
    fprintf(outfile, "x,y,cluster_id\n");

    // Escrever os dados de cada ponto
    for (int i = 0; i < data.num_points; ++i) {
        fprintf(outfile, "%.6f,%.6f,%d\n", data.points[i].x, data.points[i].y, data.points[i].cluster_id);
    }
    fclose(outfile);
    
    printf("Resultados do clustering foram salvos em '%s'.\n", output_filename);
    
    // ----------------------------------------------------
    // 5. LIBERAR MEMÓRIA
    // ----------------------------------------------------
    free(data.points);
    printf("\nConcluído.\n");

    return 0;
}