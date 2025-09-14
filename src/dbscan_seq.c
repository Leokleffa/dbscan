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
     if (!neighbors) {
        perror("Falha ao alocar memória para vizinhos em regionQuery");
        return NULL;
    }
   
    for (int i = 0; i < data->num_points; ++i) {
        if (point_idx == i) continue;
        if (euclidean_distance(data->points[point_idx], data->points[i]) <= EPSILON) {
            if (*num_neighbors >= capacity) {
                capacity *= 2;
                int* temp = (int*)realloc(neighbors, capacity * sizeof(int));
                if (!temp) {
                    perror("Falha ao realocar memória para vizinhos em regionQuery");
                    free(neighbors);
                    return NULL;
                }
                neighbors = temp;
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
void expandCluster(int point_idx, int** neighbors_ptr, int* num_neighbors_ptr, int cluster_id, Dataset* data) {
    data->points[point_idx].cluster_id = cluster_id;

    // Itera sobre a "fila" de vizinhos
    for (int i = 0; i < *num_neighbors_ptr; ++i) {
        int current_point_idx = (*neighbors_ptr)[i];

        if (data->points[current_point_idx].cluster_id == UNCLASSIFIED || data->points[current_point_idx].cluster_id == NOISE) {
            data->points[current_point_idx].cluster_id = cluster_id;

            int new_num_neighbors = 0;
            int* new_neighbors = regionQuery(current_point_idx, data, &new_num_neighbors);

            if (new_num_neighbors >= MIN_POINTS) {
                // Junta a nova lista de vizinhos à lista original
                int old_num_neighbors = *num_neighbors_ptr;
                *num_neighbors_ptr += new_num_neighbors;
                *neighbors_ptr = (int*)realloc(*neighbors_ptr, (*num_neighbors_ptr) * sizeof(int));
                
                for(int j = 0; j < new_num_neighbors; j++) {
                    (*neighbors_ptr)[old_num_neighbors + j] = new_neighbors[j];
                }
            }
            free(new_neighbors);
        }
    }
}


// Função principal do DBSCAN
void dbscan(Dataset* data) {
    int cluster_id = 1;

    for (int i = 0; i < data->num_points; ++i) {
        if (data->points[i].cluster_id != UNCLASSIFIED) {
            continue;
        }

        int num_neighbors = 0;
        int* neighbors = regionQuery(i, data, &num_neighbors);

        if (num_neighbors < MIN_POINTS) {
            data->points[i].cluster_id = NOISE;
            free(neighbors);
            continue;
        }

        // **MODIFICADO:** Passa os endereços de 'neighbors' e 'num_neighbors'
        expandCluster(i, &neighbors, &num_neighbors, cluster_id, data);
        free(neighbors);

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