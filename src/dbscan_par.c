#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <omp.h>

// ============================================================================
// ESTRUTURAS DE DADOS E CONSTANTES
// ============================================================================

#define EPSILON 1.0
#define MIN_POINTS 3
#define UNCLASSIFIED 0
#define NOISE -1

typedef struct {
    double x, y;
    int cluster_id;
} Point;

typedef struct {
    Point* points;
    int num_points;
} Dataset;

// Estrutura para Union-Find
typedef struct {
    int* parent;
    omp_lock_t* locks; // Locks para garantir a segurança da união
    int num_points;
} UnionFind;

// ============================================================================
// FUNÇÕES DA ESTRUTURA UNION-FIND
// ============================================================================

void uf_init(UnionFind* uf, int n) {
    uf->num_points = n;
    uf->parent = (int*)malloc(n * sizeof(int));
    uf->locks = (omp_lock_t*)malloc(n * sizeof(omp_lock_t));
    for (int i = 0; i < n; i++) {
        uf->parent[i] = i; // Cada ponto começa como seu próprio pai
        omp_init_lock(&uf->locks[i]);
    }
}

// Encontra o representante (raiz) do conjunto de um ponto com compressão de caminho
int uf_find(UnionFind* uf, int i) {
    if (uf->parent[i] == i) {
        return i;
    }
    // Compressão de caminho: aponta diretamente para a raiz
    uf->parent[i] = uf_find(uf, uf->parent[i]);
    return uf->parent[i];
}

// Une dois conjuntos (thread-safe)
void uf_union(UnionFind* uf, int i, int j) {
    int root_i = uf_find(uf, i);
    int root_j = uf_find(uf, j);

    if (root_i != root_j) {
        // Garante uma ordem de bloqueio consistente para evitar deadlocks
        if (root_i < root_j) {
            omp_set_lock(&uf->locks[root_i]);
            omp_set_lock(&uf->locks[root_j]);
        } else {
            omp_set_lock(&uf->locks[root_j]);
            omp_set_lock(&uf->locks[root_i]);
        }
        
        // Verifica novamente após adquirir o lock
        int current_root_i = uf_find(uf, i);
        int current_root_j = uf_find(uf, j);
        if(current_root_i != current_root_j){
            uf->parent[current_root_j] = current_root_i;
        }

        omp_unset_lock(&uf->locks[root_i]);
        omp_unset_lock(&uf->locks[root_j]);
    }
}

void uf_destroy(UnionFind* uf) {
    for (int i = 0; i < uf->num_points; i++) {
        omp_destroy_lock(&uf->locks[i]);
    }
    free(uf->parent);
    free(uf->locks);
}

// ============================================================================
// FUNÇÕES DO ALGORITMO DBSCAN
// ============================================================================

double euclidean_distance(Point p1, Point p2) {
    return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
}

void dbscan(Dataset* data) {
    int n = data->num_points;
    int* neighbor_counts = (int*)calloc(n, sizeof(int));
    int** neighbors = (int**)malloc(n * sizeof(int*));
    bool* is_core = (bool*)calloc(n, sizeof(bool));

    // --- FASE 1: Encontrar vizinhos e pontos centrais (em paralelo) ---
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; i++) {
        int capacity = 10;
        neighbors[i] = (int*)malloc(capacity * sizeof(int));
        for (int j = 0; j < n; j++) {
            if (i == j) continue;
            if (euclidean_distance(data->points[i], data->points[j]) <= EPSILON) {
                if (neighbor_counts[i] >= capacity) {
                    capacity *= 2;
                    neighbors[i] = (int*)realloc(neighbors[i], capacity * sizeof(int));
                }
                neighbors[i][neighbor_counts[i]++] = j;
            }
        }
        if (neighbor_counts[i] >= MIN_POINTS) {
            is_core[i] = true;
        }
    }

    // --- FASE 2: Unir clusters usando Union-Find (em paralelo) ---
    UnionFind uf;
    uf_init(&uf, n);

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; i++) {
        if (!is_core[i]) continue;
        for (int j = 0; j < neighbor_counts[i]; j++) {
            int neighbor_idx = neighbors[i][j];
            if (is_core[neighbor_idx]) {
                uf_union(&uf, i, neighbor_idx);
            }
        }
    }

    // --- FASE 3: Atribuir IDs de cluster (em paralelo) ---
    int* cluster_map = (int*)calloc(n, sizeof(int));
    int cluster_id_counter = 1;

    // Mapeia a raiz de cada conjunto para um ID de cluster final
    for (int i = 0; i < n; i++) {
        if (!is_core[i]) continue;
        int root = uf_find(&uf, i);
        if (cluster_map[root] == 0) {
            cluster_map[root] = cluster_id_counter++;
        }
    }

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; i++) {
        if (is_core[i]) {
            int root = uf_find(&uf, i);
            data->points[i].cluster_id = cluster_map[root];
        } else {
            data->points[i].cluster_id = NOISE; // Começa como ruído
        }
    }
    
    // Atribui pontos de borda ao cluster de um de seus vizinhos centrais
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; i++) {
        if (!is_core[i]) {
            for (int j = 0; j < neighbor_counts[i]; j++) {
                int neighbor_idx = neighbors[i][j];
                if (is_core[neighbor_idx]) {
                    int root = uf_find(&uf, neighbor_idx);
                    data->points[i].cluster_id = cluster_map[root];
                    break;
                }
            }
        }
    }

    // Liberar memória
    free(cluster_map);
    uf_destroy(&uf);
    for (int i = 0; i < n; i++) free(neighbors[i]);
    free(neighbors);
    free(neighbor_counts);
    free(is_core);
}


// ============================================================================
// MAIN
// ============================================================================
int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Uso: %s <arquivo_entrada.csv> <arquivo_saida.csv>\n", argv[0]);
        return 1;
    }
    const char* input_filename = argv[1];
    const char* output_filename = argv[2];

    FILE* infile = fopen(input_filename, "r");
    if (!infile) {
        perror("Erro ao abrir o arquivo de entrada");
        return 1;
    }

    int num_points = 0;
    char buffer[1024];
    while (fgets(buffer, sizeof(buffer), infile)) {
        num_points++;
    }

    Dataset data;
    data.num_points = num_points;
    data.points = (Point*)malloc(num_points * sizeof(Point));
    if (!data.points) {
        perror("Falha ao alocar memória para o dataset");
        fclose(infile);
        return 1;
    }

    rewind(infile);
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

    printf("Iniciando DBSCAN Paralelo (Corrigido) em C com OpenMP...\n");
    printf("Parâmetros: Epsilon = %.2f, MinPoints = %d\n", EPSILON, MIN_POINTS);
    printf("Total de pontos lidos do arquivo '%s': %d\n\n", input_filename, data.num_points);
    
    double start_time = omp_get_wtime();
    dbscan(&data);
    double end_time = omp_get_wtime();
    printf("Tempo de execução do DBSCAN: %f segundos\n", end_time - start_time);

    FILE* outfile = fopen(output_filename, "w");
    if (!outfile) {
        perror("Erro ao abrir o arquivo de saída");
        free(data.points);
        return 1;
    }

    fprintf(outfile, "x,y,cluster_id\n");
    for (int i = 0; i < data.num_points; ++i) {
        fprintf(outfile, "%.6f,%.6f,%d\n", data.points[i].x, data.points[i].y, data.points[i].cluster_id);
    }
    fclose(outfile);
    
    printf("Resultados do clustering foram salvos em '%s'.\n", output_filename);
    
    free(data.points);
    printf("\nConcluído.\n");

    return 0;
}