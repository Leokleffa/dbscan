// dbscan_seq.c (versão paralelizada com OpenMP na busca de vizinhos)
// ------------------------------------------------------------------
// Paraleliza regionQuery (varredura O(N^2)) usando buffers por thread,
// evitando contenção. Mantém a expansão de cluster sequencial para
// preservar a lógica original, sem data races em cluster_id.

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>
#include <stdarg.h>
#ifdef _OPENMP
#include <omp.h>
#endif

// ============================================================================
// PARÂMETROS E ESTRUTURAS
// ============================================================================
#define EPSILON 1.0  // Raio da vizinhança
#define MIN_POINTS 3 // Mínimo de pontos para "core"

#define UNCLASSIFIED 0
#define NOISE -1

typedef struct
{
    double x;
    double y;
    int cluster_id;
} Point;

typedef struct
{
    Point *points;
    int num_points;
} Dataset;

// ============================================================================
// PROGRESSO (globais para o sequencial)
// ============================================================================
static long long g_labeled = 0;      // pontos já rotulados (inclui NOISE)
static long long g_total_points = 0; // total de pontos do dataset
static double g_last_print_ts = 0.0; // throttle ~100ms

// ============================================================================
// AUXILIARES
// ============================================================================

// função utilitária para pegar tempo em segundos
static inline double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

static inline void progress_draw(const char *label, long long done, long long total, double *last_print_ts)
{
    if (total <= 0)
        return;
    double t = now_s();
    if (*last_print_ts != 0 && t - *last_print_ts < 0.10)
        return; // ~100 ms
    *last_print_ts = t;

    if (done > total)
        done = total;
    double frac = (double)done / (double)total;
    int pct = (int)(frac * 100.0 + 0.5);

    const int W = 40;
    int filled = (int)(frac * W);
    fprintf(stderr, "\r%s [", label);
    for (int i = 0; i < W; ++i)
        fputc(i < filled ? '#' : ' ', stderr);
    fprintf(stderr, "] %3d%%", pct);
    fflush(stderr);
}

static inline void progress_done(const char *label)
{
    const int W = 40;
    fprintf(stderr, "\r%s [", label);
    for (int i = 0; i < W; ++i)
        fputc('#', stderr);
    fprintf(stderr, "] 100%% ✓\n");
    fflush(stderr);
}

// compara distância ao quadrado (evita sqrt/pow)
static inline int within_epsilon2(const Point *p, const Point *q)
{
    double dx = p->x - q->x;
    double dy = p->y - q->y;
    return (dx * dx + dy * dy) <= (EPSILON * EPSILON);
}

// regionQuery paralela: varre todos os pontos e coleta vizinhos em buffers por thread
// Retorna vetor alocado com índices de vizinhos (ou NULL se não houver).
// num_neighbors é definido com a quantidade final (NÃO inclui o próprio ponto).
int *regionQuery(int point_idx, const Dataset *data, int *num_neighbors)
{
    *num_neighbors = 0;
    const int N = data->num_points;
    if (N <= 1)
        return NULL;

    const Point *P = &data->points[point_idx];

    int nt = 1;
#ifdef _OPENMP
    nt = omp_get_max_threads();
#endif

    // buffers por thread
    int **local_idx = (int **)malloc((size_t)nt * sizeof(int *));
    int *local_used = (int *)calloc((size_t)nt, sizeof(int));
    int *local_cap = (int *)malloc((size_t)nt * sizeof(int));
    if (!local_idx || !local_used || !local_cap)
    {
        perror("Falha ao alocar buffers locais em regionQuery");
        free(local_idx);
        free(local_used);
        free(local_cap);
        return NULL;
    }
    for (int t = 0; t < nt; ++t)
    {
        local_cap[t] = 64; // capacidade inicial por thread
        local_idx[t] = (int *)malloc((size_t)local_cap[t] * sizeof(int));
        if (!local_idx[t])
        {
            perror("Falha ao alocar buffer local em regionQuery");
            for (int k = 0; k <= t; ++k)
                free(local_idx[k]);
            free(local_idx);
            free(local_used);
            free(local_cap);
            return NULL;
        }
    }

#ifdef _OPENMP
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
#pragma omp for schedule(static)
        for (int i = 0; i < N; ++i)
        {
            if (i == point_idx)
                continue;
            if (within_epsilon2(P, &data->points[i]))
            {
                int used = local_used[tid];
                if (used >= local_cap[tid])
                {
                    int newcap = local_cap[tid] << 1;
                    if (newcap < 64)
                        newcap = 64;
                    int *tmp = (int *)realloc(local_idx[tid], (size_t)newcap * sizeof(int));
                    if (!tmp)
                        continue; // falha rara; ignora crescimento
                    local_idx[tid] = tmp;
                    local_cap[tid] = newcap;
                }
                local_idx[tid][used] = i;
                local_used[tid] = used + 1;
            }
        }
    } // end parallel
#else
    for (int i = 0; i < N; ++i)
    {
        if (i == point_idx)
            continue;
        if (within_epsilon2(P, &data->points[i]))
        {
            int used = local_used[0];
            if (used >= local_cap[0])
            {
                int newcap = local_cap[0] << 1;
                if (newcap < 64)
                    newcap = 64;
                int *tmp = (int *)realloc(local_idx[0], (size_t)newcap * sizeof(int));
                if (!tmp)
                { /* ignora crescimento */
                }
                else
                {
                    local_idx[0] = tmp;
                    local_cap[0] = newcap;
                }
            }
            if (local_used[0] < local_cap[0])
            {
                local_idx[0][used] = i;
                local_used[0] = used + 1;
            }
        }
    }
#endif

    // Soma total e aloca vetor final
    int total = 0;
    for (int t = 0; t < nt; ++t)
        total += local_used[t];

    if (total == 0)
    {
        for (int t = 0; t < nt; ++t)
            free(local_idx[t]);
        free(local_idx);
        free(local_used);
        free(local_cap);
        return NULL;
    }

    int *neighbors = (int *)malloc((size_t)total * sizeof(int));
    if (!neighbors)
    {
        perror("Falha ao alocar vetor de vizinhos");
        for (int t = 0; t < nt; ++t)
            free(local_idx[t]);
        free(local_idx);
        free(local_used);
        free(local_cap);
        return NULL;
    }

    int offset = 0;
    for (int t = 0; t < nt; ++t)
    {
        if (local_used[t] > 0)
        {
            memcpy(neighbors + offset, local_idx[t], (size_t)local_used[t] * sizeof(int));
            offset += local_used[t];
        }
        free(local_idx[t]);
    }
    free(local_idx);
    free(local_used);
    free(local_cap);

    *num_neighbors = total; // não inclui o próprio ponto
    return neighbors;
}

// ============================================================================
// ALGORITMO DBSCAN (expansão sequencial, como no original)
// ============================================================================

// Acrescenta à fila apenas vizinhos AINDA NÃO ENFILEIRADOS.
// Usa o bitmap in_queue[N] para deduplicar pushes.
static inline void append_unique_neighbors(int **queue_ptr, int *qsize_ptr,
                                           const int *cand, int ncand,
                                           unsigned char *in_queue, int N)
{
    if (!cand || ncand <= 0)
        return;

    // primeiro, conta quantos são realmente novos
    int add = 0;
    for (int k = 0; k < ncand; ++k)
    {
        int idx = cand[k];
        if (idx < 0 || idx >= N)
            continue;
        if (!in_queue[idx])
        {
            in_queue[idx] = 1;
            add++;
        }
    }
    if (add == 0)
        return;

    // cresce a fila e insere na ordem que aparecer
    int old = *qsize_ptr;
    int *tmp = (int *)realloc(*queue_ptr, (size_t)(old + add) * sizeof(int));
    if (!tmp)
    {
        // fallback: tenta malloc+copy (evita perder expansão)
        tmp = (int *)malloc((size_t)(old + add) * sizeof(int));
        if (!tmp)
        {
            fprintf(stderr, "Falta de memória ao expandir fila de vizinhos.\n");
            exit(EXIT_FAILURE);
        }
        memcpy(tmp, *queue_ptr, (size_t)old * sizeof(int));
        free(*queue_ptr);
    }
    *queue_ptr = tmp;

    int w = old;
    for (int k = 0; k < ncand; ++k)
    {
        int idx = cand[k];
        if (idx < 0 || idx >= N)
            continue;
        // Só escreve os que marcamos como 1 acima; e volta a marcar 2 para
        // não recontar se cand vier repetido em chamadas subsequentes
        if (in_queue[idx] == 1)
        {
            (*queue_ptr)[w++] = idx;
            in_queue[idx] = 2; // 2 = já escrito no buffer
        }
    }
    *qsize_ptr = w;
}

void expandCluster(int point_idx, int **neighbors_ptr, int *num_neighbors_ptr,
                   int cluster_id, Dataset *data)
{
    const int N = data->num_points;

    // Bitmap de "já enfileirado" para esta expansão
    unsigned char *in_queue = (unsigned char *)calloc((size_t)N, 1);
    if (!in_queue)
    {
        fprintf(stderr, "Falha ao alocar bitmap in_queue.\n");
        exit(EXIT_FAILURE);
    }

    // Seed: rotula e marca como já processado (não precisa entrar na fila)
    if (data->points[point_idx].cluster_id == UNCLASSIFIED ||
        data->points[point_idx].cluster_id == NOISE)
    {
        data->points[point_idx].cluster_id = cluster_id;
        g_labeled++;
        progress_draw("DBSCAN (seq)", g_labeled, g_total_points, &g_last_print_ts);
    }

    // Marca vizinhos iniciais como enfileirados
    for (int i = 0; i < *num_neighbors_ptr; ++i)
    {
        int idx = (*neighbors_ptr)[i];
        if (idx >= 0 && idx < N)
            in_queue[idx] = 2; // já estão no buffer
    }

    // percorre "fila" de vizinhos que vai crescendo
    for (int i = 0; i < *num_neighbors_ptr; ++i)
    {
        int current_point_idx = (*neighbors_ptr)[i];

        if (data->points[current_point_idx].cluster_id == UNCLASSIFIED ||
            data->points[current_point_idx].cluster_id == NOISE)
        {
            data->points[current_point_idx].cluster_id = cluster_id;
            g_labeled++;
            progress_draw("DBSCAN (seq)", g_labeled, g_total_points, &g_last_print_ts);

            int new_num_neighbors = 0;
            int *new_neighbors = regionQuery(current_point_idx, data, &new_num_neighbors);

            // Se o vizinho é core (contando o próprio), anexa SEUS vizinhos,
            // mas somente aqueles que ainda não estão na fila.
            if (new_neighbors && (new_num_neighbors + 1) >= MIN_POINTS)
            {
                append_unique_neighbors(neighbors_ptr, num_neighbors_ptr,
                                        new_neighbors, new_num_neighbors,
                                        in_queue, N);
            }
            free(new_neighbors);
        }
    }

    free(in_queue);
}

void dbscan(Dataset *data)
{
    int cluster_id = 1;

    // Inicializa progresso global
    g_total_points = data->num_points;
    g_labeled = 0;
    g_last_print_ts = 0.0;

    for (int i = 0; i < data->num_points; ++i)
    {
        if (data->points[i].cluster_id != UNCLASSIFIED)
            continue;

        int num_neighbors = 0;
        int *neighbors = regionQuery(i, data, &num_neighbors);

        // inclui o próprio ponto no critério de "core"
        if (!neighbors || (num_neighbors + 1) < MIN_POINTS)
        {
            // ponto isolado => NOISE
            data->points[i].cluster_id = NOISE;
            g_labeled++;
            progress_draw("DBSCAN (seq)", g_labeled, g_total_points, &g_last_print_ts);
            free(neighbors);
            continue;
        }

        // Expande cluster a partir do seed i (queue pode realocar)
        expandCluster(i, &neighbors, &num_neighbors, cluster_id, data);
        free(neighbors);
        cluster_id++;
    }

    // finaliza barra
    progress_done("DBSCAN (seq)");
}

// ============================================================================
// MAIN
// ============================================================================
int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        fprintf(stderr, "Uso: %s <arquivo_entrada.csv> <arquivo_saida.csv>\n", argv[0]);
        return 1;
    }
    const char *input_filename = argv[1];
    const char *output_filename = argv[2];

    FILE *infile = fopen(input_filename, "r");
    if (!infile)
    {
        perror("Erro ao abrir o arquivo de entrada");
        return 1;
    }

    // conta linhas
    int num_points = 0;
    char buffer[1024];
    while (fgets(buffer, sizeof(buffer), infile))
    {
        num_points++;
    }

    Dataset data;
    data.num_points = num_points;
    data.points = (Point *)malloc((size_t)num_points * sizeof(Point));
    if (!data.points)
    {
        perror("Falha ao alocar memória para o dataset");
        fclose(infile);
        return 1;
    }

    // lê pontos (x,y) — arquivo de entrada sem cabeçalho
    rewind(infile);
    for (int i = 0; i < num_points; ++i)
    {
        if (fscanf(infile, "%lf,%lf", &data.points[i].x, &data.points[i].y) != 2)
        {
            fprintf(stderr, "Erro ao ler a linha %d do arquivo de entrada.\n", i + 1);
            free(data.points);
            fclose(infile);
            return 1;
        }
        data.points[i].cluster_id = UNCLASSIFIED;
    }
    fclose(infile);

    printf("Iniciando DBSCAN (sequencial + regionQuery paralela com OpenMP)...\n");
    printf("Parâmetros: Epsilon = %.2f, MinPoints = %d\n", EPSILON, MIN_POINTS);
    printf("Total de pontos: %d\n\n", data.num_points);

#ifdef _OPENMP
    double t0 = omp_get_wtime();
#endif

    dbscan(&data);

#ifdef _OPENMP
    double t1 = omp_get_wtime();
    double dt = t1 - t0;
    if (dt < 0)
        dt = 0; // robustez
    printf("Tempo total (medido): %.6f s\n", dt);
#endif

    // escreve saída com cabeçalho
    FILE *outfile = fopen(output_filename, "w");
    if (!outfile)
    {
        perror("Erro ao abrir o arquivo de saída");
        free(data.points);
        return 1;
    }
    fprintf(outfile, "x,y,cluster_id\n");
    for (int i = 0; i < data.num_points; ++i)
    {
        fprintf(outfile, "%.6f,%.6f,%d\n",
                data.points[i].x, data.points[i].y, data.points[i].cluster_id);
    }
    fclose(outfile);

    free(data.points);
    printf("Resultados salvos em '%s'.\nConcluído.\n", output_filename);
    return 0;
}
