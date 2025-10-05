// dbscan_par.c — DBSCAN paralelo com OpenMP e Union-Find “lock-based”
// Melhorias:
// - Distância ao quadrado (evita sqrt/pow)
// - Fase 1 (vizinhança + core) paralela
// - Fase 2 (unions) paralela, com ordem de locks consistente
// - Passo de compressão de caminho paralelo
// - Fase 3 (rótulos de core + borda) paralela
// - Checagens de alocação e mensagens de erro

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <omp.h>

/* =====================[ PROGRESS BAR - INÍCIO ]===================== */
#include <time.h>
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
    if (*last_print_ts != 0.0 && t - *last_print_ts < 0.10)
        return; // ~100ms
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
/* =====================[  PROGRESS BAR - FIM  ]===================== */

// ============================================================================
// PARÂMETROS E ESTRUTURAS
// ============================================================================
#define EPSILON 1.0
#define MIN_POINTS 3
#define UNCLASSIFIED 0
#define NOISE -1

// Processamento em blocos para limitar pico de memória
#define CHUNK_SIZE 50000

typedef struct
{
    double x, y;
    int cluster_id;
} Point;

typedef struct
{
    Point *points;
    int num_points;
} Dataset;

// Union-Find com locks por nó (para unions thread-safe)
typedef struct
{
    int *parent;
    omp_lock_t *locks;
    int n;
} UnionFind;

// ============================================================================
// FUNÇÕES AUXILIARES
// ============================================================================

static inline int within_epsilon2(const Point *a, const Point *b)
{
    double dx = a->x - b->x;
    double dy = a->y - b->y;
    return (dx * dx + dy * dy) <= (EPSILON * EPSILON);
}

static void *xmalloc(size_t nbytes, const char *what)
{
    void *p = malloc(nbytes);
    if (!p)
    {
        fprintf(stderr, "Falha ao alocar %s (%zu bytes)\n", what, nbytes);
        exit(EXIT_FAILURE);
    }
    return p;
}

// ============================================================================
// UNION-FIND (com locks)
// ============================================================================

static void uf_init(UnionFind *uf, int n)
{
    uf->n = n;
    uf->parent = (int *)xmalloc((size_t)n * sizeof(int), "UF parent");
    uf->locks = (omp_lock_t *)xmalloc((size_t)n * sizeof(omp_lock_t), "UF locks");
    for (int i = 0; i < n; ++i)
    {
        uf->parent[i] = i;
        omp_init_lock(&uf->locks[i]);
    }
}

static int uf_find(UnionFind *uf, int i)
{
    // find com compressão recursiva
    if (uf->parent[i] == i)
        return i;
    uf->parent[i] = uf_find(uf, uf->parent[i]);
    return uf->parent[i];
}

static void uf_union(UnionFind *uf, int a, int b)
{
    int ra = uf_find(uf, a);
    int rb = uf_find(uf, b);
    if (ra == rb)
        return;

    // ordem de lock consistente para evitar deadlock
    int first = (ra < rb) ? ra : rb;
    int second = (ra < rb) ? rb : ra;

    omp_set_lock(&uf->locks[first]);
    omp_set_lock(&uf->locks[second]);

    // revalida após adquirir locks
    ra = uf_find(uf, ra);
    rb = uf_find(uf, rb);
    if (ra != rb)
    {
        uf->parent[rb] = ra; // attach rb -> ra (heurística simples)
    }

    omp_unset_lock(&uf->locks[second]);
    omp_unset_lock(&uf->locks[first]);
}

static void uf_destroy(UnionFind *uf)
{
    for (int i = 0; i < uf->n; ++i)
    {
        omp_destroy_lock(&uf->locks[i]);
    }
    free(uf->parent);
    free(uf->locks);
}

// Compressão de caminho paralela: acelera finds subsequentes
static void uf_compress_all(UnionFind *uf)
{
#pragma omp parallel for schedule(static)
    for (int i = 0; i < uf->n; ++i)
    {
        uf->parent[i] = uf_find(uf, i);
    }
}

// ============================================================================
// DBSCAN
// ============================================================================

static void dbscan(Dataset *data)
{
    const int n = data->num_points;

    // buffers: contagem de vizinhos e se é core
    int *neighbor_counts = (int *)calloc((size_t)n, sizeof(int));
    bool *is_core = (bool *)calloc((size_t)n, sizeof(bool));
    if (!neighbor_counts || !is_core)
    {
        fprintf(stderr, "Falha ao alocar buffers principais\n");
        exit(EXIT_FAILURE);
    }

    /* -------- PROGRESS: contadores globais e throttle -------- */
    volatile long long prog_done = 0;
    // Aproximação: F1(n) + F2(n) + F3A(n) + F3B(n)  (unions contam por i core)
    long long prog_total = 4LL * (long long)n;
    double last_print = 0.0;

    // ---------------- FASE 1: vizinhança + pontos core (paralela) ----------------
    // Aqui não armazenamos listas de vizinhos; apenas contamos.
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i)
    {
        const Point *Pi = &data->points[i];
        int used = 0;

        // varredura completa (O(N)) para contar vizinhos
        for (int j = 0; j < n; ++j)
        {
            if (i == j)
                continue;
            if (within_epsilon2(Pi, &data->points[j]))
                used++;
        }
        neighbor_counts[i] = used;
        if (used >= MIN_POINTS)
            is_core[i] = true;

        /* PROGRESS: +1 ponto processado na Fase 1 */
#pragma omp atomic update
        prog_done++;
        if (omp_get_thread_num() == 0)
            progress_draw("DBSCAN (par)", prog_done, prog_total, &last_print);
    }

    // ---------------- FASE 2: union de cores conectados (paralela) ---------------
    UnionFind uf;
    uf_init(&uf, n);

    // Processamos i em blocos de CHUNK_SIZE para reduzir pico e manter cache-friendly
    for (int ib = 0; ib < n; ib += CHUNK_SIZE)
    {
        int i_end = ib + CHUNK_SIZE;
        if (i_end > n)
            i_end = n;

#pragma omp parallel for schedule(static)
        for (int i = ib; i < i_end; ++i)
        {
            if (!is_core[i])
            {
                // ainda assim contamos progresso para manter barra suave
#pragma omp atomic update
                prog_done++;
                if (omp_get_thread_num() == 0)
                    progress_draw("DBSCAN (par)", prog_done, prog_total, &last_print);
                continue;
            }

            const Point *Pi = &data->points[i];
            // não guardamos vizinhos; aplicamos union on-the-fly
            for (int j = 0; j < n; ++j)
            {
                if (i == j)
                    continue;
                if (!is_core[j])
                    continue;
                if (within_epsilon2(Pi, &data->points[j]))
                    uf_union(&uf, i, j);
            }

            /* PROGRESS: +1 para cada i (core ou não) avaliado na Fase 2 */
#pragma omp atomic update
            prog_done++;
            if (omp_get_thread_num() == 0)
                progress_draw("DBSCAN (par)", prog_done, prog_total, &last_print);
        }
    }

    // Compressão de caminho paralela após unions
    uf_compress_all(&uf);

    // Mapa raiz->cluster_id
    int *cluster_map = (int *)calloc((size_t)n, sizeof(int));
    if (!cluster_map)
    {
        fprintf(stderr, "Falha ao alocar cluster_map\n");
        exit(EXIT_FAILURE);
    }

    int next_cluster_id = 1;
    for (int i = 0; i < n; ++i)
    {
        if (!is_core[i])
            continue;
        int r = uf.parent[i]; // já comprimido
        if (cluster_map[r] == 0)
            cluster_map[r] = next_cluster_id++;
    }

    // ---------------- FASE 3A: atribuir rótulo aos cores (paralela) --------------
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i)
    {
        if (is_core[i])
        {
            int r = uf.parent[i];
            data->points[i].cluster_id = cluster_map[r];
        }
        else
        {
            data->points[i].cluster_id = NOISE; // começa como ruído
        }

        /* PROGRESS: +1 ponto rotulado na Fase 3A */
#pragma omp atomic update
        prog_done++;
        if (omp_get_thread_num() == 0)
            progress_draw("DBSCAN (par)", prog_done, prog_total, &last_print);
    }

    // ---------------- FASE 3B: atribuir rótulo aos borda (paralela) --------------
    // cada não-core herda de QUALQUER vizinho core (primeiro que achar)
    for (int ib = 0; ib < n; ib += CHUNK_SIZE)
    {
        int i_end = ib + CHUNK_SIZE;
        if (i_end > n)
            i_end = n;

#pragma omp parallel for schedule(static)
        for (int i = ib; i < i_end; ++i)
        {
            if (is_core[i])
            {
                // ainda assim contamos progresso para manter barra suave
#pragma omp atomic update
                prog_done++;
                if (omp_get_thread_num() == 0)
                    progress_draw("DBSCAN (par)", prog_done, prog_total, &last_print);
                continue;
            }

            const Point *Pi = &data->points[i];
            int label = NOISE; // será NOISE se não achar core
            // busca on-the-fly por um vizinho core
            for (int j = 0; j < n; ++j)
            {
                if (!is_core[j])
                    continue;
                if (within_epsilon2(Pi, &data->points[j]))
                {
                    int r = uf.parent[j];
                    label = cluster_map[r];
                    break; // basta um core vizinho
                }
            }
            data->points[i].cluster_id = label;

            /* PROGRESS: +1 ponto rotulado na Fase 3B */
#pragma omp atomic update
            prog_done++;
            if (omp_get_thread_num() == 0)
                progress_draw("DBSCAN (par)", prog_done, prog_total, &last_print);
        }
    }

    /* PROGRESS: finalizar barra */
    progress_done("DBSCAN (par)");

    // limpeza
    free(neighbor_counts);
    free(is_core);
    free(cluster_map);
    uf_destroy(&uf);
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

    FILE *in = fopen(input_filename, "r");
    if (!in)
    {
        perror("Erro ao abrir entrada");
        return 1;
    }

    // contar linhas
    int n = 0;
    char buf[1024];
    while (fgets(buf, sizeof(buf), in))
        n++;

    Dataset data;
    data.num_points = n;
    data.points = (Point *)xmalloc((size_t)n * sizeof(Point), "Dataset points");

    // ler pontos (x,y), entrada sem cabeçalho
    rewind(in);
    for (int i = 0; i < n; ++i)
    {
        if (fscanf(in, "%lf,%lf", &data.points[i].x, &data.points[i].y) != 2)
        {
            fprintf(stderr, "Erro ao ler linha %d do arquivo\n", i + 1);
            free(data.points);
            fclose(in);
            return 1;
        }
        data.points[i].cluster_id = UNCLASSIFIED;
    }
    fclose(in);

    printf("Iniciando DBSCAN Paralelo (OpenMP)\n");
    printf("Parâmetros: Epsilon=%.3f  MinPoints=%d  N=%d  Chunk=%d\n",
           EPSILON, MIN_POINTS, data.num_points, CHUNK_SIZE);

    double t0 = omp_get_wtime();
    dbscan(&data);
    double t1 = omp_get_wtime();
    double dt = t1 - t0;
    if (dt < 0)
        dt = 0;
    printf("Tempo: %.6f s\n", dt);

    FILE *out = fopen(output_filename, "w");
    if (!out)
    {
        perror("Erro ao abrir saída");
        free(data.points);
        return 1;
    }
    fprintf(out, "x,y,cluster_id\n");
    for (int i = 0; i < data.num_points; ++i)
    {
        fprintf(out, "%.6f,%.6f,%d\n", data.points[i].x, data.points[i].y, data.points[i].cluster_id);
    }
    fclose(out);

    free(data.points);
    printf("Resultados em '%s'.\nConcluído.\n", output_filename);
    return 0;
}
