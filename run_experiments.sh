#!/bin/bash

# ============================================================================
# Script para Automatizar a Execução e Coleta de Tempos do DBSCAN
# ============================================================================

# --- ESTRUTURA DE DIRETÓRIOS ---
BIN_DIR="./bin"
DATA_DIR="./data"
RESULTS_DIR="./results"
CSV_DIR="${RESULTS_DIR}/csv"
PLOTS_DIR="${RESULTS_DIR}/plots"

# Criar diretórios de resultados se não existirem
mkdir -p ${CSV_DIR}
mkdir -p ${PLOTS_DIR}

# --- CONFIGURAÇÕES DO EXPERIMENTO ---
INPUT_FILE="${DATA_DIR}/pontos_entrada.csv"
EXEC_SEQ="${BIN_DIR}/dbscan_seq"
EXEC_PAR="${BIN_DIR}/dbscan_par_corrected" # Usando a versão corrigida

# Arquivo de saída para os tempos
OUTPUT_TIMES="${RESULTS_DIR}/tempos.csv"

# Threads a serem testadas
THREADS=(1 2 4 8)

# Gerar dados de entrada (ex: 10000 pontos, 5 clusters)
echo "Gerando dados de entrada..."
python3 scripts/generate_data.py 10000 5 ${INPUT_FILE}

# Verificar se os executáveis existem
if [ ! -f "$EXEC_SEQ" ] || [ ! -f "$EXEC_PAR" ]; then
    echo "Erro: Executáveis não encontrados. Compile o projeto primeiro com 'make'."
    exit 1
fi

# Criar o cabeçalho do arquivo de tempos
echo "Versao,Threads,Tempo (s)" > ${OUTPUT_TIMES}

# --- Execução Sequencial ---
echo "--------------------------------------------------"
echo "Executando a versão Sequencial..."
OUTPUT_CSV_SEQ="${CSV_DIR}/resultado_seq.csv"
OUTPUT_PLOT_SEQ="${PLOTS_DIR}/saida_seq.png"

SEQ_TIME=$( { time -p ${EXEC_SEQ} ${INPUT_FILE} ${OUTPUT_CSV_SEQ}; } 2>&1 | awk '/real/ {print $2}' )
echo "Sequencial,1,${SEQ_TIME}" >> ${OUTPUT_TIMES}
echo "Tempo Sequencial: ${SEQ_TIME}s"
python3 scripts/plot_clusters.py ${OUTPUT_CSV_SEQ} ${OUTPUT_PLOT_SEQ}


# --- Execução Paralela ---
for T in "${THREADS[@]}"; do
    echo "--------------------------------------------------"
    echo "Executando a versão Paralela com ${T} threads..."
    
    export OMP_NUM_THREADS=${T}
    
    OUTPUT_CSV_PAR="${CSV_DIR}/resultado_par_${T}t.csv"
    OUTPUT_PLOT_PAR="${PLOTS_DIR}/saida_par_${T}t.png"
    
    PAR_TIME=$( { time -p ${EXEC_PAR} ${INPUT_FILE} ${OUTPUT_CSV_PAR}; } 2>&1 | awk '/real/ {print $2}' )
    
    echo "Paralelo,${T},${PAR_TIME}" >> ${OUTPUT_TIMES}
    echo "Tempo Paralelo (${T} threads): ${PAR_TIME}s"
    
    python3 scripts/plot_clusters.py ${OUTPUT_CSV_PAR} ${OUTPUT_PLOT_PAR}
done

echo "--------------------------------------------------"
echo "Experimentos concluídos!"
echo "Resultados de tempo salvos em '${OUTPUT_TIMES}'."
echo "Resultados de clusterização em '${CSV_DIR}'."
echo "Gráficos de clusterização em '${PLOTS_DIR}'."