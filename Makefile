# Nome do compilador
CC=gcc

# Flags de compilação
CFLAGS=-O3 -Wall
CFLAGS_OMP=-O3 -Wall -fopenmp
LIBS=-lm

# --- ESTRUTURA DE DIRETÓRIOS ---
SRC_DIR=src
BIN_DIR=bin
# Cria o diretório bin/ se ele não existir
$(shell mkdir -p $(BIN_DIR))

# --- ARQUIVOS DE ORIGEM ---
# Adiciona o caminho src/ a cada arquivo .c
SRC_SEQ=$(SRC_DIR)/dbscan_seq.c
SRC_PAR=$(SRC_DIR)/dbscan_par.c

# --- EXECUTÁVEIS ---
# Adiciona o caminho bin/ a cada executável
EXEC_SEQ=$(BIN_DIR)/dbscan_seq
EXEC_PAR=$(BIN_DIR)/dbscan_par

# Regra padrão: compila tudo
all: $(EXEC_SEQ) $(EXEC_PAR)

# --- REGRAS DE COMPILAÇÃO ---
$(EXEC_SEQ): $(SRC_SEQ)
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)

$(EXEC_PAR): $(SRC_PAR)
	$(CC) $(CFLAGS_OMP) -o $@ $< $(LIBS)

# Regra para limpar os arquivos gerados
clean:
	rm -rf $(BIN_DIR)

# Informa ao Make que 'all' e 'clean' não são nomes de arquivos
.PHONY: all clean