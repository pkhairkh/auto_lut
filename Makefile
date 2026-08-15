# Makefile for auto_lut - Fully Automated C LUT Palettization Engine
#
# Pure C11. No external libs (only libc, libm, libz, libpthread via OpenMP).
# gcc -O3 -march=native -fopenmp -std=c11
#
# Build:   make            -> ./auto_lut
# Test:    make test       -> runs test_deps
# Clean:   make clean

CC       := gcc
CFLAGS   := -O3 -march=native -fopenmp -std=c11 -Wall -Wextra
LDFLAGS  := -fopenmp
LIBS     := -lm -lz

# Include paths for all modules
INCLUDES := -Isafetensors -Ifp16 -Ipng -Itokenizer -Ipreprocess -Ipack -Iforward

# Module directories
MOD_DIRS := safetensors fp16 png tokenizer preprocess pack forward

# Object files: one .o per .c in each module dir (excluding test_*.c and
# duplicates like tokenizer/json.c which is a copy of safetensors/json.c)
OBJS := $(filter-out tokenizer/json.o, \
         $(filter-out %/test_%.o, \
          $(foreach d,$(MOD_DIRS),$(patsubst %.c,%.o,$(wildcard $(d)/*.c)))))

# Final binary
BIN := auto_lut

.PHONY: all clean test

all: $(BIN)

# Link the final binary
$(BIN): main.c $(OBJS)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ main.c $(OBJS) $(LIBS) $(LDFLAGS)

# Generic compile rule for each module .c -> .o
%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# Module-specific dependency declarations (so changes trigger rebuilds)
safetensors/json.o:        safetensors/json.c safetensors/json.h
safetensors/safetensors.o: safetensors/safetensors.c safetensors/safetensors.h safetensors/json.h
fp16/fp16.o:               fp16/fp16.c fp16/fp16.h
png/png.o:                 png/png.c png/png.h
tokenizer/tokenizer.o:     tokenizer/tokenizer.c tokenizer/tokenizer.h safetensors/json.h
preprocess/preprocess.o:   preprocess/preprocess.c preprocess/preprocess.h png/png.h
pack/pack.o:               pack/pack.c pack/pack.h fp16/fp16.h
forward/forward.o:         forward/forward.c forward/forward.h safetensors/safetensors.h safetensors/json.h fp16/fp16.h
forward/metadata.o:        forward/metadata.c forward/metadata.h safetensors/safetensors.h forward/forward.h fp16/fp16.h

# Smoke test for dependency modules
test: test_deps
	./test_deps

test_deps: test_deps.c $(OBJS)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ test_deps.c $(OBJS) $(LIBS) $(LDFLAGS)

clean:
	rm -f $(OBJS) $(BIN) test_deps
