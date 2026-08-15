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
INCLUDES := -Isafetensors -Ifp16 -Ipng -Itokenizer -Ipreprocess -Ipack -Iforward -Illoyd_max -Icosine_percentile

# Module directories
MOD_DIRS := safetensors fp16 png tokenizer preprocess pack forward lloyd_max cosine_percentile

# All .c files across modules (excluding test_*.c and the duplicate tokenizer/json.c)
# GNU make filter-out with % requires the pattern to match the full word; the
# safest way to exclude test files is to NOT glob them in the first place.
ALL_SRCS := $(foreach d,$(MOD_DIRS),$(filter-out $(d)/test_%.c,$(wildcard $(d)/*.c)))
ALL_SRCS := $(filter-out tokenizer/json.c,$(ALL_SRCS))
OBJS := $(ALL_SRCS:.c=.o)

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

# tokenizer.c needs the json_compat.h shim (force-include it)
tokenizer/tokenizer.o: tokenizer/tokenizer.c tokenizer/tokenizer.h tokenizer/json_compat.h safetensors/json.h
	$(CC) $(CFLAGS) $(INCLUDES) -include tokenizer/json_compat.h -c tokenizer/tokenizer.c -o $@

# Module-specific dependency declarations
safetensors/json.o:        safetensors/json.c safetensors/json.h
safetensors/safetensors.o: safetensors/safetensors.c safetensors/safetensors.h safetensors/json.h
fp16/fp16.o:               fp16/fp16.c fp16/fp16.h
png/png.o:                 png/png.c png/png.h
preprocess/preprocess.o:   preprocess/preprocess.c preprocess/preprocess.h png/png.h
pack/pack.o:               pack/pack.c pack/pack.h fp16/fp16.h
forward/forward.o:         forward/forward.c forward/forward.h safetensors/safetensors.h safetensors/json.h fp16/fp16.h
forward/metadata.o:        forward/metadata.c forward/metadata.h safetensors/safetensors.h forward/forward.h fp16/fp16.h
lloyd_max/lloyd_max.o:     lloyd_max/lloyd_max.c lloyd_max/lloyd_max.h
cosine_percentile/cosine.o:    cosine_percentile/cosine.c cosine_percentile/cosine.h
cosine_percentile/percentile.o: cosine_percentile/percentile.c cosine_percentile/percentile.h

# Smoke test for dependency modules
test: test_deps
	./test_deps

test_deps: test_deps.c $(OBJS)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ test_deps.c $(OBJS) $(LIBS) $(LDFLAGS)

clean:
	rm -f $(OBJS) $(BIN) test_deps
