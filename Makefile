# Makefile for auto_lut - Fully Automated C LUT Palettization Engine
#
# C11 + optional CUDA. No external libs (libc, libm, libz, OpenMP, cuSOLVER/cuBLAS).
# gcc -O3 -march=native -fopenmp -std=c11
#
# Build:   make            -> ./auto_lut
# Clean:   make clean

CC       := gcc
CFLAGS   := -O3 -march=native -fopenmp -std=c11 -Wall -Wextra
LDFLAGS  := -fopenmp
LIBS     := -lm -lz

# Include paths
INCLUDES := -Isafetensors -Ifp16 -Ipng -Itokenizer -Ipreprocess -Ipack -Iforward -Icosine_percentile -Igptq -Ikmeans1d -Iaccf

# Module directories
MOD_DIRS := safetensors fp16 png tokenizer preprocess pack forward cosine_percentile gptq kmeans1d accf

# --- CUDA detection ---
CUDA_AVAILABLE := $(shell which nvcc > /dev/null 2>&1 && echo "yes" || echo "no")

ifeq ($(CUDA_AVAILABLE), yes)
    CUDA_CC  := $(shell nvcc --compile -arch=native -o /dev/null -x cu /dev/null 2>&1 | grep -oP 'sm_\d+' | head -1)
    ifeq ($(CUDA_CC), )
        CUDA_CC := sm_75
    endif
    NVCC     := nvcc
    CUDA_CFLAGS := -O3 -arch=$(CUDA_CC) -std=c++14 -Xcompiler -fopenmp
    CUDA_LDFLAGS := -lcusolver -lcublas -lcudart
    CUDA_DEFS := -DHAVE_CUDA
    CFLAGS += $(CUDA_DEFS)
    CUDA_SRCS := gptq/gptq_gpu.cu accf/accf_gpu.cu
    CUDA_OBJS := $(CUDA_SRCS:.cu=.o)
else
    CUDA_OBJS :=
    CUDA_LDFLAGS :=
endif

# All .c files (excluding test files and duplicate json.c)
ALL_SRCS := $(foreach d,$(MOD_DIRS),$(filter-out $(d)/test_%.c,$(wildcard $(d)/*.c)))
ALL_SRCS := $(filter-out tokenizer/json.c,$(ALL_SRCS))
# Exclude dispatch files from generic rule (they need special handling)
ALL_SRCS := $(filter-out gptq/gptq_dispatch.c accf/accf_dispatch.c,$(ALL_SRCS))
OBJS := $(ALL_SRCS:.c=.o)

# Dispatch objects (need CUDA_DEFS)
GPTQ_DISPATCH := gptq/gptq_dispatch.o
ACCF_DISPATCH := accf/accf_dispatch.o

# Final binary
BIN := auto_lut

.PHONY: all clean

all: $(BIN)

# Link
$(BIN): main.c $(OBJS) $(GPTQ_DISPATCH) $(ACCF_DISPATCH) $(CUDA_OBJS)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ main.c $(OBJS) $(GPTQ_DISPATCH) $(ACCF_DISPATCH) $(CUDA_OBJS) $(LIBS) $(LDFLAGS) $(CUDA_LDFLAGS)

# Generic C compile
%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# Dispatch files need CUDA_DEFS
gptq/gptq_dispatch.o: gptq/gptq_dispatch.c gptq/gptq.h
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

accf/accf_dispatch.o: accf/accf_dispatch.c accf/accf.h
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# CUDA compile (only if nvcc available)
ifeq ($(CUDA_AVAILABLE), yes)
%.o: %.cu
	$(NVCC) $(CUDA_CFLAGS) $(INCLUDES) -c $< -o $@
endif

# tokenizer needs json_compat.h shim
tokenizer/tokenizer.o: tokenizer/tokenizer.c tokenizer/tokenizer.h tokenizer/json_compat.h safetensors/json.h
	$(CC) $(CFLAGS) $(INCLUDES) -include tokenizer/json_compat.h -c tokenizer/tokenizer.c -o $@

# Module dependencies
safetensors/json.o:        safetensors/json.c safetensors/json.h
safetensors/safetensors.o: safetensors/safetensors.c safetensors/safetensors.h safetensors/json.h
fp16/fp16.o:               fp16/fp16.c fp16/fp16.h
png/png.o:                 png/png.c png/png.h
preprocess/preprocess.o:   preprocess/preprocess.c preprocess/preprocess.h png/png.h
pack/pack.o:               pack/pack.c pack/pack.h fp16/fp16.h
forward/forward.o:         forward/forward.c forward/forward.h safetensors/safetensors.h safetensors/json.h fp16/fp16.h
forward/metadata.o:        forward/metadata.c forward/metadata.h safetensors/safetensors.h forward/forward.h fp16/fp16.h
cosine_percentile/cosine.o:    cosine_percentile/cosine.c cosine_percentile/cosine.h
cosine_percentile/percentile.o: cosine_percentile/percentile.c cosine_percentile/percentile.h
gptq/gptq_cpu.o:           gptq/gptq_cpu.c gptq/gptq.h
accf/accf_cpu.o:            accf/accf_cpu.c accf/accf.h

clean:
	rm -f $(OBJS) $(GPTQ_DISPATCH) $(ACCF_DISPATCH) $(CUDA_OBJS) $(BIN)
