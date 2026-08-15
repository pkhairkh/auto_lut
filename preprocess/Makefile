# Makefile for the auto_lut preprocess module.
#
# Builds the preprocess library (preprocess.o + png.o) and the DoD test
# binary. Uses the same flags as the rest of the project:
#   -O3 -march=native -fopenmp -std=c11
# The JSON parser is compiled from ../safetensors/json.c so the module
# has no external source dependencies beyond the system zlib.

CC      ?= gcc
CFLAGS  ?= -O3 -march=native -fopenmp -std=c11 -Wall -Wextra
INCLUDE := -I. -I../safetensors
LDLIBS  := -lz -lm

PRE_OBJS := preprocess.o png.o ../safetensors/json.o

all: test_preprocess

../safetensors/json.o: ../safetensors/json.c ../safetensors/json.h
        $(CC) $(CFLAGS) $(INCLUDE) -c -o $@ ../safetensors/json.c

preprocess.o: preprocess.c preprocess.h png.h fp16.h
        $(CC) $(CFLAGS) $(INCLUDE) -c -o $@ preprocess.c

png.o: png.c png.h
        $(CC) $(CFLAGS) $(INCLUDE) -c -o $@ png.c

test_preprocess: test_preprocess.c $(PRE_OBJS)
        $(CC) $(CFLAGS) $(INCLUDE) -o $@ test_preprocess.c $(PRE_OBJS) $(LDLIBS)

run: test_preprocess
        ./test_preprocess

clean:
        rm -f *.o test_preprocess ../safetensors/json.o

.PHONY: all run clean
