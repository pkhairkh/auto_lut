# Makefile for cosine_percentile module of auto_lut.
#
# Pure C11, no external libraries, AVX-512/AVX2 via -march=native, OpenMP.
#
# Usage:
#   make            -- build test_cosine_percentile
#   make test       -- build then run tests
#   make clean      -- remove build artifacts
#
# All object files and test binaries are gitignored.

CC      ?= gcc
CFLAGS  ?= -O3 -march=native -fopenmp -std=c11 -Wall -Wextra
LDFLAGS ?= -lm -fopenmp

OBJS = cosine.o percentile.o

.PHONY: all test clean

all: test_cosine_percentile

test_cosine_percentile: test_cosine_percentile.c $(OBJS)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

cosine.o: cosine.c cosine.h
	$(CC) $(CFLAGS) -c cosine.c -o cosine.o

percentile.o: percentile.c percentile.h
	$(CC) $(CFLAGS) -c percentile.c -o percentile.o

test: test_cosine_percentile
	./test_cosine_percentile

clean:
	rm -f $(OBJS) test_cosine_percentile
