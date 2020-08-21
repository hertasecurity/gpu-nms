# Compilers
NVCC = nvcc
CC = gcc

# CUDA 
GPU_ARCH = compute_75
SM_ARCH = sm_75
CUDA_HEADERS = /usr/local/cuda/include
CUDA_LIBS = /usr/local/cuda/lib64


all:    nmstest

nmstest: nmstest.o nms.o
	$(CC) -I$(CUDA_HEADERS) -onmstest nmstest.o nms.o -L$(CUDA_LIBS) -lcudart -lcuda

nmstest.o: nmstest.c nms.h
	$(CC) -I$(CUDA_HEADERS) -onmstest.o -c nmstest.c

nms.o:  nms.cu nms.h config.h
	$(NVCC) -o nms.o -c nms.cu -O3 -gencode=arch=$(GPU_ARCH),code=$(SM_ARCH) 

clean:
	rm -f nmstest nmstest.o nms.o
