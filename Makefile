SHELL := /bin/bash
TARGET = solver
NVCC   = nvcc
NVCCFLAGS = -O3 -std=c++17

SRC = src/main.cu src/solver.cu

all:
	source /etc/profile && module load eigen && $(NVCC) $(NVCCFLAGS) $$(pkg-config --cflags eigen3) $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET)
