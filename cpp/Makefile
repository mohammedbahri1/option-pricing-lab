CXX      ?= g++
CXXFLAGS ?= -O3 -march=native -std=c++17 -Wall -Wextra

all: mc_pricer mc_pricer_omp

# Version séquentielle
mc_pricer: mc_pricer.cpp
	$(CXX) $(CXXFLAGS) -Wno-unknown-pragmas -o $@ $<

# Version parallèle (OpenMP)
mc_pricer_omp: mc_pricer.cpp
	$(CXX) $(CXXFLAGS) -fopenmp -o $@ $<

clean:
	rm -f mc_pricer mc_pricer_omp

.PHONY: all clean
