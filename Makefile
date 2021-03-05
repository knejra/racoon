CFLAGS_COMMON = -std=c++0x -g -O3 -Wall
CFLAGS_NVCC   = .\include\cublas

pangolin : %.o
	g++ ${CFLAGS_COMMON} $^ -o $@

%.o : %.cc
	g++ -c ${CFLAGS_COMMON} $< -o $@

# MatrixG.o : Matrix.cu
#	nvcc -c $< -o $@ -I ${CFLAGS_COMMON}

.PHONY : clean

clean :
	rm -rf Matrix.o