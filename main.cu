#include <cstdio>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h"

/**
 * Отладочные сообщения
 */
//#define DEBUG

/*
 * Количество нитей на блок. Для Pascal оптимальное значение 128/256
 */
constexpr const int maxThreadsPerBlock = 256;
/*
 * Количество CUDA ядер на один SM. Для Pascal - 128
 */
constexpr const unsigned int blocksPerSm = 8;

constexpr const size_t MATRIX_DIM = 2000;
constexpr const size_t MATRIX_SIZE = MATRIX_DIM*MATRIX_DIM;
using matrixType = float;

enum PriorityModes {
    MAX_BLOCKS_WITH_OPT_NUM_TH, ALL_BLOCKS_WITH_MAX_THREADS
};

class optimalParams {
public:
    dim3 blocksDim;
    dim3 threadsDim;
    unsigned int sharedMem = 0;

    optimalParams(dim3 b, dim3 t) : blocksDim{b}, threadsDim{t} {}

    optimalParams(dim3 b, dim3 t, unsigned int s) : blocksDim{b}, threadsDim{t}, sharedMem{s} {}
};

__global__ void helloworld(void) {
    printf("Hello, World from GPU! %d, %d\n", blockIdx.x,
           threadIdx.x);
}

__global__ void multiply(matrixType *matrixA, matrixType *matrixB, matrixType *matrixC, int threadNum) {
    /**
     * blockIdx.x * threadNum + threadIdx.x
     */
#ifdef DEBUG
    if (blockIdx.x + threadIdx.x == 0) {
        printf("A0, B0, C0: %f, %f, %f\n", matrixA[0], matrixB[0], matrixC[0]);
    }
    if (threadIdx.x == 0) {
        printf("Block %d\n", blockIdx.x);
    }
#endif
    size_t n = blockIdx.x * threadNum + threadIdx.x;
    if (n < MATRIX_SIZE) {
        size_t x = n/MATRIX_DIM;
        size_t y = n%MATRIX_DIM;
        for(size_t i = 0; i < MATRIX_DIM; ++i) {
            matrixC[n] += (matrixA[x*MATRIX_DIM + i] * matrixB[i*MATRIX_DIM + y]);
        }
#ifdef DEBUG
        printf("x: %llu y: %llu %f \n",x,y, matrixC[n]);
#endif
    }
}

__global__ void multiplyShared(matrixType *matrixA, matrixType *matrixB, matrixType *matrixC, int threadNum) {
    /**
     * blockIdx.x * threadNum + threadIdx.x
     */
#ifdef DEBUG
    if (blockIdx.x + threadIdx.x == 0) {
        printf("A0, B0, C0: %f, %f, %f\n", matrixA[0], matrixB[0], matrixC[0]);
    }
    if (threadIdx.x == 0) {
        printf("Block %d\n", blockIdx.x);
    }
#endif
    size_t n = blockIdx.x * threadNum + threadIdx.x;
    if (n < MATRIX_SIZE) {
        size_t x = n/MATRIX_DIM;
        size_t y = n%MATRIX_DIM;
        for(size_t i = 0; i < MATRIX_DIM; ++i) {
            matrixC[n] += (matrixA[x*MATRIX_DIM + i] * matrixB[i*MATRIX_DIM + y]);
        }
#ifdef DEBUG
        printf("x: %llu y: %llu %f \n",x,y, matrixC[n]);
#endif
    }
}

__global__ void printMatrix(matrixType* A) {
    for(size_t i = 0; i<MATRIX_SIZE; ++i){
        printf("%f ", A[i]);
        if ((i+1)%MATRIX_DIM == 0) printf("\n");
    }
    printf("\n");
}
/*__global__ void add(int a, int b, int *const c) {
    *c = a + b;
}*/

unsigned next_power_of_two(unsigned int x) {
    --x;
    x |= (x >> 1);
    x |= (x >> 2);
    x |= (x >> 4);
    x |= (x >> 8);
    return x + 1;
}

cudaDeviceProp showProperties() {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);//определение параметров GPU с номером 0
#ifdef DEBUG
    printf("Device name : %s\n", deviceProp.name);
    printf("Total global memory : %llu MB\n",
           deviceProp.totalGlobalMem / 1024 / 1024);
    printf("Shared memory per block : %zu\n",
           deviceProp.sharedMemPerBlock);
    printf("Registers per block : %d\n",
           deviceProp.regsPerBlock);
    printf("Warp size : %d\n", deviceProp.warpSize);
    printf("Memory pitch : %zu\n", deviceProp.memPitch);
    printf("Max threads per block : %d\n",
           deviceProp.maxThreadsPerBlock);
    printf("Max threads dimensions : x = %d, y = %d, z = %d\n",
           deviceProp.maxThreadsDim[0],
           deviceProp.maxThreadsDim[1],
           deviceProp.maxThreadsDim[2]);
    printf("Max grid size: x = %d, y = %d, z = %d\n",
           deviceProp.maxGridSize[0], deviceProp.maxGridSize[1],
           deviceProp.maxGridSize[2]);
    printf("Clock rate: %d\n", deviceProp.clockRate);
    printf("Total constant memory: %zu\n",
           deviceProp.totalConstMem);
    printf("Compute capability: %d.%d\n",
           deviceProp.major, deviceProp.minor);
    printf("Texture alignment: %zu\n",
           deviceProp.textureAlignment);
    printf("Device overlap: %d\n",
           deviceProp.deviceOverlap);
    printf("Multiprocessor count: %d\n",
           deviceProp.multiProcessorCount);
    printf("Kernel execution timeout enabled: %s\n",
           deviceProp.kernelExecTimeoutEnabled ? "true" :
           "false");
    printf("Can map host memory: %s\n",
           deviceProp.canMapHostMemory ? "true" :
           "false");
    printf("Device has Compute Capability %d.%d\n",
           deviceProp.major, deviceProp.minor);
#endif
    return deviceProp;
}

optimalParams detectOptimalBlockAndGridSize(cudaDeviceProp *prop, size_t matrixSize, PriorityModes mode) {
    switch (mode) {
        case MAX_BLOCKS_WITH_OPT_NUM_TH: {
            unsigned int blocks = (matrixSize / maxThreadsPerBlock) + 1;
            return optimalParams{blocks, maxThreadsPerBlock};
        }
        default: {
            return optimalParams{blocksPerSm, maxThreadsPerBlock};
        }
    }
}

size_t getMatrixSize() {
    return MATRIX_SIZE * sizeof(matrixType);
}

int main() {
    curandGenerator_t gen;
    matrixType *A, *B, *C;
    auto size = getMatrixSize();

    /**
     * Выделяем память на GPU
     */
    cudaMalloc((void **) &A, size);
    cudaMalloc((void **) &B, size);
    cudaMalloc((void **) &C, size);

    /**
     * Создаём ГПСЧ для заполнения матриц и заполняем их
     */
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MTGP32);
    curandSetPseudoRandomGeneratorSeed(gen, 2281489ULL);
    curandGenerateNormal(gen, A, MATRIX_SIZE, 5, 2);
    curandGenerateNormal(gen, B, MATRIX_SIZE, 9, 3);
    curandGenerateNormal(gen, C, MATRIX_SIZE, 0, 0);

    //printMatrix<<<1,1>>>(A);
    //printMatrix<<<1,1>>>(B);

    auto prop = showProperties();
    auto optimal = detectOptimalBlockAndGridSize(&prop, MATRIX_SIZE, MAX_BLOCKS_WITH_OPT_NUM_TH);
    printf("Optimal: %d, %d\n", optimal.blocksDim.x, optimal.threadsDim.x);

    // инициализируем события
    cudaEvent_t start, stop;
    float elapsedTime;

    // создаем события
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // запись события
    cudaEventRecord(start, 0);

    // вызов ядра
    multiply<<<optimal.blocksDim, optimal.threadsDim>>>(A, B, C, optimal.threadsDim.x);
    cudaEventRecord(stop, 0);

    // ожидание завершения работы ядра
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    // вывод информации
    printf("Time spent executing by the GPU: %.2f millseconds\n", elapsedTime);

    // уничтожение события
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    //printMatrix<<<1,1>>>(C);
    /*int res;
    int* res_ptr;
    cudaMalloc((void**)&res_ptr, sizeof(int));
    add<<<1,1>>>(2, 7, res_ptr);
    cudaMemcpy(&res, res_ptr, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Result: %d\n", res);
    cudaFree((void*)res_ptr);*/
    //helloworld<<<2, 5>>>();
    //printf("Hello, World from CPU!\n");

    /**
     * Очищаем память
     */
    curandDestroyGenerator(gen);
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;
}
