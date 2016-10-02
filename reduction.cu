/*
Author: Dongwei Wang
Email: wdw828@gmail.com
version: 1.0
*/

/*
   This is a CUDA program to implement the array reduction.
   we sum all the elements in the first element of the array.
   we implement two kernel functions, one just access glboal memory
   another utilize the shared memory to accelerate the summation
*/

/*
   The number of elements in this program should be the power of 2
   such as 1024, 2048, 4096, and 8192
   For those input which is not the power of 2, we did not add the boundary check
   Probably, it can not run
*/

#include <stdio.h>

#define UPPER_BOUND 1000
#define BLOCK_SIZE 256

// This is the kernel function employs global memory
__global__ void reduction_global_memory(int *data, unsigned len){
	unsigned tid = (blockDim.x * blockIdx.x) + threadIdx.x ;
	for( unsigned int step = len/2; step > 0; step>>=1 ){
		if(tid<step){
			data[tid] += data[tid+step];
		}
		__syncthreads();
	}
}

// this is the kernel function employs shared memory
__global__ void reduction_shared_memory(int* in_array, int* out_array ){

	unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int tid = threadIdx.x;
	__shared__ int s_mem[BLOCK_SIZE];

	s_mem[tid] = in_array[index];

	__syncthreads();

	for(unsigned step = blockDim.x/2; step>0; step>>=1){
		if(tid < step )
			s_mem[tid] += s_mem[tid+step];
		__syncthreads();
	}

	if(tid==0)
		out_array[blockIdx.x] = s_mem[0];
}

// allocate the array
int* arrayInit(unsigned len){
	int* data = (int*) malloc(sizeof(int)*len);
	for( unsigned i=0; i<len; i++ ){
		//data[i] = rand()%UPPER_BOUND;
		data[i] = 1;
	}
	return data;
}

// assign 0 to an array
int* arrayZero(unsigned int len){
	int *data = (int*)malloc(sizeof(int)*len);
	for(unsigned i=0; i<len; i++){
		data[i] = 0;
	}
	return data;
}

// compute in CPU
int computeGold( int* data, int len){
	int total_sum = 0;
	for( int i = 0; i < len; ++i){
		total_sum += data[i];
	}
	return total_sum;
}

// compute in GPU with global memory
int computeOnDevice_global_memory(int* h_data, int len){
	int* d_data = NULL;
	unsigned block_cnt = ((len>>1) + BLOCK_SIZE - 1)/BLOCK_SIZE;

	printf("The length is %d\n", len);
	printf("The block  is %d\n", block_cnt );

	dim3 gridDim(block_cnt, 1);
	dim3 blockDim(BLOCK_SIZE, 1);

	cudaMalloc((void**)&d_data, len * sizeof(int));
	cudaMemcpy(d_data, h_data, len * sizeof(int), cudaMemcpyHostToDevice);
	reduction_global_memory<<<gridDim, blockDim>>>(d_data, len);
	cudaMemcpy(h_data, d_data, len * sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(d_data);
	return h_data[0];
}

// compute in GPU with shared memory
int computeOnDevice_shared_memory(int* h_data, int len){
	// declare two device memory
	int* d_data = NULL;
	int* d_out_data = NULL;

	// calculate the number of blocks
	unsigned int block_cnt = (len + BLOCK_SIZE - 1)/ BLOCK_SIZE;

	// initialize a host device
	int* out_data = arrayZero(block_cnt);

	printf("The length is %d\n", len);
	printf("The block  is %d\n", block_cnt );

	// initizlize the kernel dimension
	dim3 gridDim(block_cnt, 1);
	dim3 blockDim(BLOCK_SIZE, 1);

	// allocate the memory in device
	cudaMalloc((void**)&d_data, len*sizeof(int));

	// allocate the memory of output data in device
	cudaMalloc((void**)&d_out_data, block_cnt*sizeof(int));
	cudaMemcpy(d_data, h_data, len*sizeof(int), cudaMemcpyHostToDevice);

	// assign a initial value for output data
	cudaMemset(d_out_data,0, block_cnt*sizeof(int));

	// launch the kernel
	reduction_shared_memory<<<gridDim, blockDim>>>(d_data, d_out_data);
	cudaMemcpy(out_data, d_out_data, block_cnt*sizeof(int), cudaMemcpyDeviceToHost);

	// sum all the first elements of output data
	int reference = 0;
	for( unsigned i=0; i<block_cnt; i++ ){
		reference += out_data[i];
	}

	return reference;
}

void runTest(){
	printf("Please input the number of elements in the array: \n");
	unsigned num_elements;
	if( scanf("%d", &num_elements) != 1){
		printf("Input Failed\n");
		return;
	}

	int* h_data = arrayInit(num_elements);
	int reference = computeGold(h_data, num_elements);
	/* int result_global_memory = computeOnDevice_global_memory(h_data, num_elements); */
	/* printf( "Global memory test %s !!!\n", (reference == result_global_memory) ? "PASSED" : "FAILED"); */
	/* printf( "Device: %d  Host: %d\n", result_global_memory, reference); */

	int result_shared_memory = computeOnDevice_shared_memory(h_data, num_elements);
	printf( "Shared memory test %s !!!\n", (reference == result_shared_memory) ? "PASSED" : "FAILED");
	printf( "Device: %d  Host: %d\n", result_shared_memory, reference);
	free(h_data);
	return;
}

int main(){
	runTest();
	return 0;
}
