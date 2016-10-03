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
// Probably this approach can not work at GPU, especially when the size of array is too large!!!
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
__global__ void reduction_sm_idle_threads(int* in_array, int* out_array ){

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

__global__ void reduction_sm_no_idle_threads(int* in_array, int* out_array ){

	//unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int tid = threadIdx.x;
	unsigned int start_idx = 2*blockIdx.x *blockDim.x;
	__shared__ int s_mem[BLOCK_SIZE<<1];

	s_mem[tid] = in_array[start_idx+tid];
	s_mem[tid+BLOCK_SIZE] = in_array[start_idx+BLOCK_SIZE+tid];

	__syncthreads();

	for(unsigned step = blockDim.x; step>0; step>>=1){
		if(tid < step )
			s_mem[tid] += s_mem[tid+step];
		__syncthreads();
	}

	if(tid==0)
		out_array[blockIdx.x] = s_mem[0];
}


/* Function to check if x is power of 2*/
int isPowerOfTwo(int n)
{
	if (n == 0)
		return 0;

	while (n != 1)
	{
		if (n%2 != 0)
			return 0;
		n = n/2;
	}
	return 1;
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
	// start to time
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float milliseconds = 0;
	cudaEventRecord(start);

	int total_sum = 0;
	for( int i = 0; i < len; ++i){
		total_sum += data[i];
	}

	// stop to time
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("CPU time to array reduction %f ms\n", milliseconds);

	return total_sum;
}

// compute in GPU with shared memory
// This approach will assign threads which is same to the number of elements
int computeOnDevice_sm_idle_threads(int* h_data, int len){
	// declare two device memory
	int* d_data = NULL;
	int* d_out_data = NULL;

	// calculate the number of blocks
	unsigned int block_cnt = (len + BLOCK_SIZE - 1)/ BLOCK_SIZE;

	// initialize a host device
	int* out_data = arrayZero(block_cnt);

	printf("The array size  %d\n", len);
	printf("The block count  %d\n", block_cnt );

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

	// start to time
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float milliseconds = 0;
	cudaEventRecord(start);

	// launch the kernel
	reduction_sm_idle_threads<<<gridDim, blockDim>>>(d_data, d_out_data);


	// stop to time
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("GPU time to array reduction is %f ms\n", milliseconds);

	cudaMemcpy(out_data, d_out_data, block_cnt*sizeof(int), cudaMemcpyDeviceToHost);

	// sum all the first elements of output data
	int reference = 0;
	for( unsigned i=0; i<block_cnt; i++ ){
		reference += out_data[i];
	}
	return reference;
}

// compute in GPU with shared memory
// this approach will assign thread which is half to the number of elements
int computeOnDevice_sm_no_idle_threads(int* h_data, int len){
	// declare two device memory
	int* d_data = NULL;
	int* d_out_data = NULL;

	// calculate the number of blocks
	unsigned int block_cnt = ((len>>1) + BLOCK_SIZE - 1)/BLOCK_SIZE;

	// initialize a host device
	int* out_data = arrayZero(block_cnt);

	printf("The array size is %d\n", len);
	printf("The block count is %d\n", block_cnt );

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

	// start to time
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float milliseconds = 0;
	cudaEventRecord(start);

	// launch the kernel
	reduction_sm_no_idle_threads<<<gridDim, blockDim>>>(d_data, d_out_data);

	// stop to time
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("GPU time to array reduction %f ms\n", milliseconds);

	cudaMemcpy(out_data, d_out_data, block_cnt*sizeof(int), cudaMemcpyDeviceToHost);

	// sum all the first elements of output data
	int reference = 0;
	for( unsigned i=0; i<block_cnt; i++ ){
		reference += out_data[i];
	}
	return reference;
}

void runTest(){
	unsigned num_elements = 4194304;
	/* if( scanf("%d", &num_elements) != 1){ */
	/*     printf("Input Failed\n"); */
	/*     return; */
	/* } */

	/* if( isPowerOfTwo(num_elements)==0 ) */
	/* { */
	/*     printf("The input should be power of 2!!!\n"); */
	/*     return;  */
	/* } */

	int* h_data = arrayInit(num_elements);

	printf("\n***** CPU processing ..... *****\n");
	int reference = computeGold(h_data, num_elements);

	printf("\n***** GPU kernel 1 processing ..... *****\n");
	printf("///// This approach assign threads which is the same number to array size /////\n");
	int result_sm_idle_threads = computeOnDevice_sm_idle_threads(h_data, num_elements);
	printf( "Shared memory test %s !!!\n", (reference == result_sm_idle_threads) ? "PASSED" : "FAILED");
	printf( "Device: %d  Host: %d\n", result_sm_idle_threads, reference);

	cudaDeviceSynchronize();

	printf("\n***** GPU kernel 2 processing ..... *****\n");
	printf("///// This approach assign threads which is half of array size /////\n");
	int result_sm_no_idle_threads = computeOnDevice_sm_no_idle_threads(h_data, num_elements);
	printf( "Shared memory test %s !!!\n", (reference == result_sm_no_idle_threads) ? "PASSED" : "FAILED");
	printf( "Device: %d  Host: %d\n", result_sm_no_idle_threads, reference);

	free(h_data);
	return;
}

int main(){
	runTest();
	return 0;
}
