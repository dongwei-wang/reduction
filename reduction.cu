#include <stdio.h>
#include "reduction_kernel.cu"

#define UPPER_BOUND 1000
#define ELEMENTS_PER_BLOCK 2048
#define BLOCK_SIZE 1024

void runTest();
int* arrayInit(int len);
int computeOnDevice(int* h_data, int array_mem_size);
int computeGold( int* data, const int len);

int main(){
	runTest();
	return 0;
}

void runTest(){
	printf("Please input the number of elements in the array: \n");
	int num_elements;
	if( scanf("%d", &num_elements) != 1){
		printf("Input Failed\n");
		return;
	}

	int* h_data = arrayInit(num_elements);


	int reference = computeGold(h_data, num_elements);
	int result = computeOnDevice(h_data, num_elements);
	printf( "Test %s !!!\n", (reference == result) ? "PASSED" : "FAILED");
	printf( "Device: %d  Host: %d\n", result, reference);
	free(h_data);
	return;
}

int* arrayInit(int len){
	int* data = (int*) malloc( sizeof( int ) * len );
	for( int i=0; i<len; i++ ){
		//data[i] = rand()%UPPER_BOUND;
		data[i] = 1;
	}
	return data;
}

int computeOnDevice(int* h_data, int len){
	int* d_data = NULL;
	int depth = ceil(log2((double)len));
	int block_cnt = (pow(2, depth-1)+BLOCK_SIZE-1)/BLOCK_SIZE;

	//int block_cnt = (len+1)/(2*BLOCK_SIZE);
	printf("The length is %d\n", len);
	printf("The depth  is %d\n", depth);
	printf("The block  is %d\n", block_cnt );

	cudaMalloc((void**)&d_data, len * sizeof(int));
	cudaMemcpy(d_data, h_data, len * sizeof(int), cudaMemcpyHostToDevice);
	reduction1<<<block_cnt, BLOCK_SIZE>>>(d_data, depth, len);
	cudaMemcpy(h_data, d_data, len * sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(d_data);
	return h_data[0];
}

int computeGold( int* data, const int len){
	int total_sum = 0;
	for( int i = 0; i < len; ++i){
		total_sum += data[i];
	}
	return total_sum;
}

