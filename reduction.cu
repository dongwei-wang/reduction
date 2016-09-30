#include <stdio.h>
#include "reduction_kernel.cu"

#define UPPER_BOUND 1000

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
	int num_elements=512;
	/* if( scanf("%d", &num_elements) != 1){ */
	/*     printf("Input Failed\n"); */
	/*     return; */
	/* } */

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
		data[i] = rand()%UPPER_BOUND;
		//data[i] = i;
	}
	return data;
}

int computeOnDevice(int* h_data, int num_elements){
	int* d_data = NULL;
	cudaMalloc((void**)&d_data, num_elements * sizeof(int));
	cudaMemcpy(d_data, h_data, num_elements * sizeof(int), cudaMemcpyHostToDevice);
	reduction1<<<1, num_elements/2>>>(d_data, num_elements);
	cudaMemcpy(h_data, d_data, num_elements * sizeof(int), cudaMemcpyDeviceToHost);
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

