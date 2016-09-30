// kernel function
__global__ void reduction1( int *data, int n){
	int step;
	for(int i=0; i<9; i++){
		step = pow((double)2, (double)9-i-1);
		if(threadIdx.x<step){
			data[threadIdx.x] += data[threadIdx.x+step];
			__syncthreads();
		}
	}
}


__global__ void reduction2(int *data, int n){

}

