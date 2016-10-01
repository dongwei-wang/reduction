// kernel function
__global__ void reduction(int *data, unsigned len){
	unsigned tid = (blockDim.x * blockIdx.x) + threadIdx.x ;
	unsigned step= len;
	while( step != 0 ){
		step = step>>1;
		if( tid<step){
			data[tid] += data[tid+step];
		}
		__syncthreads();
	}
}
