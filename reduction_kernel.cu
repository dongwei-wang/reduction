// kernel function
__global__ void reduction1(int *data, int depth, int len){
	int tid = blockIdx.x * blockDim.x + threadIdx.x ;
	int step;
	for(int i=0; i<depth; i++){
		step = pow((double)2, (double)depth-i-1);
		if(tid<step && (tid+step)<len){
			data[tid] += data[tid+step];
		}
		__syncthreads();
	}
}


/* __global__ void reduction1(int *data, int depth, int len){ */
/*     int tid = blockIdx.x * blockDim.x + threadIdx.x ; */
/*     int step=(len+1)/2; */
/*     while(step){ */
/*         if( tid <step && tid+step<2*step){ */
/*             data[tid] += data[tid+step]; */
/*             step = (step+1)/2; */
/*         } */
/*         __syncthreads(); */
/*     } */
/* } */
