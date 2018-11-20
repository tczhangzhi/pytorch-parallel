#include <stdio.h>

__global__ void d_sigmoid(float *data){
  *data = (1.0 - *data) * *data;
}

int main(){
  float *d_data, h_data = 0;
  cudaMalloc((void **)&d_data, sizeof(float));
  cudaMemcpy(d_data, &h_data, sizeof(float), cudaMemcpyHostToDevice);
  d_sigmoid<<<1,1>>>(d_data);
  cudaMemcpy(&h_data, d_data, sizeof(float), cudaMemcpyDeviceToHost);
  printf("data = %d\n", h_data);
  return 0;
}