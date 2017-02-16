#pragma once

#define Block_Size 1024

/*
Input an array of numbers(d_input) and it will be seperated to many blocks and added to a number
stored in another array of numbers(d_output).
*/
__global__
void d_reduceSum(const int const* d_input, int* d_output, const unsigned int sizeOfInput)
{
	extern __shared__ int sdata[];
	const unsigned int soi = sizeOfInput;
	const unsigned int tid = threadIdx.x;
	const unsigned int gridSize = gridDim.x*Block_Size * 2;
	unsigned int index = blockIdx.x*(Block_Size * 2) + tid;

	sdata[tid] = 0;
	while (index < soi)
	{
		sdata[tid] += d_input[index] + d_input[index + Block_Size];
		index += gridSize;
	}

	__syncthreads();

	if (tid < 512) { sdata[tid] += sdata[tid + 512]; } __syncthreads();
	if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads();
	if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads();
	if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads();
	if (tid < 32) { sdata[tid] += sdata[tid + 32]; } __syncthreads();
	if (tid < 16) { sdata[tid] += sdata[tid + 16]; } __syncthreads();
	if (tid < 8) { sdata[tid] += sdata[tid + 8]; } __syncthreads();
	if (tid < 4) { sdata[tid] += sdata[tid + 4]; } __syncthreads();
	if (tid < 2) { sdata[tid] += sdata[tid + 2]; } __syncthreads();
	if (tid < 1) { sdata[tid] += sdata[tid + 1]; } __syncthreads();

	if (tid == 0)d_output[blockIdx.x] = sdata[0];


}

/*
Input an array (g_idata) and it will be seperated to many blocks.
Each blocks will do prescan and be stored in output array (g_odata).
*/
__global__
void prescan(int *g_odata, int *g_idata, int nn)
{
	extern __shared__ int temp[];// allocated on invocation
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	int index = bid*Block_Size + tid;
	int n = Block_Size;
	int offset = 1;
	temp[tid] = index < nn ? g_idata[index] : 0; // load input into shared memory
												 //temp[2 * thid + 1] = 2 * thid + 1 < nn ? g_idata[2 * thid + 1] : 0;
	for (int d = n >> 1; d > 0; d >>= 1) // build sum in place up the tree
	{
		__syncthreads();
		if (tid < d)
		{
			int ai = offset*(2 * tid + 1) - 1;
			int bi = offset*(2 * tid + 2) - 1;
			temp[bi] += temp[ai];
		}
		offset *= 2;
	}
	if (tid == 0) { temp[n - 1] = 0; } // clear the last element
	for (int d = 1; d < n; d *= 2) // traverse down tree & build scan
	{
		offset >>= 1;
		__syncthreads();
		if (tid < d)
		{
			int ai = offset*(2 * tid + 1) - 1;
			int bi = offset*(2 * tid + 2) - 1;
			int t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	__syncthreads();
	g_odata[index] = temp[tid]; // write results to device memory
}

/*
Forming Auxiliary Array
Each number of array need sum of the numbers in each block.
But using our last number of each block calculated by prescan device function
will lose a number from last number of each block in original array numbers.
So we need add it back.
*/
__global__
void formAuxiliary(int *g_odata, int *g_idata, int *g_iidata, int blockNN)
{
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	int ind = bid*Block_Size + tid;
	int index = (ind + 1)*Block_Size - 1;
	if (ind<blockNN)
		g_odata[ind] = g_idata[index] + g_iidata[index];
}

/*
Let Auxiliary Array attach to the output array.
*/
__global__
void addAuxiliary(int *g_odata, int *g_idata)
{
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	int index = bid*Block_Size + tid;
	g_odata[index] += g_idata[bid];
}

/*
There are fixed numbers of blocks.
Each thread in each block have its local histogram.
And every thread helps to seperate each number(d_input) to its local histogram.
At a time a grid of threads parallelly processes a grid of numbers.
Let local histogram be stored in output array(d_output).
Output array will be arranged to a list of local histogram[0] following local histogram[1]
and following local histogram[2]....and so on.
*/
__global__
void d_histogram(const int* d_input, int* d_output, const unsigned int sizeOfInput, const unsigned int histogramSize, const unsigned int gap)
{
	int l_histogram[100];
	for (int i = 0; i < 100; i++)l_histogram[i] = 0;
	int index = blockIdx.x*Block_Size + threadIdx.x;
	int gridSize = gridDim.x*Block_Size;
	for (int i = index; i < sizeOfInput; i += gridSize)
	{
		int t = __min(d_input[i] / gap, histogramSize - 1);
		l_histogram[t]++;
	}

	__syncthreads();

	for (int i = 0; i < histogramSize; i++)
	{
		d_output[i*gridSize + index] = l_histogram[i];
	}
}

/*
radix sort
distinguish the bit is 1 or 0 
*/
__global__
void d_conditionInput(int* d_condition, int* d_input, int radix)
{
	int index = blockIdx.x*Block_Size + threadIdx.x;
	d_condition[index] = (d_input[index] / radix) % 2 == 0 ? 0 : 1;
}

/*
radix sort
Compact input which have the 1 bit or 0
*/
__global__
void d_placeCondition(int* d_conditionIndex, int* d_condition, int* d_temp, int* d_input)
{
	int index = blockIdx.x*Block_Size + threadIdx.x;
	int temp = d_input[index];
	int trueIndex = d_conditionIndex[index];
	if (d_condition[index])d_temp[trueIndex] = temp;
}

/*
make distinguish result reverse
*/
__global__
void d_reverseCondition(int* d_condition)
{
	int index = blockIdx.x*Block_Size + threadIdx.x;
	d_condition[index] = d_condition[index] == 1 ? 0 : 1;
}

/*
connect two arrays
*/
__global__
void d_arrayCat(int* d_input, int* d_temp, int* d_temp2, int* d_conditionIndex, int* d_condition, int sizeOfInput)
{
	int index = blockIdx.x*Block_Size + threadIdx.x;
	if (index > sizeOfInput)return;
	int offset = d_conditionIndex[sizeOfInput - 1] + d_condition[sizeOfInput - 1];
	if (index < offset)d_input[index] = d_temp2[index];
	else d_input[index] = d_temp[index - offset];
}

