#include <cstdlib>
#include <iostream>
#include <ctime>
#include <cmath>
#include <iomanip>
#include <stdio.h>
#include "parallel.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


using namespace std;
float cpu_time = 0;
float gpu_time = 0;
int times = 40;
int size1 = 40000000;
int* randNumbers;

int reduceSum(int* input, int sizeOfInput)
{
	int h_output;
	int* d_input;
	int* d_output;
	unsigned int blockNum = 1024;

	cudaMalloc(&d_input, sizeOfInput * sizeof(int));
	cudaMalloc(&d_output,sizeof(int));
	cudaMemcpy(d_input, input, sizeOfInput * sizeof(int), cudaMemcpyHostToDevice);

	reduceSumIter(d_output, d_input, sizeOfInput, blockNum);

	cudaMemcpy(&h_output, d_output, sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(d_input);
	cudaFree(d_output);
	return h_output;
}

void prefixSum(int* h_output, int* input, int sizeOfInput)
{
	int* d_input;
	int* d_output;
	cudaMalloc(&d_input, sizeOfInput * sizeof(int));
	cudaMalloc(&d_output, sizeOfInput * sizeof(int));
	cudaMemcpy(d_input, input, sizeOfInput * sizeof(int), cudaMemcpyHostToDevice);
	prefixSumReCur(d_output, d_input, sizeOfInput);
	cudaMemcpy(h_output, d_output, sizeof(int)*sizeOfInput, cudaMemcpyDeviceToHost);
	cudaFree(d_input);
	cudaFree(d_output);
}

void histogram(int* h_output, int* input, int sizeOfInput, int histogramSize, int gap)
{
	int* d_input;
	int* d_output;
	unsigned int blockNum = 512;
	cudaMalloc(&d_input, sizeOfInput * sizeof(int));
	cudaMalloc(&d_output, histogramSize * blockNum * sizeof(int));
	cudaMemcpy(d_input, input, sizeOfInput * sizeof(int), cudaMemcpyHostToDevice);
	histogramLocal(d_output, d_input, sizeOfInput, histogramSize, gap, blockNum);
	cudaMemcpy(h_output, d_output, sizeof(int) *histogramSize, cudaMemcpyDeviceToHost);
	cudaFree(d_input);
	cudaFree(d_output);
}

void radixsort(int* h_output, int* input, int sizeOfInput)
{
	int* d_input;
	cudaMalloc(&d_input, sizeOfInput * sizeof(int));
	cudaMemcpy(d_input, input, sizeOfInput * sizeof(int), cudaMemcpyHostToDevice);

	radixsortIter(d_input, sizeOfInput);

	cudaMemcpy(h_output, d_input, sizeof(int)*sizeOfInput, cudaMemcpyDeviceToHost);
	cudaFree(d_input);
}

//cpu libary

int maxbit(int data[], int n) //辅助函数，求数据的最大位数
{
	int maxData = data[0];		///< 最大数
								/// 先求出最大数，再求其位数，这样有原先依次每个数判断其位数，稍微优化点。
	for (int i = 1; i < n; ++i)
	{
		if (maxData < data[i])
			maxData = data[i];
	}
	int d = 1;
	int p = 10;
	while (maxData >= p)
	{
		//p *= 10; // Maybe overflow
		maxData /= 10;
		++d;
	}
	return d;
}

void radixsortCPU(int data[], int n) //基数排序
{
	int d = maxbit(data, n);
	int *tmp = new int[n];
	int *count = new int[10]; //计数器
	int i, j, k;
	int radix = 1;
	for (i = 1; i <= d; i++) //进行d次排序
	{
		for (j = 0; j < 10; j++)
			count[j] = 0; //每次分配前清空计数器
		for (j = 0; j < n; j++)
		{
			k = (data[j] / radix) % 10; //统计每个桶中的记录数
			count[k]++;
		}
		for (j = 1; j < 10; j++)
			count[j] = count[j - 1] + count[j]; //将tmp中的位置依次分配给每个桶
		for (j = n - 1; j >= 0; j--) //将所有桶中记录依次收集到tmp中
		{
			k = (data[j] / radix) % 10;
			tmp[count[k] - 1] = data[j];
			count[k]--;
		}
		for (j = 0; j < n; j++) //将临时数组的内容复制到data中
			data[j] = tmp[j];
		radix = radix * 10;
	}
	delete[]tmp;
	delete[]count;
}

//single test comparing cpu & gpu speed one time.

void ParallelReduceSumTest()
{
	clock_t t1;
	clock_t t2;

	t1 = clock();

	int maxNum = 0;
	for (int i = 0; i < size1; i++)
	{
		maxNum += randNumbers[i];
	}

	t2 = clock();

	//cout << maxNum << endl;
	cpu_time += t2 - t1;
	//cout << "CPU Time(ms): " << setw(5) << t2 - t1 << " ";

	t1 = clock();

	int ans = reduceSum(randNumbers, size1);

	t2 = clock();

	//cout << ans << endl;
	gpu_time += t2 - t1;
	//cout << "GPU Time(ms): " << setw(5) << t2 - t1 << endl;
	if (maxNum != ans)
	{
		cout << "Wrong!" <<endl;
		cout << maxNum << " " << ans << endl;
	}
}

void ParallelPrefixSumTest()
{
	clock_t t1;
	clock_t t2;

	int* cpu_prefix_sum = new int[size1];
	int* gpu_prefix_sum = new int[size1];

	

	t1 = clock();

	for (int i = 0; i < size1; i++)
		cpu_prefix_sum[i] = 0;
	cpu_prefix_sum[0] = randNumbers[0];
	for (int i = 1; i < size1; i++)
			cpu_prefix_sum[i] = cpu_prefix_sum[i-1]+randNumbers[i];

	t2 = clock();
	cpu_time += t2 - t1;
	//cout << "CPU Time(ms): " << setw(5) << t2 - t1 << " ";

	t1 = clock();

	prefixSum(gpu_prefix_sum ,randNumbers , size1);

	t2 = clock();
	gpu_time += t2 - t1;
	//cout << "GPU Time(ms): " << setw(5) << t2 - t1 << endl;

	for (int i = 1; i < size1; i++)
	{
		
		if (cpu_prefix_sum[i-1] != gpu_prefix_sum[i])
		{
			cout << "Wrong! at " <<i<< endl;
			cout << cpu_prefix_sum[i-1] <<" "<< gpu_prefix_sum[i] << endl;
			cout << cpu_prefix_sum[i] << " " << gpu_prefix_sum[i+1] << endl;
			cout << cpu_prefix_sum[i+1] << " " << gpu_prefix_sum[i+2] << endl;
			break;
		}
		
		//cout << cpu_prefix_sum[i] << " " << gpu_prefix_sum[i] << endl;
	}
	delete[] cpu_prefix_sum;
	delete[] gpu_prefix_sum;
}

void ParallelHistogramTest(int histogramSize, int gap)
{
	clock_t t1;
	clock_t t2;
	int* cpu_histogram = new int[histogramSize];
	int* gpu_histogram = new int[histogramSize];

	for (int i = 0; i < histogramSize; i++)cpu_histogram[i] = 0;

	

	t1 = clock();

	for (int i = 0; i < size1; i++)
	{
		int j = randNumbers[i] / gap;
		if (j < histogramSize)cpu_histogram[j]++;
		else cpu_histogram[histogramSize-1]++;
	}

	t2 = clock();
	cpu_time += t2 - t1;
	//cout << "CPU Time(ms): " << setw(5) << t2 - t1<<" ";

	t1 = clock();

	histogram(gpu_histogram, randNumbers, size1, histogramSize, gap);

	t2 = clock();
	gpu_time += t2 - t1;
	//cout << "GPU Time(ms): " << setw(5) << t2 - t1 << endl;

	for (int i = 0; i < histogramSize; i++)
	{

		if (cpu_histogram[i] != gpu_histogram[i])
		{
			cout << "Wrong! at " << i << endl;
			cout << cpu_histogram[i] << " " << gpu_histogram[i] << endl;
			break;
		}

		//cout << cpu_histogram[i] << " " << gpu_histogram[i] << endl;
	}
	delete[] cpu_histogram;
	delete[] gpu_histogram;
}

void ParallelRadixSortTest()
{
	clock_t t1;
	clock_t t2;

	int* cpu_radixSort = new int[size1];
	int* gpu_radixSort = new int[size1];

	for (int i = 0; i < size1; i++)cpu_radixSort[i] = randNumbers[i];

	t1 = clock();

	radixsortCPU(cpu_radixSort,size1);

	//for (int i = 0; i < size1; i++)cout << cpu_radixSort[i] << endl;

	t2 = clock();
	cpu_time += t2 - t1;

	t1 = clock();

	//prefixSum(gpu_prefix_sum, randNumbers, size1);
	radixsort(gpu_radixSort, randNumbers, size1);

	t2 = clock();
	gpu_time += t2 - t1;

	for (int i = 0; i < size1; i++)
	{

		if (cpu_radixSort[i] != gpu_radixSort[i])
		{
			cout << "Wrong! at " << i << endl;
			cout << cpu_radixSort[i] << " " << gpu_radixSort[i] << endl;
			break;
		}

		//cout << cpu_prefix_sum[i] << " " << gpu_prefix_sum[i] << endl;
	}

	delete[] cpu_radixSort;
	delete[] gpu_radixSort;
}

//do lots of test comparing cpu & gpu average speed.

void ParallelReduceSumTestRepeatly()
{
	cout << "Start Cpu vs Gpu Sum Test" << endl;
	cpu_time = 0;
	gpu_time = 0;
	for (int i = 0; i < times; i++)
	{
		cout << ".";
		ParallelReduceSumTest();
	}cout << endl;
	cpu_time /= times;
	gpu_time /= times;
	cout << cpu_time << " " << gpu_time << endl;
}

void ParallelPrefixSumTestRepeatly()
{
	cout << "Start Cpu vs Gpu Prifix Sum Test" << endl;
	cpu_time = 0;
	gpu_time = 0;
	for (int i = 0; i < times; i++)
	{
		cout << ".";
		ParallelPrefixSumTest();
	}cout << endl;
	cpu_time /= times;
	gpu_time /= times;
	cout << cpu_time << " " << gpu_time << endl;
}

void ParallelHistogramTestRepeatly()
{
	cout << "Start Cpu vs Gpu Histogram Test" << endl;
	cpu_time = 0;
	gpu_time = 0;
	for (int i = 0; i < times; i++)
	{
		cout << ".";
		ParallelHistogramTest(100, 300);
	}cout << endl;
	cpu_time /= times;
	gpu_time /= times;
	cout << cpu_time << " " << gpu_time << endl;
}

void ParallelRadixSortTestRepeatly()
{
	cout << "Start Cpu vs Gpu Radix Sort Test" << endl;
	cpu_time = 0;
	gpu_time = 0;
	for (int i = 0; i < times; i++)
	{
		cout << ".";
		ParallelRadixSortTest();
	}cout << endl;
	cpu_time /= times;
	gpu_time /= times;
	cout << cpu_time << " " << gpu_time << endl;
}

//Code Entrance

int main()
{
	//cout.fill(' ');
	//cout.width(5);
	cudaSetDevice(0);
	cudaMallocHost(&randNumbers,size1*sizeof(int));

	srand(std::time(0));
	for (int i = 0; i < size1; i++)
	{
		randNumbers[i] = i+1;
	}
	
	ParallelReduceSumTestRepeatly();
	
	ParallelPrefixSumTestRepeatly();
	
	ParallelHistogramTestRepeatly();

	/*
	ParallelRadixSortTestRepeatly();
	*/
	system("pause");
	return 0;
}
