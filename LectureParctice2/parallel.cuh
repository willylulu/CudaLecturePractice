#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include "kernel.cuh"

using namespace std;

/*
print an array of gpu data.
*/
void printGPU(int* input, int sizeA)
{
	int* temp = new int[sizeA];
	cudaMemcpy(temp, input, sizeof(int)*sizeA, cudaMemcpyDeviceToHost);
	for (int i = 0; i < sizeA; i++)
	{
		cout << temp[i] << endl;
	}
	delete[] temp;
}

/*
Input an array(d_input) having size of sizeOfInput.
And them are seperated to blocks havinf size of Block_Size.
Adding numbers in each block together and storing in output array(d_output).
Output array having size of blockNum need to add togeter to get a final answer.
*/
void reduceSumPerBlock(int* d_output, int* d_input, int sizeOfInput, int blockNum)
{
	d_reduceSum << <blockNum, Block_Size, Block_Size * sizeof(int) >> > (d_input, d_output, sizeOfInput);
}

void reduceSumIter(int* d_output, int* d_input, int sizeOfInput, int blockNum)
{
	int* d_temp;
	cudaMalloc(&d_temp, blockNum * sizeof(int));
	reduceSumPerBlock(d_temp, d_input, sizeOfInput, blockNum);
	reduceSumPerBlock(d_output, d_temp, blockNum, 1);
	cudaFree(d_temp);
}

/*
Input an array(d_input) and it will prescan to Output array(d_output).
*/
void prefixSumReCur(int* d_output, int* d_input, int sizeOfInput)
{
	unsigned int blockNum = sizeOfInput / Block_Size;
	if (blockNum * Block_Size < sizeOfInput)blockNum++;
	prescan << <blockNum, Block_Size, Block_Size * sizeof(int) >> > (d_output, d_input, sizeOfInput); //cudaDeviceSynchronize();
	if (blockNum == 1)return;
	else
	{
		int* d_auxiliaryInput;
		int* d_auxiliaryOutput;
		cudaMalloc(&d_auxiliaryInput, blockNum * sizeof(int));
		cudaMalloc(&d_auxiliaryOutput, blockNum * sizeof(int));
		int auxiliaryBlock = blockNum / Block_Size;
		if (auxiliaryBlock*Block_Size < blockNum)auxiliaryBlock++;
		formAuxiliary << <auxiliaryBlock, Block_Size >> > (d_auxiliaryInput, d_output, d_input, blockNum); //cudaDeviceSynchronize();
		prefixSumReCur(d_auxiliaryOutput, d_auxiliaryInput, blockNum);
		addAuxiliary << <blockNum, Block_Size >> > (d_output, d_auxiliaryOutput); //cudaDeviceSynchronize();
		cudaFree(d_auxiliaryInput);
		cudaFree(d_auxiliaryOutput);
	}
}

/*
Input an array(d_input) and define an histogram having size of histogramSize and number of gap.
Output an array(d_output) having sizeof blockNum * histogramSize.

Output an array need to be seperate to many section and each section have size of blockNum.
Adding each section together and then it will be a final histogram.
*/
void histogramLocal(int* d_output, int* d_input, int sizeOfInput, int histogramSize, int gap, int blockNum)
{
	int* d_temp;
	unsigned int gridSize = blockNum*Block_Size;
	cudaMalloc(&d_temp, histogramSize * gridSize * sizeof(int));
	d_histogram << <blockNum, Block_Size >> > (d_input, d_temp, sizeOfInput, histogramSize, gap);
	//cudaDeviceSynchronize();
	for (int i = 0; i < histogramSize; i++)
		reduceSumIter(d_output + i , d_temp + i * gridSize, gridSize, blockNum);
	cudaFree(d_temp);
}

/*
Input an array and use parallel radix sort
Bacause It change It's own value, Output is the Input.
*/
void radixsortIter(int* d_input, int sizeOfInput)
{
	int* d_temp;
	int* d_temp2;
	int* d_condition;
	int h_firstCount;
	int* d_conditionIndex;

	unsigned int blockNum = sizeOfInput / Block_Size;
	if (blockNum * Block_Size < sizeOfInput)blockNum++;

	cudaMalloc(&d_temp, sizeOfInput * sizeof(int));
	cudaMalloc(&d_temp2, sizeOfInput * sizeof(int));
	cudaMalloc(&d_condition, sizeOfInput * sizeof(int));
	cudaMalloc(&d_conditionIndex, sizeOfInput * sizeof(int));

	int radix = 1;
	for (int i = 0; i < 32; i++)
	{

		d_conditionInput << <blockNum, Block_Size >> > (d_condition, d_input, radix);

		prefixSumReCur(d_conditionIndex, d_condition, sizeOfInput);

		d_placeCondition << <blockNum, Block_Size >> > (d_conditionIndex, d_condition, d_temp, d_input);

		d_reverseCondition << <blockNum, Block_Size >> > (d_condition);

		prefixSumReCur(d_conditionIndex, d_condition, sizeOfInput);

		d_placeCondition << <blockNum, Block_Size >> > (d_conditionIndex, d_condition, d_temp2, d_input);

		d_arrayCat << <blockNum, Block_Size >> > (d_input,d_temp,d_temp2, d_conditionIndex, d_condition, sizeOfInput);

		radix *= 2;
	}

	cudaFree(d_conditionIndex);
	cudaFree(d_condition);
	cudaFree(d_temp2);
	cudaFree(d_temp);
}