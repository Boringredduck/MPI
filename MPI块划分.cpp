#include<mpi.h>
#include <time.h>
#include<ctime>
#include<stdlib.h>
#include <stdio.h>
#include <iostream>
using namespace std;
const int N=1000;
void LU(float A[][N], int rank, int num_proc)
{
	int block = N / num_proc, remain = N % num_proc;
	int begin = rank * block;
	int end = rank == num_proc - 1 ? begin + block + remain : begin + block;

	for (int k = 0; k < N; k++) {
		if (k >= begin && k < end)
		{
			for (int j = k + 1; j < N; j++)
				A[k][j] = A[k][j] / A[k][k];
			A[k][k] = 1.0;
			for (int p = 0; p < num_proc; p++)
			{
				if (p != rank)
					MPI_Send(&A[k], N, MPI_FLOAT, p, 2, MPI_COMM_WORLD);
			}
		}
		else
		{
			int cur_p = k / block;
			MPI_Recv(&A[k], N, MPI_FLOAT, cur_p, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
		for (int i = begin; i < end && i < N; i++)
		{
			if (i >= k + 1) {
				for (int j = k + 1; j < N; j++)
					A[i][j] = A[i][j] - A[i][k] * A[k][j];
				A[i][k] = 0.0;
			}
		}
	}
}
void f_mpi()
{
	float A[N][N];
	//初始化数组
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			A[i][j] = rand();
		}
	}

	int num_proc;//进程数
	int rank;//识别调用进程的rank,值从0-size-1
	MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	int block = N / num_proc, remain = N % num_proc;
	//任务划分
	clock_t start = clock();
	if (rank == 0) {

		for (int i = 1; i < num_proc; i++)
		{
			if (i != num_proc - 1) {
				for (int j = 0; j < block + remain; j++)
					MPI_Send(&A[i * block + j], N, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
			}
			else {
				for (int j = 0; j < block + remain; j++)
					MPI_Send(&A[i * block + j], N, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
			}

		}

		LU(A, rank, num_proc);
		for (int i = 1; i < num_proc; i++)
		{
			if (i != num_proc - 1)
				for (int j = 0; j < block; j++)
				{
					MPI_Recv(&A[i * block + j], N, MPI_FLOAT, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				}
			else
			{
				for (int j = 0; j < block + remain; j++)
					MPI_Recv(&A[i * block + j], N, MPI_FLOAT, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			}
		}

		clock_t end = clock();
		double elapsed_time = double(end - start) / CLOCKS_PER_SEC * 1000.0;
		cout << "MPI块划分运行时间为：" << elapsed_time << " ms\n";
	}
//其它进程
	else {
		if (rank != num_proc - 1) {
			for (int j = 0; j < block + remain; j++)
				MPI_Recv(&A[rank*block+j],N,MPI_FLOAT,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
		}
		else {
			for (int j = 0; j < block + remain; j++)
				MPI_Recv(&A[rank * block + j], N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

		}
		LU(A, rank, num_proc);
		if (rank != num_proc - 1) {
			for (int j = 0; j < block; j++)
				MPI_Send(&A[rank * block + j], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
		}
		else {
			for (int j = 0; j < block + remain; j++)
				MPI_Send(&A[rank * block + j], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
		}
	}
}

int main(int argc, char** argv) {
	int rank, size;
	MPI_Init(&argc, &argv);
	
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	size = 5;

	if (size <= 1) {
		std::cout << "There should be at least 2 processes for MPI program!" << std::endl;
		MPI_Finalize();
		return -1;
	}

	f_mpi();

	MPI_Finalize();
	return 0;
}
	
	