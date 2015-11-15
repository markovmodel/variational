#include <stdlib.h>
#include <stdio.h>

void _subtract_row_double(double* X, double* row, int M, int N)
{
        int i, j, ro;
        for (i=0; i!=M; ++i)
        {
                ro = i*N;
                for (j=0; j!=N; ++j)
                {
                        X[ro + j] -= row[j];
                }
        }
}

void _subtract_row_float(float* X, float* row, int M, int N)
{
        int i, j, ro;
        for (i=0; i!=M; ++i)
        {
                ro = i*N;
                for (j=0; j!=N; ++j)
                {
                        X[ro + j] -= row[j];
                }
        }
}

int* _bool_to_list(int* b, int N, int nnz)
{
        int i;
        int k=0;
        int* list = (int*)malloc(nnz*sizeof(int));
        for (i=0; i<N; i++)
                if (b[i] == 1)
                        list[k++] = i;
        return (list);
}

void _nonconstant_cols_char(int* cols, char* X, int M, int N)
{
        // compare first and last row to get constant candidates
        int i,j;
        int ro = (M-1)*N;

        // by default all 0 (constant)
        for (j=0; j<N; j++)
                cols[j] = 0;

        // go through all rows in order to confirm constant candidates
        for (i=0; i<M; i++)
        {
                ro = i*N;
                for (j=0; j<N; j++)
                {
                        if (X[j] != X[ro+j])
                                cols[j] = 1;
                }
        }
}


void _nonconstant_cols_int(int* cols, int* X, int M, int N)
{
        // compare first and last row to get constant candidates
        int i,j;
        int ro = (M-1)*N;

        // by default all 0 (constant)
        for (j=0; j<N; j++)
                cols[j] = 0;

        // go through all rows in order to confirm constant candidates
        for (i=0; i<M; i++)
        {
                ro = i*N;
                for (j=0; j<N; j++)
                {
                        if (X[j] != X[ro+j])
                                cols[j] = 1;
                }
        }
}


void _nonconstant_cols_long(int* cols, long* X, int M, int N)
{
        // compare first and last row to get constant candidates
        int i,j;
        int ro = (M-1)*N;

        // by default all 0 (constant)
        for (j=0; j<N; j++)
                cols[j] = 0;

        // go through all rows in order to confirm constant candidates
        for (i=0; i<M; i++)
        {
                ro = i*N;
                for (j=0; j<N; j++)
                {
                        if (X[j] != X[ro+j])
                                cols[j] = 1;
                }
        }
}


void _nonconstant_cols_float(int* cols, float* X, int M, int N)
{
        // compare first and last row to get constant candidates
        int i,j;
        int ro = (M-1)*N;

        // by default all 0 (constant)
        for (j=0; j<N; j++)
                cols[j] = 0;

        // go through all rows in order to confirm constant candidates
        for (i=0; i<M; i++)
        {
                ro = i*N;
                for (j=0; j<N; j++)
                {
                        if (X[j] != X[ro+j])
                                cols[j] = 1;
                }
        }
}


void _nonconstant_cols_double(int* cols, double* X, int M, int N)
{
        // compare first and last row to get constant candidates
        int i,j;
        int ro = (M-1)*N;

        // by default all 0 (constant)
        for (j=0; j<N; j++)
                cols[j] = 0;

        // go through all rows in order to confirm constant candidates
        for (i=0; i<M; i++)
        {
                ro = i*N;
                for (j=0; j<N; j++)
                {
                        if (X[j] != X[ro+j])
                                cols[j] = 1;
                }
        }
}
