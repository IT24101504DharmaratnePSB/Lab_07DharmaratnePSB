#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1000000

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int chunk_size = N / size;

    int *array = NULL;
    if (rank == 0) {
        array = (int *)malloc(N * sizeof(int));
        for (int i = 0; i < N; i++)
            array[i] = i + 1;
        printf("Root filled array with values 1 to %d\n", N);
    }

    int *local_chunk = (int *)malloc(chunk_size * sizeof(int));

    double start = MPI_Wtime();

    MPI_Scatter(array, chunk_size, MPI_INT,
                local_chunk, chunk_size, MPI_INT,
                0, MPI_COMM_WORLD);

    long long local_sum = 0;
    for (int i = 0; i < chunk_size; i++)
        local_sum += local_chunk[i];

    printf("  Rank %d: summed local chunk of size %d => local_sum = %lld\n",
           rank, chunk_size, local_sum);

    /*
     * GATHER: Every process sends its local_sum; root collects them,
     * in rank order, into all_sums[]. Root still has to add them up itself —
     * Gather only collects data, it performs no computation.
     */
    long long *all_sums = NULL;
    if (rank == 0)
        all_sums = (long long *)malloc(size * sizeof(long long));

    MPI_Gather(&local_sum, 1, MPI_LONG_LONG,
               all_sums, 1, MPI_LONG_LONG,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        long long total_sum = 0;
        for (int r = 0; r < size; r++)
            total_sum += all_sums[r];

        double elapsed = MPI_Wtime() - start;
        long long expected = (long long)N * (N + 1) / 2;
        printf("\n[Gather] Total sum   = %lld\n", total_sum);
        printf("[Gather] Expected    = %lld\n", expected);
        printf("[Gather] Correct?    = %s\n", total_sum == expected ? "YES" : "NO");
        printf("[Gather] Time        = %.4f sec\n", elapsed);

        free(all_sums);
    }

    free(local_chunk);
    if (rank == 0) free(array);
    MPI_Finalize();
    return 0;
}
