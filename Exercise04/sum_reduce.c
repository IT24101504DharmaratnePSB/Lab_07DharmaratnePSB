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
     * REDUCE: combines every process's local_sum with MPI_SUM and
     * deposits the single result in total_sum, but ONLY on root.
     * Internally MPI uses an O(log P) tree algorithm — no manual loop needed.
     */
    long long total_sum = 0;
    MPI_Reduce(&local_sum, &total_sum, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double elapsed = MPI_Wtime() - start;
        long long expected = (long long)N * (N + 1) / 2;
        printf("\n[Reduce] Total sum   = %lld\n", total_sum);
        printf("[Reduce] Expected    = %lld\n", expected);
        printf("[Reduce] Correct?    = %s\n", total_sum == expected ? "YES" : "NO");
        printf("[Reduce] Time        = %.4f sec\n", elapsed);
    }
    /* Note: on non-root processes, total_sum is left at 0 (its initial
     * value) rather than a meaningful result — MPI_Reduce only writes
     * a valid answer into recvbuf on the root process. */

    free(local_chunk);
    if (rank == 0) free(array);
    MPI_Finalize();
    return 0;
}
