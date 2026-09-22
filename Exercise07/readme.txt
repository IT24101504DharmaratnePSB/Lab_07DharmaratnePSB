Exercise 7: Summary Comparison
===============================

1. Comparison table
--------------------

| # | Program           | Collectives used              | Array allocation                  | Manual summation loop on root?         | Where is the final result available?          |
|---|--------------------|--------------------------------|------------------------------------|------------------------------------------|-------------------------------------------------|
| 1 | sum_bcast.c        | MPI_Bcast                     | Every process allocates the FULL array (N ints) | Yes - root loops over `size-1` MPI_Recv calls and adds each partial sum | Root only |
| 2 | sum_scatter.c      | MPI_Scatter                   | Only root allocates the full array; every process allocates a small `local_chunk` (N/size ints) | Yes - root still loops over MPI_Recv calls | Root only |
| 3 | sum_gather.c       | MPI_Scatter, MPI_Gather        | Only root allocates the full array; every process allocates `local_chunk` | Yes - root loops over the `all_sums[]` array gathered by MPI_Gather | Root only |
| 4 | sum_reduce.c       | MPI_Scatter, MPI_Reduce        | Only root allocates the full array; every process allocates `local_chunk` | No - MPI_Reduce(..., MPI_SUM, ...) does the addition internally | Root only (recvbuf on non-root ranks is not defined) |
| 5 | sum_allreduce.c    | MPI_Scatter, MPI_Allreduce     | Only root allocates the full array; every process allocates `local_chunk` | No - MPI_Allreduce does the addition internally | All processes |
| 6 | sum_scan.c         | MPI_Scatter, MPI_Scan          | Only root allocates the full array; every process allocates `local_chunk` | No - MPI_Scan does the addition internally | All processes, but each rank gets a DIFFERENT value (its own prefix sum, not the same global total) |

Key trend: as we move from Exercise 1 -> 6, memory usage drops (full array -> chunk only),
point-to-point loops are progressively replaced by single collective calls, and the amount
of code (and the number of places a bug can hide) on root keeps shrinking.

2. Timing runs (2, 4, 8 processes)
------------------------------------

Run `make run NP=2`, `make run NP=4`, and `make run NP=8` (or `mpirun -np <P> ./program`
for each program directly) and fill in the wall-clock times printed by each program
(the "Time = ... sec" line) below. Actual numbers depend on your machine/cluster, core
count, and network, so record what you observe rather than these placeholders:

| Program      | np=2 (sec) | np=4 (sec) | np=8 (sec) |
|--------------|------------|------------|------------|
| sum_bcast        |            |            |            |
| sum_scatter      |            |            |            |
| sum_gather       |            |            |            |
| sum_reduce       |            |            |            |
| sum_allreduce    |            |            |            |
| sum_scan         |            |            |            |

What to expect and why:

* sum_bcast is usually the slowest, especially as `np` grows: MPI_Bcast ships the
  entire 1,000,000-element array to every process (wasted bandwidth and memory), and
  the final collection is a root-side loop of blocking MPI_Recv calls that costs O(P) time.
* sum_scatter is faster than sum_bcast because each process only receives its own
  N/size chunk instead of the whole array - the communication volume per process drops
  from N to N/size. The Send/Recv collection loop is unchanged, so it is still O(P).
* sum_gather is about the same speed as sum_scatter for the communication of results
  (Gather is also O(P) work in the worst case for a naive implementation) but replaces
  a hand-written loop with a single, less error-prone library call. Any speed gain over
  Send/Recv is implementation-dependent (MPICH/OpenMPI often optimize Gather internally).
* sum_reduce should be noticeably faster than sum_gather at higher process counts:
  MPI_Reduce uses a tree-based algorithm (O(log P) instead of O(P)), and it never
  materializes the `all_sums[]` array or a manual addition loop.
* sum_allreduce costs a little more than sum_reduce for the same np (it must deliver
  the result to every process, not just root), but it is still cheaper than doing
  MPI_Reduce followed by a separate MPI_Bcast, because Allreduce is implemented with a
  single, purpose-built algorithm (e.g. recursive doubling/butterfly) instead of two
  sequential steps.
* sum_scan does similar work to sum_allreduce (also O(log P)-ish) but computes a
  running prefix instead of a single global value, so timings should be close to
  sum_allreduce's.
* As `np` increases, all versions using Scatter benefit from smaller per-process chunks
  and communication volume, but sum_bcast does not, because every process still
  receives the full N-element array regardless of `np` - so the gap between sum_bcast
  and the rest widens as `np` grows.

3. Thinking question: MPI_Scan vs MPI_Allreduce
--------------------------------------------------

Use MPI_Scan instead of MPI_Allreduce whenever each process needs a value that depends
on its POSITION in the group relative to the others, not just the overall group total.
Allreduce gives every rank the identical final answer; Scan gives every rank a distinct,
cumulative answer.

Concrete example - parallel file writing / global index assignment:
Suppose each process has produced a variable number of output records (`local_count`)
that must be written into one shared output file, in rank order, with no gaps or
overlaps. Each process needs to know the starting offset (in records) at which to begin
writing its own data. This offset is exactly `sum_before_me` from Exercise 6:

    MPI_Scan(&local_count, &prefix_count, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    long long my_offset = prefix_count - local_count;   /* records written by earlier ranks */

Rank 0 starts at offset 0, rank 1 starts right after rank 0's records, and so on -
computed with a single collective call and no extra point-to-point messages. Using
MPI_Allreduce here would give every rank only the grand total record count, which is
useless for deciding *where* to start writing; it does not tell a process how much data
came before it in rank order. Other classic uses of MPI_Scan: assigning globally unique
IDs to locally generated items, computing cumulative distribution functions across
distributed data, and load-balanced partitioning where each rank needs to know the
running total of work assigned to ranks before it.
