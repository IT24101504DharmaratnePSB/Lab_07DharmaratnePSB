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

Test environment: AWS EC2 t2.micro instance (Amazon Linux 2023), 1 vCPU, MPICH 3.4.1.
All 6 programs run correctly and produce Total sum = 500,000,500,000 at every process
count tested (np = 2, 4, 8) - all runs printed "Correct? = YES".

| Program      | np=2 (sec) | np=4 (sec) | np=8 (sec) |
|--------------|------------|------------|------------|
| sum_bcast        | 0.3050     | 1.2901     | 2.8200     |
| sum_scatter      | 0.1752     | 0.5700     | 1.2698     |
| sum_gather       | 0.1598     | 0.6005     | 1.3800     |
| sum_reduce       | 0.1635     | 0.5151     | 1.3600     |
| sum_allreduce    | 0.1700     | 0.5401     | 1.4303     |
| sum_scan         | 0.1572     | 0.4675     | 1.5173     |

Observations and explanation:

* sum_bcast is the slowest program at every process count, and the gap widens as np
  grows (0.31s -> 1.29s -> 2.82s). This matches expectations: MPI_Bcast ships the
  *entire* 1,000,000-element array to every process, regardless of how many processes
  there are, while every other program only sends each process its own N/size-sized
  chunk via MPI_Scatter. More processes means more full-array copies being made and
  transmitted, so Bcast's cost scales worst with np.

* sum_scatter, sum_gather, sum_reduce, sum_allreduce, and sum_scan all cluster closely
  together at every process count (e.g. at np=4: 0.51-0.60 sec). We expected Reduce,
  Allreduce, and Scan to noticeably outperform Gather because they use an O(log P)
  tree/butterfly algorithm internally instead of Gather's more naive collection, but on
  this hardware the differences between them are small and inconsistent (sometimes
  Reduce is fastest, sometimes Scatter is). This is because the theoretical algorithmic
  advantage only becomes visible when communication between separate CPU cores is the
  bottleneck - and that isn't the case here.

* Counter-intuitively, EVERY program got slower as np increased, rather than faster.
  This is the opposite of what "more parallelism" usually promises, and it is a direct
  consequence of the test hardware: a t2.micro instance has only 1 vCPU. Running
  np=8 does not put 8 processes on 8 cores - it timeslices 8 processes onto a single
  core. Adding more processes past the core count only adds context-switching and
  scheduling overhead with no genuine parallel speedup, so wall-clock time increases
  with np instead of decreasing. On a machine with 8 or more real cores, we would
  expect the opposite trend (time decreasing as np grows, up to the core count), and
  we would expect Reduce/Allreduce/Scan to pull further ahead of Gather as np grows,
  since their O(log P) advantage over O(P) grows with P.

* Fastest overall: sum_scan and sum_reduce were consistently among the quickest at
  np=2 and np=4; sum_scatter and sum_reduce were quickest at np=8. sum_bcast was the
  slowest at every process count tested, by a wide and growing margin.

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