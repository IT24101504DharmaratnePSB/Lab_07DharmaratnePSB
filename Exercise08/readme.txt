Exercise 8: Compile and Run
=============================

A Makefile at the repository root builds all 6 programs and can run them all with a
single command:

    make            # build all 6 binaries (Exercise01..Exercise06)
    make run        # run all 6 with 4 processes (default)
    make run NP=8   # run all 6 with 8 processes instead
    make clean      # remove built binaries

Individual targets are also available if you only want one program, e.g.:

    make run-bcast
    make run-scatter
    make run-gather
    make run-reduce
    make run-allreduce
    make run-scan

Under the hood each target compiles with `mpicc -O2 -Wall` and runs with
`mpirun -np $(NP) ./program`, matching the compile/run commands given in the main
Readme.md.

Remember to push the source files (Exercise01/sum_bcast.c ... Exercise06/sum_scan.c),
this Makefile, and your written answers (Exercise07) to your GitHub repository.
