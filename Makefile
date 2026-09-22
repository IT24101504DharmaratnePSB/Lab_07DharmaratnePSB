CC      = mpicc
CFLAGS  = -O2 -Wall
NP      = 4
MPIRUN  = mpirun -np $(NP)

BCAST     = Exercise01/sum_bcast
SCATTER   = Exercise02/sum_scatter
GATHER    = Exercise03/sum_gather
REDUCE    = Exercise04/sum_reduce
ALLREDUCE = Exercise05/sum_allreduce
SCAN      = Exercise06/sum_scan

BINARIES = $(BCAST) $(SCATTER) $(GATHER) $(REDUCE) $(ALLREDUCE) $(SCAN)

.PHONY: all clean run \
        run-bcast run-scatter run-gather run-reduce run-allreduce run-scan

all: $(BINARIES)

$(BCAST): $(BCAST).c
	$(CC) $(CFLAGS) -o $@ $<

$(SCATTER): $(SCATTER).c
	$(CC) $(CFLAGS) -o $@ $<

$(GATHER): $(GATHER).c
	$(CC) $(CFLAGS) -o $@ $<

$(REDUCE): $(REDUCE).c
	$(CC) $(CFLAGS) -o $@ $<

$(ALLREDUCE): $(ALLREDUCE).c
	$(CC) $(CFLAGS) -o $@ $<

$(SCAN): $(SCAN).c
	$(CC) $(CFLAGS) -o $@ $<

# Run all 6 programs, one after another, with NP processes (default 4).
# Override process count with: make run NP=8
run: all run-bcast run-scatter run-gather run-reduce run-allreduce run-scan

run-bcast: $(BCAST)
	@echo "=== Exercise 1: Bcast + Send/Recv (np=$(NP)) ==="
	$(MPIRUN) ./$(BCAST)
	@echo

run-scatter: $(SCATTER)
	@echo "=== Exercise 2: Scatter + Send/Recv (np=$(NP)) ==="
	$(MPIRUN) ./$(SCATTER)
	@echo

run-gather: $(GATHER)
	@echo "=== Exercise 3: Scatter + Gather (np=$(NP)) ==="
	$(MPIRUN) ./$(GATHER)
	@echo

run-reduce: $(REDUCE)
	@echo "=== Exercise 4: Scatter + Reduce (np=$(NP)) ==="
	$(MPIRUN) ./$(REDUCE)
	@echo

run-allreduce: $(ALLREDUCE)
	@echo "=== Exercise 5: Scatter + Allreduce (np=$(NP)) ==="
	$(MPIRUN) ./$(ALLREDUCE)
	@echo

run-scan: $(SCAN)
	@echo "=== Exercise 6: Scatter + Scan (np=$(NP)) ==="
	$(MPIRUN) ./$(SCAN)
	@echo

clean:
	rm -f $(BINARIES)
