
# vmcache-n     : 3-tier buffer manager (DRAM -> remote NUMA -> SSD). Set REMOTEGB>0
#                 to enable the remote tier; REMOTEGB=0 falls back to 2-tier.
# vmcache-leis  : original 2-tier buffer manager (DRAM -> SSD). No NUMA code.

vmcache-n: vmcache-n.cpp tpcc/*pp ycsb/*pp
	g++ -DNDEBUG -O3 -std=c++20 -g -fnon-call-exceptions -fasynchronous-unwind-tables vmcache-n.cpp -o vmcache-n -laio -lnuma

vmcache-n-debug: vmcache-n.cpp tpcc/*pp ycsb/*pp
	g++ -O1 -g -fsanitize=address -fno-omit-frame-pointer -std=c++20 vmcache-n.cpp -o vmcache-n-debug -laio -lnuma

# Baseline has no YCSB workload, hence no ycsb/ dependency.
vmcache-leis: vmcache-leis.cpp tpcc/*pp
	g++ -DNDEBUG -O3 -std=c++20 -g -fnon-call-exceptions -fasynchronous-unwind-tables vmcache-leis.cpp -o vmcache-leis -laio -lnuma

# vmcache-n plus rdtsc instrumentation for disk-I/O and page-migration cycles.
vmcache-n-memtrk: vmcache-n-memtrk.cpp tpcc/*pp ycsb/*pp
	g++ -DNDEBUG -O3 -std=c++20 -g -fnon-call-exceptions -fasynchronous-unwind-tables vmcache-n-memtrk.cpp -o vmcache-n-memtrk -laio -lnuma

clean:
	rm -f vmcache-n vmcache-leis vmcache-n-debug vmcache-n-memtrk
