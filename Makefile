
vmcache: vmcache.cpp tpcc/*pp ycsb/*pp
	g++ -DNDEBUG -O3 -std=c++20 -g -fnon-call-exceptions -fasynchronous-unwind-tables vmcache.cpp -o vmcache -laio -lnuma

vmcache_debug: vmcache.cpp tpcc/*pp ycsb/*pp
	g++ -O1 -g -fsanitize=address -fno-omit-frame-pointer -std=c++20 vmcache.cpp -o vmcache_debug -laio -lnuma


vmcache2: vmcache2.cpp tpcc/*pp
	g++ -DNDEBUG -O3 -std=c++20 -g -fnon-call-exceptions -fasynchronous-unwind-tables vmcache2.cpp -o vmcache2 -laio -lnuma

clean:
	rm -f vmcache vmcache2 vmcache_debug vmcache_old

vmcache_memtrk: vmcache_memtrk.cpp tpcc/*pp ycsb/*pp
	g++ -DNDEBUG -O3 -std=c++20 -g -fnon-call-exceptions -fasynchronous-unwind-tables vmcache_memtrk.cpp -o vmcache_memtrk -laio -lnuma

vmcache_in_mem: vmcache_in-mem.cpp tpcc/*pp ycsb/*pp
	g++ -DNDEBUG -O3 -std=c++20 -g -fnon-call-exceptions -fasynchronous-unwind-tables vmcache_in-mem.cpp -o vmcache_in-mem -laio -lnuma
