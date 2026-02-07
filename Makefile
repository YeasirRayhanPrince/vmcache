
vmcache: vmcache.cpp tpcc/*pp ycsb/*pp
	g++ -DNDEBUG -O3 -std=c++20 -g -fnon-call-exceptions -fasynchronous-unwind-tables vmcache.cpp -o vmcache -laio -lnuma

vmcache_debug: vmcache.cpp tpcc/*pp ycsb/*pp
	g++ -O1 -g -fsanitize=address -fno-omit-frame-pointer -std=c++20 vmcache.cpp -o vmcache_debug -laio -lnuma


vmcache_old: vmcache_old.cpp tpcc/*pp
	g++ -DNDEBUG -O3 -std=c++20 -g -fnon-call-exceptions -fasynchronous-unwind-tables vmcache_old.cpp -o vmcache_old -laio -lnuma

clean:
	rm -f vmcache vmcache_debug vmcache_old
