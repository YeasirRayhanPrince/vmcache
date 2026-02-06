#!/usr/bin/env bash
set -euo pipefail

# Core configuration
export BLOCK=${BLOCK:-/users/yrayhan/vmcache/vm}
export VIRTGB=${VIRTGB:-16}
export PHYSGB=${PHYSGB:-4}
export EXMAP=${EXMAP:-0}

# Benchmark and eviction configuration
export BATCH=${BATCH:-64}
export RUNFOR=${RUNFOR:-100}
export RNDREAD=${RNDREAD:-0}
export THREADS=${THREADS:-64}
export DATASIZE=${DATASIZE:-20}

# Execute
./vmcache_old
