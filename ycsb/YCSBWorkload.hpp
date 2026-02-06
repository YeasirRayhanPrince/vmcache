#pragma once
#include "ZipfianGenerator.hpp"
#include <cstring>
#include <vector>
#include <atomic>
#include <random>

static constexpr unsigned YCSB_FIELD_SIZE = 100; // 10 fields x 10 bytes

struct ycsb_t {
   static constexpr int id = 20;
   struct Key {
      u64 ycsb_key;
   };
   u8 field[YCSB_FIELD_SIZE];

   static constexpr unsigned maxFoldLength() { return sizeof(u64); }

   template <class T>
   static unsigned foldKey(uint8_t* out, const T& key) {
      unsigned pos = 0;
      pos += fold(out + pos, key.ycsb_key);
      return pos;
   }

   template <class T>
   static unsigned unfoldKey(const uint8_t* in, T& key) {
      unsigned pos = 0;
      pos += unfold(in + pos, key.ycsb_key);
      return pos;
   }
};

enum class YCSBOp : u8 { Read, Update, Insert, Scan, ReadModifyWrite };

struct YCSBOperation {
   u64 key;
   YCSBOp op;
};

enum class YCSBWorkloadType : char { A='A', B='B', C='C', D='D', E='E', F='F' };

template <template<typename> class AdapterType>
struct YCSBWorkload {
   AdapterType<ycsb_t>& table;
   u64 recordCount;
   std::atomic<u64> nextInsertKey;
   double theta;

   YCSBWorkload(AdapterType<ycsb_t>& table, u64 recordCount, double theta = 0.99)
      : table(table), recordCount(recordCount), nextInsertKey(recordCount), theta(theta) {}

   void load(unsigned nthreads, u64 count,
             void (*parallel_for_fn)(uint64_t, uint64_t, uint64_t,
                std::function<void(uint64_t, uint64_t, uint64_t)>)) {
      parallel_for_fn(0, count, nthreads, [&](uint64_t worker, uint64_t begin, uint64_t end) {
         workerThreadId = worker;
         for (u64 i = begin; i < end; i++) {
            ycsb_t record;
            RandomGenerator::getRandString(record.field, YCSB_FIELD_SIZE);
            table.insert({i}, record);
         }
      });
   }

   std::vector<YCSBOperation> generateOps(u64 count, YCSBWorkloadType type, unsigned seed = 42) {
      std::vector<YCSBOperation> ops;
      ops.reserve(count);
      std::mt19937_64 gen(seed);
      ZipfianGenerator zipf(recordCount, theta);
      std::uniform_real_distribution<double> coinFlip(0.0, 1.0);
      std::uniform_int_distribution<u64> scanLen(1, 100);

      // Shuffle keys so zipfian rank doesn't correlate with B-tree key order
      // (prevents hot keys from being physically adjacent)
      std::vector<u64> shuffledKeys(recordCount);
      std::iota(shuffledKeys.begin(), shuffledKeys.end(), 0);
      std::shuffle(shuffledKeys.begin(), shuffledKeys.end(), gen);

      for (u64 i = 0; i < count; i++) {
         YCSBOperation op;
         double coin = coinFlip(gen);
         switch (type) {
            case YCSBWorkloadType::A: // 50/50 read/update
               op.key = shuffledKeys[zipf(gen)];
               op.op = (coin < 0.5) ? YCSBOp::Read : YCSBOp::Update;
               break;
            case YCSBWorkloadType::B: // 95/5 read/update
               op.key = shuffledKeys[zipf(gen)];
               op.op = (coin < 0.95) ? YCSBOp::Read : YCSBOp::Update;
               break;
            case YCSBWorkloadType::C: // 100% read
               op.key = shuffledKeys[zipf(gen)];
               op.op = YCSBOp::Read;
               break;
            case YCSBWorkloadType::D: // 95% read latest, 5% insert
               op.key = shuffledKeys[zipf(gen)]; // key will be adjusted at runtime for "latest"
               op.op = (coin < 0.95) ? YCSBOp::Read : YCSBOp::Insert;
               break;
            case YCSBWorkloadType::E: // 95% scan, 5% insert
               op.key = shuffledKeys[zipf(gen)];
               op.op = (coin < 0.95) ? YCSBOp::Scan : YCSBOp::Insert;
               break;
            case YCSBWorkloadType::F: // 50% read, 50% read-modify-write
               op.key = shuffledKeys[zipf(gen)];
               op.op = (coin < 0.5) ? YCSBOp::Read : YCSBOp::ReadModifyWrite;
               break;
         }
         ops.push_back(op);
      }
      return ops;
   }

   void executeOp(const YCSBOperation& op) {
      switch (op.op) {
         case YCSBOp::Read: {
            ycsb_t result;
            table.lookup1({op.key}, [&](const ycsb_t& r) {
               result = r;
            });
            break;
         }
         case YCSBOp::Update: {
            table.update1({op.key}, [&](ycsb_t& r) {
               // Update a random 10-byte segment
               unsigned offset = RandomGenerator::getRand<unsigned>(0, YCSB_FIELD_SIZE - 10);
               RandomGenerator::getRandString(r.field + offset, 10);
            });
            break;
         }
         case YCSBOp::Insert: {
            u64 newKey = nextInsertKey.fetch_add(1);
            ycsb_t record;
            RandomGenerator::getRandString(record.field, YCSB_FIELD_SIZE);
            table.insert({newKey}, record);
            break;
         }
         case YCSBOp::Scan: {
            u64 scanLength = RandomGenerator::getRand<u64>(1, 100);
            u64 count = 0;
            table.scan({op.key}, [&](const ycsb_t::Key&, const ycsb_t&) {
               count++;
               return count < scanLength;
            }, [](){});
            break;
         }
         case YCSBOp::ReadModifyWrite: {
            ycsb_t result;
            table.lookup1({op.key}, [&](const ycsb_t& r) {
               result = r;
            });
            table.update1({op.key}, [&](ycsb_t& r) {
               unsigned offset = RandomGenerator::getRand<unsigned>(0, YCSB_FIELD_SIZE - 10);
               RandomGenerator::getRandString(r.field + offset, 10);
            });
            break;
         }
      }
   }
};
