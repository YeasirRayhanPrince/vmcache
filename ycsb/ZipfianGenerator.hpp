#pragma once
#include <cmath>
#include <cstdint>
#include <random>

// Zipfian distribution generator using rejection-inversion method
// Default theta=0.99 matches standard YCSB
class ZipfianGenerator {
   double theta;
   uint64_t n;
   double alpha;
   double zetaN;
   double eta;
   double hIntegralX1;

   double hIntegral(double x) const {
      double logX = std::log(x);
      return helper2((1.0 - theta) * logX) * logX;
   }

   double h(double x) const {
      return std::exp(-(theta) * std::log(x));
   }

   // (exp(x) - 1) / x, numerically stable for small x
   static double helper2(double x) {
      if (std::abs(x) > 1e-8)
         return std::expm1(x) / x;
      return 1.0 + x * 0.5 * (1.0 + x / 3.0 * (1.0 + x * 0.25));
   }

   double hIntegralInverse(double x) const {
      double t = x * (1.0 - theta);
      if (t < -1.0) t = -1.0;
      return std::exp(helper1(t) * x);
   }

   // log(1 + x) / x, numerically stable for small x
   static double helper1(double x) {
      if (std::abs(x) > 1e-8)
         return std::log1p(x) / x;
      return 1.0 - x * 0.5 * (1.0 - x / 3.0 * (1.0 - x * 0.25));
   }

public:
   ZipfianGenerator(uint64_t n, double theta = 0.99)
      : theta(theta), n(n) {
      alpha = 1.0 / (1.0 - theta);
      zetaN = hIntegral(n + 1);
      hIntegralX1 = hIntegral(1.5) - 1.0;
      eta = zetaN - hIntegralX1;
   }

   // Returns value in [0, n)
   template <class Engine>
   uint64_t operator()(Engine& eng) {
      std::uniform_real_distribution<double> dist(0.0, 1.0);
      while (true) {
         double u = eta * dist(eng) + hIntegralX1;
         double x = hIntegralInverse(u);
         uint64_t k = static_cast<uint64_t>(x + 0.5);
         if (k < 1) k = 1;
         if (k > n) k = n;
         if (k - x <= theta || u >= hIntegral(k + 0.5) - h(k)) {
            return k - 1; // zero-indexed
         }
      }
   }
};
