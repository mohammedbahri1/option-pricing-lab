// Pricer Monte Carlo d'un call européen sous Black-Scholes, en C++17.
//
// - variables antithétiques ;
// - parallélisation OpenMP (une graine par thread, résultats reproductibles) ;
// - comparaison avec la formule fermée.
//
// Compilation : voir le Makefile (make), puis ./mc_pricer [n_paths]

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

struct Params {
    double S = 100.0, K = 100.0, T = 1.0, r = 0.05, sigma = 0.2;
};

double norm_cdf(double x) { return 0.5 * std::erfc(-x / std::sqrt(2.0)); }

double bs_call(const Params& p) {
    const double vst = p.sigma * std::sqrt(p.T);
    const double d1 = (std::log(p.S / p.K) + (p.r + 0.5 * p.sigma * p.sigma) * p.T) / vst;
    return p.S * norm_cdf(d1) - p.K * std::exp(-p.r * p.T) * norm_cdf(d1 - vst);
}

struct MCResult {
    double price, std_error;
};

MCResult mc_call(const Params& p, std::int64_t n_pairs, std::uint64_t seed) {
    const double drift = (p.r - 0.5 * p.sigma * p.sigma) * p.T;
    const double vol = p.sigma * std::sqrt(p.T);
    const double disc = std::exp(-p.r * p.T);

    double sum = 0.0, sum_sq = 0.0;

#pragma omp parallel reduction(+ : sum, sum_sq)
    {
        int tid = 0, n_threads = 1;
#ifdef _OPENMP
        tid = omp_get_thread_num();
        n_threads = omp_get_num_threads();
#endif
        std::mt19937_64 gen(seed + 1000003ULL * static_cast<std::uint64_t>(tid));
        std::normal_distribution<double> normal(0.0, 1.0);

        const std::int64_t chunk = n_pairs / n_threads;
        const std::int64_t begin = tid * chunk;
        const std::int64_t end = (tid == n_threads - 1) ? n_pairs : begin + chunk;

        for (std::int64_t i = begin; i < end; ++i) {
            const double z = normal(gen);
            const double up = p.S * std::exp(drift + vol * z);
            const double down = p.S * std::exp(drift - vol * z);
            // moyenne de la paire antithétique : une seule observation i.i.d.
            const double x = 0.5 * disc * (std::max(up - p.K, 0.0) + std::max(down - p.K, 0.0));
            sum += x;
            sum_sq += x * x;
        }
    }

    const double n = static_cast<double>(n_pairs);
    const double mean = sum / n;
    const double var = (sum_sq - n * mean * mean) / (n - 1.0);
    return {mean, std::sqrt(var / n)};
}

int main(int argc, char** argv) {
    const std::int64_t n_paths = (argc > 1) ? std::atoll(argv[1]) : 10'000'000;
    const Params p;

    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif

    const auto t0 = std::chrono::steady_clock::now();
    const MCResult res = mc_call(p, n_paths / 2, 42);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

    const double exact = bs_call(p);
    std::cout << std::fixed << std::setprecision(6)
              << "Trajectoires     : " << n_paths << " (" << threads << " thread(s))\n"
              << "Prix Monte Carlo : " << res.price << "  (erreur std " << res.std_error << ")\n"
              << "Prix exact BSM   : " << exact << "\n"
              << "Écart            : " << std::abs(res.price - exact) << "\n"
              << std::setprecision(1) << "Temps            : " << ms << " ms\n";
    return 0;
}
