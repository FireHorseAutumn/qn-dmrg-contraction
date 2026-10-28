// Gate 1 (DESIGN.md section 6): the histogram decides. Measures, per
// profile and chi:
//   - the exact matched-triple FLOP histogram (2*m*k*n per triple):
//     top-1/4/8 shares, triple count, tail count and tail FLOP share
//     (tail := max(m, k, n) < 64);
//   - the per-block cuBLAS loop wall-clock split: full matched list vs
//     head-only vs tail-only (CUDA-event bracketed, median of reps);
//     tail wall share = tail-only / full.
// Pre-committed decision rule (DESIGN.md): top-8 FLOP share >= 95% AND
// tail wall share < 20% => the kernel idea dies for that profile.
// No dense oracle, no correctness gate here — gate 0 lives in qnd_ref.
//
// usage: qnd_gate1 <profile> <chi> [reps]

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "qnd/profile.hpp"

namespace {

void cu_check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) {
    std::fprintf(stderr, "cuda error at %s: %s\n", what, cudaGetErrorString(e));
    std::exit(2);
  }
}

void cb_check(cublasStatus_t s, const char* what) {
  if (s != CUBLAS_STATUS_SUCCESS) {
    std::fprintf(stderr, "cublas error at %s: %d\n", what, static_cast<int>(s));
    std::exit(2);
  }
}

bool is_tail(const qnd::Triple& t) {
  return std::max({t.m, t.k, t.n}) < 64;
}

// Median event-bracketed wall time of the block loop over a subset.
double time_subset(cublasHandle_t handle, const qnd::Schedule& sched,
                   const double* d_L, const double* d_A, double* d_C,
                   bool want_tail, int reps) {
  const double one = 1.0, zero = 0.0;
  std::vector<double> ts;
  cudaEvent_t ev0, ev1;
  cu_check(cudaEventCreate(&ev0), "event create");
  cu_check(cudaEventCreate(&ev1), "event create");
  for (int r = 0; r < reps; ++r) {
    cu_check(cudaDeviceSynchronize(), "pre sync");
    cu_check(cudaEventRecord(ev0), "record");
    for (const auto& t : sched.triples) {
      if (is_tail(t) != want_tail) continue;
      cb_check(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, t.m, t.n, t.k,
                           &one, d_L + t.offL, t.ldL,
                           d_A + sched.offA[t.q_sector], sched.ldA[t.q_sector],
                           &zero, d_C + t.offC, t.ldC),
               "block gemm");
    }
    cu_check(cudaEventRecord(ev1), "record");
    cu_check(cudaDeviceSynchronize(), "post sync");
    float ms = 0.0f;
    cu_check(cudaEventElapsedTime(&ms, ev0, ev1), "elapsed");
    ts.push_back(static_cast<double>(ms));
  }
  cudaEventDestroy(ev0);
  cudaEventDestroy(ev1);
  std::sort(ts.begin(), ts.end());
  return ts[ts.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: qnd_gate1 <profile> <chi> [reps]\nprofiles:");
    for (int i = 0; i < qnd::kNumProfiles; ++i)
      std::fprintf(stderr, " %s", qnd::profiles()[i].name);
    std::fprintf(stderr, "\n");
    return 2;
  }
  const qnd::Profile* prof = nullptr;
  for (int i = 0; i < qnd::kNumProfiles; ++i)
    if (std::strcmp(argv[1], qnd::profiles()[i].name) == 0) prof = &qnd::profiles()[i];
  if (!prof) {
    std::fprintf(stderr, "unknown profile %s\n", argv[1]);
    return 2;
  }
  const std::int64_t chi = std::atoll(argv[2]);
  const int reps = argc > 3 ? std::atoi(argv[3]) : 20;
  if (chi < 1) {
    std::fprintf(stderr, "qnd_gate1: chi must be >= 1\n");
    return 2;
  }
  if (reps < 1) {
    std::fprintf(stderr, "qnd_gate1: reps must be >= 1\n");
    return 2;
  }

  const qnd::Schedule sched = qnd::build_schedule(*prof, chi);
  if (sched.triples.empty()) {
    std::fprintf(stderr, "empty matched list\n");
    return 2;
  }

  // Exact FLOP histogram (host, no model — this IS the schedule).
  std::vector<double> flops;
  double total = 0.0, tail_flops = 0.0;
  std::size_t tail_count = 0;
  for (const auto& t : sched.triples) {
    const double f = 2.0 * static_cast<double>(t.m) * t.k * t.n;
    flops.push_back(f);
    total += f;
    if (is_tail(t)) {
      tail_flops += f;
      ++tail_count;
    }
  }
  std::sort(flops.rbegin(), flops.rend());
  auto topk = [&](std::size_t k) {
    double s = 0.0;
    for (std::size_t i = 0; i < k && i < flops.size(); ++i) s += flops[i];
    return s / total;
  };

  double *d_L, *d_A, *d_C;
  cu_check(cudaMalloc(&d_L, sched.arenaL * sizeof(double)), "alloc L");
  cu_check(cudaMalloc(&d_A, sched.arenaA * sizeof(double)), "alloc A");
  cu_check(cudaMalloc(&d_C, sched.arenaC * sizeof(double)), "alloc C");
  cu_check(cudaMemset(d_L, 0, sched.arenaL * sizeof(double)), "memset L");
  cu_check(cudaMemset(d_A, 0, sched.arenaA * sizeof(double)), "memset A");

  cublasHandle_t handle;
  cb_check(cublasCreate(&handle), "create");

  // Warmup both subsets (cuBLAS heuristic/clock state), then measure.
  time_subset(handle, sched, d_L, d_A, d_C, false, 2);
  time_subset(handle, sched, d_L, d_A, d_C, true, 2);
  const double head_ms = time_subset(handle, sched, d_L, d_A, d_C, false, reps);
  const double tail_ms = time_subset(handle, sched, d_L, d_A, d_C, true, reps);
  const double tail_wall =
      (head_ms + tail_ms) > 0.0 ? tail_ms / (head_ms + tail_ms) : 0.0;

  const double top8 = topk(8);
  const bool dies = (top8 >= 0.95) && (tail_wall < 0.20);
  std::printf("profile=%s chi=%lld triples=%zu tail(<64)=%zu\n", prof->name,
              static_cast<long long>(chi), sched.triples.size(), tail_count);
  std::printf("flop shares: top1=%.3f top4=%.3f top8=%.3f tail=%.3f\n",
              topk(1), topk(4), top8, tail_flops / total);
  std::printf("wall ms (median of %d): head=%.3f tail=%.3f tail_share=%.3f\n",
              reps, head_ms, tail_ms, tail_wall);
  std::printf("gate1 rule (top8>=0.95 AND tail_wall<0.20): %s\n",
              dies ? "DIE for this profile" : "survives");

  cublasDestroy(handle);
  cudaFree(d_L);
  cudaFree(d_A);
  cudaFree(d_C);
  return 0;
}
