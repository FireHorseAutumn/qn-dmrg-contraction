// Rung-1 driver (DESIGN.md section 8): generate a synthetic sector profile,
// run the per-block cuBLAS loop over the matched list, then the dense
// reference contraction (full dense tensors, zeros outside matched blocks),
// and apply gate 0: ||diff||_F <= 1e-12 * ||ref||_F in FP64. No timing is
// recorded here; this rung is correctness only.
//
// usage: qnd_ref <profile> <chi> [seed]

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <random>
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

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: qnd_ref <profile> <chi> [seed]\nprofiles:");
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
  const std::uint64_t seed = argc > 3 ? std::strtoull(argv[3], nullptr, 10) : 12345;

  const qnd::Schedule sched = qnd::build_schedule(*prof, chi);
  if (sched.triples.empty()) {
    std::fprintf(stderr, "empty matched list\n");
    return 2;
  }

  // Host fill: normal(0,1), blocks in schedule order, column-major within a
  // block. Deterministic given the seed.
  std::mt19937_64 rng(seed);
  std::normal_distribution<double> gauss;
  std::vector<double> h_L(sched.arenaL), h_A(sched.arenaA);
  for (const auto& t : sched.triples)
    for (std::int64_t j = 0; j < t.k; ++j)
      for (std::int64_t i = 0; i < t.m; ++i) h_L[t.offL + i + j * t.ldL] = gauss(rng);
  for (std::size_t iq = 0; iq < sched.sectors.size(); ++iq) {
    if (!sched.has_A[iq]) continue;
    const std::int64_t k = sched.sectors[iq].m;
    std::int32_t comp_q[qnd::kMaxCharges];
    for (int c = 0; c < qnd::kMaxCharges; ++c)
      comp_q[c] = prof->total[c] - sched.sectors[iq].q[c];
    const int comp = qnd::find_sector(sched.sectors, comp_q);
    const std::int64_t n = prof->phys_dim * sched.sectors[comp].m;
    for (std::int64_t j = 0; j < n; ++j)
      for (std::int64_t i = 0; i < k; ++i)
        h_A[sched.offA[iq] + i + j * sched.ldA[iq]] = gauss(rng);
  }

  // Dense oracle assembly on the host.
  const std::int64_t Mg = sched.Mg, Kg = sched.Kg, Ng = sched.Ng;
  std::vector<double> h_Ld(Mg * Kg, 0.0), h_Ad(Kg * Ng, 0.0);
  for (const auto& t : sched.triples) {
    const std::int64_t r0 = sched.sec_off[t.p_sector];
    const std::int64_t c0 = sched.sec_off[t.q_sector];
    for (std::int64_t j = 0; j < t.k; ++j)
      for (std::int64_t i = 0; i < t.m; ++i)
        h_Ld[(r0 + i) + (c0 + j) * Mg] = h_L[t.offL + i + j * t.ldL];
  }
  for (std::size_t iq = 0; iq < sched.sectors.size(); ++iq) {
    if (!sched.has_A[iq]) continue;
    const std::int64_t k = sched.sectors[iq].m;
    std::int32_t comp_q[qnd::kMaxCharges];
    for (int c = 0; c < qnd::kMaxCharges; ++c)
      comp_q[c] = prof->total[c] - sched.sectors[iq].q[c];
    const int comp = qnd::find_sector(sched.sectors, comp_q);
    const std::int64_t n = prof->phys_dim * sched.sectors[comp].m;
    const std::int64_t r0 = sched.sec_off[iq], c0 = sched.acol_off[iq];
    for (std::int64_t j = 0; j < n; ++j)
      for (std::int64_t i = 0; i < k; ++i)
        h_Ad[(r0 + i) + (c0 + j) * Kg] = h_A[sched.offA[iq] + i + j * sched.ldA[iq]];
  }

  double *d_L, *d_A, *d_C, *d_Ld, *d_Ad, *d_T;
  cu_check(cudaMalloc(&d_L, sched.arenaL * sizeof(double)), "alloc L");
  cu_check(cudaMalloc(&d_A, sched.arenaA * sizeof(double)), "alloc A");
  cu_check(cudaMalloc(&d_C, sched.arenaC * sizeof(double)), "alloc C");
  cu_check(cudaMalloc(&d_Ld, Mg * Kg * sizeof(double)), "alloc Ld");
  cu_check(cudaMalloc(&d_Ad, Kg * Ng * sizeof(double)), "alloc Ad");
  cu_check(cudaMalloc(&d_T, Mg * Ng * sizeof(double)), "alloc T");
  cu_check(cudaMemcpy(d_L, h_L.data(), h_L.size() * sizeof(double), cudaMemcpyHostToDevice), "up L");
  cu_check(cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(double), cudaMemcpyHostToDevice), "up A");
  cu_check(cudaMemcpy(d_Ld, h_Ld.data(), h_Ld.size() * sizeof(double), cudaMemcpyHostToDevice), "up Ld");
  cu_check(cudaMemcpy(d_Ad, h_Ad.data(), h_Ad.size() * sizeof(double), cudaMemcpyHostToDevice), "up Ad");

  cublasHandle_t handle;
  cb_check(cublasCreate(&handle), "create");
  const double one = 1.0, zero = 0.0;

  // Per-block cuBLAS loop over the matched list.
  for (const auto& t : sched.triples)
    cb_check(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, t.m, t.n, t.k, &one,
                         d_L + t.offL, t.ldL, d_A + sched.offA[t.q_sector],
                         sched.ldA[t.q_sector], &zero, d_C + t.offC, t.ldC),
             "block gemm");

  // Dense reference contraction.
  cb_check(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, Mg, Ng, Kg, &one, d_Ld, Mg,
                       d_Ad, Kg, &zero, d_T, Mg),
           "dense gemm");

  std::vector<double> h_C(sched.arenaC), h_T(Mg * Ng);
  cu_check(cudaMemcpy(h_C.data(), d_C, h_C.size() * sizeof(double), cudaMemcpyDeviceToHost), "down C");
  cu_check(cudaMemcpy(h_T.data(), d_T, h_T.size() * sizeof(double), cudaMemcpyDeviceToHost), "down T");

  // Gate 0: compare every C block against its region of the dense product.
  long double diff2 = 0.0L, ref2 = 0.0L;
  for (const auto& t : sched.triples) {
    const std::int64_t r0 = sched.sec_off[t.p_sector];
    const std::int64_t c0 = sched.acol_off[t.q_sector];
    for (std::int64_t j = 0; j < t.n; ++j)
      for (std::int64_t i = 0; i < t.m; ++i) {
        const double ref = h_T[(r0 + i) + (c0 + j) * Mg];
        const double got = h_C[t.offC + i + j * t.ldC];
        diff2 += static_cast<long double>(ref - got) * (ref - got);
        ref2 += static_cast<long double>(ref) * ref;
      }
  }
  const double diff = std::sqrt(static_cast<double>(diff2));
  const double ref = std::sqrt(static_cast<double>(ref2));
  const double tol = 1e-12 * ref;
  const bool pass = diff <= tol;

  std::printf("profile=%s chi=%lld seed=%llu sectors=%zu triples=%zu top1_flop=%.3f\n",
              prof->name, static_cast<long long>(chi),
              static_cast<unsigned long long>(seed), sched.sectors.size(),
              sched.triples.size(), qnd::top1_flop_share(sched));
  std::printf("gate0: ||diff||_F=%.3e ||ref||_F=%.3e tol=%.3e -> %s\n", diff, ref, tol,
              pass ? "PASS" : "FAIL");

  cublasDestroy(handle);
  cudaFree(d_L);
  cudaFree(d_A);
  cudaFree(d_C);
  cudaFree(d_Ld);
  cudaFree(d_Ad);
  cudaFree(d_T);
  return pass ? 0 : 1;
}
