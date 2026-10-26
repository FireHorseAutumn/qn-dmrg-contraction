// Synthetic quantum-number-blocked contraction profiles (DESIGN.md §8 rung
// 1): seeded sector generation under the Gaussian sector-weight model,
// single-arena block packing, and the matched-block schedule. Pure host
// code; no DMRG code exists here and none is claimed — the profiles are
// parameter sets (centers, sigmas, chi) following DESIGN.md §2.
#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

namespace qnd {

constexpr double kWeightCutoff = 1e-6;
constexpr int kMaxCharges = 4;

struct Profile {
  const char* name;
  int ncharges;          // 2 (single-band) or 4 (two-orbital)
  double center[kMaxCharges];   // Gaussian center per charge, in sector labels
  double sigma[kMaxCharges];    // width per charge
  int label_max[kMaxCharges];   // sector labels run 0..label_max per charge
  int total[kMaxCharges];       // total charge Q
  int phys_dim;        // d
};

// Presets follow the DESIGN.md §2 sector statistics. Unused charges are
// pinned: label_max = 0, center = 0 (weight exp(0) = 1 exactly).
// two-orb-L64 is the DESIGN.md §2 two-orbital control: 4 U(1) charges,
// sigma 0.55 each (~300 blocks at L = 64), Hubbard-like d = 16 (4^2
// single-site states for two spin-1/2 orbitals).
inline const Profile* profiles() {
  static const Profile p[] = {
      {"tj-L64", 2, {14.0, 14.0, 0.0, 0.0}, {0.55, 0.55, 1.0, 1.0}, {32, 32, 0, 0}, {28, 28, 0, 0}, 3},
      {"tj-L256", 2, {56.0, 56.0, 0.0, 0.0}, {0.67, 0.67, 1.0, 1.0}, {128, 128, 0, 0}, {112, 112, 0, 0}, 3},
      {"hubbard-hf-L64", 2, {16.0, 16.0, 0.0, 0.0}, {0.30, 0.55, 1.0, 1.0}, {32, 32, 0, 0}, {32, 32, 0, 0}, 4},
      {"cylinder-w4", 2, {14.0, 14.0, 0.0, 0.0}, {1.1, 1.1, 1.0, 1.0}, {40, 40, 0, 0}, {28, 28, 0, 0}, 3},
      {"two-orb-L64", 4, {14.0, 14.0, 14.0, 14.0}, {0.55, 0.55, 0.55, 0.55}, {32, 32, 32, 32}, {28, 28, 28, 28}, 16},
  };
  return p;
}
inline constexpr int kNumProfiles = 5;

struct Sector {
  std::int32_t q[kMaxCharges];
  std::int64_t m;
  double weight;
};

inline std::vector<Sector> generate_sectors(const Profile& prof, std::int64_t chi) {
  std::vector<Sector> out;
  for (std::int32_t n0 = 0; n0 <= prof.label_max[0]; ++n0) {
    const double d0 = (n0 - prof.center[0]) / prof.sigma[0];
    const double w0 = std::exp(-0.5 * d0 * d0);
    for (std::int32_t n1 = 0; n1 <= prof.label_max[1]; ++n1) {
      const double d1 = (n1 - prof.center[1]) / prof.sigma[1];
      const double w1 = w0 * std::exp(-0.5 * d1 * d1);
      for (std::int32_t n2 = 0; n2 <= prof.label_max[2]; ++n2) {
        const double d2 = (n2 - prof.center[2]) / prof.sigma[2];
        const double w2 = w1 * std::exp(-0.5 * d2 * d2);
        for (std::int32_t n3 = 0; n3 <= prof.label_max[3]; ++n3) {
          const double d3 = (n3 - prof.center[3]) / prof.sigma[3];
          const double w = w2 * std::exp(-0.5 * d3 * d3);
          if (w < kWeightCutoff) continue;
          std::int64_t m = static_cast<std::int64_t>(std::llround(static_cast<double>(chi) * w));
          if (m < 1) m = 1;
          out.push_back({{n0, n1, n2, n3}, m, w});
        }
      }
    }
  }
  return out;
}

inline int find_sector(const std::vector<Sector>& secs, const std::int32_t* q) {
  for (std::size_t i = 0; i < secs.size(); ++i)
    if (secs[i].q[0] == q[0] && secs[i].q[1] == q[1] &&
        secs[i].q[2] == q[2] && secs[i].q[3] == q[3])
      return static_cast<int>(i);
  return -1;
}

struct Triple {
  std::int64_t m, k, n;      // C(m x n) = L(m x k) * A(k x n)
  int p_sector;              // row block of T/L
  int q_sector;              // column block of T, column block of L, row block of A
  std::int64_t offL, ldL;    // offsets in doubles within each tensor's arena
  std::int64_t offC, ldC;
};

struct Schedule {
  std::vector<Sector> sectors;
  std::vector<Triple> triples;
  std::vector<std::int64_t> offA, ldA;     // per sector q with complement present
  std::vector<bool> has_A;
  std::vector<std::int64_t> sec_off;       // prefix sums of m over sectors (M_g = K_g)
  std::vector<std::int64_t> acol_off;      // A/T column-block prefix per sector
  std::int64_t arenaL = 0, arenaA = 0, arenaC = 0;  // sizes in doubles
  std::int64_t Mg = 0, Kg = 0, Ng = 0;
};

inline std::int64_t align32(std::int64_t doubles) { return (doubles + 31) & ~std::int64_t(31); }
inline std::int64_t pad_ld(std::int64_t rows) { return (rows + 3) & ~std::int64_t(3); }

// MPO bond sectors for a nearest-neighbor model: delta = 0 or +-1 on one
// charge (DESIGN.md section 3; D_w = 1 + 2*ncharges: 5 for single-band,
// 9 for two-orbital).
inline Schedule build_schedule(const Profile& prof, std::int64_t chi) {
  Schedule s;
  s.sectors = generate_sectors(prof, chi);
  const int nc = prof.ncharges;
  std::int32_t deltas[1 + 2 * kMaxCharges][kMaxCharges] = {};
  for (int c = 0; c < nc; ++c) {
    deltas[1 + 2 * c][c] = 1;
    deltas[2 + 2 * c][c] = -1;
  }
  const int ndelta = 1 + 2 * nc;

  // A blocks: one per sector whose charge complement Q - q is also present.
  s.offA.assign(s.sectors.size(), -1);
  s.ldA.assign(s.sectors.size(), 0);
  s.has_A.assign(s.sectors.size(), false);
  for (std::size_t i = 0; i < s.sectors.size(); ++i) {
    const auto& sec = s.sectors[i];
    std::int32_t comp_q[kMaxCharges];
    for (int c = 0; c < kMaxCharges; ++c) comp_q[c] = prof.total[c] - sec.q[c];
    const int comp = find_sector(s.sectors, comp_q);
    if (comp < 0) continue;
    s.has_A[i] = true;
    s.ldA[i] = pad_ld(sec.m);
    s.offA[i] = s.arenaA;
    s.arenaA += align32(s.ldA[i] * (prof.phys_dim * s.sectors[comp].m));
  }

  // Matched triples and L/C blocks.
  for (int di = 0; di < ndelta; ++di) {
    const auto& delta = deltas[di];
    for (std::size_t iq = 0; iq < s.sectors.size(); ++iq) {
      const auto& qsec = s.sectors[iq];
      std::int32_t pq[kMaxCharges], comp_q[kMaxCharges];
      for (int c = 0; c < kMaxCharges; ++c) {
        pq[c] = qsec.q[c] + delta[c];
        comp_q[c] = prof.total[c] - qsec.q[c];
      }
      const int ip = find_sector(s.sectors, pq);
      if (ip < 0 || !s.has_A[iq]) continue;
      const int comp = find_sector(s.sectors, comp_q);
      Triple t;
      t.p_sector = ip;
      t.q_sector = static_cast<int>(iq);
      t.m = s.sectors[ip].m;
      t.k = qsec.m;
      t.n = prof.phys_dim * s.sectors[comp].m;
      t.ldL = pad_ld(t.m);
      t.offL = s.arenaL;
      s.arenaL += align32(t.ldL * t.k);
      t.ldC = pad_ld(t.m);
      t.offC = s.arenaC;
      s.arenaC += align32(t.ldC * t.n);
      s.triples.push_back(t);
    }
  }

  // Dense-oracle index maps.
  s.sec_off.assign(s.sectors.size(), 0);
  s.acol_off.assign(s.sectors.size(), 0);
  std::int64_t acc = 0, acol = 0;
  for (std::size_t i = 0; i < s.sectors.size(); ++i) {
    s.sec_off[i] = acc;
    acc += s.sectors[i].m;
    s.acol_off[i] = acol;
    if (s.has_A[i]) {
      std::int32_t comp_q[kMaxCharges];
      for (int c = 0; c < kMaxCharges; ++c)
        comp_q[c] = prof.total[c] - s.sectors[i].q[c];
      const int comp = find_sector(s.sectors, comp_q);
      acol += prof.phys_dim * s.sectors[comp].m;
    }
  }
  s.Mg = acc;
  s.Kg = acc;
  s.Ng = acol;
  return s;
}

// FLOP concentration record (DESIGN.md section 2 uses sum of m^3-style
// block costs; here the exact matched-triple cost 2*m*k*n).
inline double top1_flop_share(const Schedule& s) {
  double total = 0.0, top = 0.0;
  for (const auto& t : s.triples) {
    const double f = 2.0 * static_cast<double>(t.m) * t.k * t.n;
    total += f;
    if (f > top) top = f;
  }
  return total > 0.0 ? top / total : 0.0;
}

}  // namespace qnd
