#include <cinttypes>
#include <cstdio>
#include <vector>
#include "qnd/profile.hpp"

static int failures = 0;

static void check(bool ok, const char* what) {
  if (!ok) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
  }
}

int main() {
  for (int i = 0; i < qnd::kNumProfiles; ++i) {
    const qnd::Profile& p = qnd::profiles()[i];

    // Determinism: identical inputs give byte-identical schedules.
    const qnd::Schedule a = qnd::build_schedule(p, 128);
    const qnd::Schedule b = qnd::build_schedule(p, 128);
    check(a.sectors.size() == b.sectors.size() && a.triples.size() == b.triples.size(),
          "schedule size determinism");
    bool same = true;
    for (std::size_t t = 0; t < a.triples.size(); ++t)
      same &= a.triples[t].offL == b.triples[t].offL && a.triples[t].offC == b.triples[t].offC &&
              a.triples[t].m == b.triples[t].m;
    check(same, "schedule byte determinism");

    // chi scaling: doubling chi doubles every sector dimension within
    // rounding and never changes the sector count.
    const qnd::Schedule big = qnd::build_schedule(p, 256);
    check(big.sectors.size() == a.sectors.size(), "sector count chi-invariant");
    for (std::size_t s = 0; s < a.sectors.size(); ++s) {
      const std::int64_t want = 2 * a.sectors[s].m;
      const std::int64_t delta = big.sectors[s].m - want;
      check(delta >= -1 && delta <= 1, "m doubles with chi");
    }

    // Alignment: block starts on 256 B (32 doubles), leading dims padded to
    // 32 B (4 doubles).
    for (const auto& t : a.triples) {
      check(t.offL % 32 == 0 && t.offC % 32 == 0, "arena offsets 256 B aligned");
      check(t.ldL % 4 == 0 && t.ldC % 4 == 0 && t.ldL >= t.m && t.ldC >= t.m,
            "leading dims padded and valid");
    }
    for (std::size_t s = 0; s < a.sectors.size(); ++s)
      if (a.has_A[s]) check(a.offA[s] % 32 == 0 && a.ldA[s] % 4 == 0, "A blocks aligned");

    // Matched-list consistency: every triple references real blocks with
    // the DESIGN.md charge arithmetic n = d * m_{Q-q}.
    for (const auto& t : a.triples) {
      check(t.p_sector >= 0 && t.q_sector >= 0 && a.has_A[t.q_sector], "triple blocks exist");
      std::int32_t comp_q[qnd::kMaxCharges];
      for (int c = 0; c < qnd::kMaxCharges; ++c)
        comp_q[c] = p.total[c] - a.sectors[t.q_sector].q[c];
      const int comp = qnd::find_sector(a.sectors, comp_q);
      check(comp >= 0 && t.n == p.phys_dim * a.sectors[comp].m, "n = d * m_{Q-q}");
      check(t.m == a.sectors[t.p_sector].m && t.k == a.sectors[t.q_sector].m,
            "m/k match sector dimensions");
    }

    const qnd::Schedule rec = qnd::build_schedule(p, 1000);
    std::printf("%-16s chi=1000 sectors=%zu triples=%zu top1_flop=%.3f Mg=%" PRId64
                " Ng=%" PRId64 "\n",
                p.name, rec.sectors.size(), rec.triples.size(), qnd::top1_flop_share(rec),
                rec.Mg, rec.Ng);
  }

  if (failures != 0) {
    std::fprintf(stderr, "%d unit checks failed\n", failures);
    return 1;
  }
  std::puts("unit: all checks passed");
  return 0;
}
