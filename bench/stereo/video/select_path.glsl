// The last path selects directly from its register-held eight-direction sums.
int path_minimum(int value, uint lane, uint local_line, bool use_subgroup) {
  if (use_subgroup) return subgroupMin(value);
  uint base = local_line * 32;
  fallback_minimum[base + lane] = value;
  barrier();
  for (uint span = 16; span > 0; span /= 2) {
    if (lane < span) fallback_minimum[base + lane] = min(fallback_minimum[base + lane], fallback_minimum[base + lane + span]);
    barrier();
  }
  int result = fallback_minimum[base];
  barrier();
  return result;
}
int decode_score(int value) { return value >= 32768 ? 8000000 + value - 32768 : value; }
void select_pixel(ivec2 scores[CHUNKS], uint lane, uint local_line,
                  bool use_subgroup, bool valid_pixel, int pixel) {
  int best = 2147483647;
  for (int chunk = 0; chunk < CHUNKS; ++chunk) {
    uint d = 2 * (chunk * 32 + lane);
    if (d < DISPARITIES) best = min(best, scores[chunk].x);
    if (d + 1 < DISPARITIES) best = min(best, scores[chunk].y);
  }
  best = path_minimum(best, lane, local_line, use_subgroup);
  int winner = DISPARITIES;
  for (int chunk = 0; chunk < CHUNKS; ++chunk) {
    int d = 2 * (chunk * 32 + int(lane));
    if (d < DISPARITIES && scores[chunk].x == best) winner = min(winner, d);
    if (d + 1 < DISPARITIES && scores[chunk].y == best) winner = min(winner, d + 1);
  }
  winner = path_minimum(winner, lane, local_line, use_subgroup);
  int second = 2147483647;
  for (int chunk = 0; chunk < CHUNKS; ++chunk) {
    int d = 2 * (chunk * 32 + int(lane));
    if (d < DISPARITIES && abs(d - winner) > 1) second = min(second, scores[chunk].x);
    if (d + 1 < DISPARITIES && abs(d + 1 - winner) > 1) second = min(second, scores[chunk].y);
  }
  second = path_minimum(second, lane, local_line, use_subgroup);
  float result = float(winner);
  if (!use_subgroup) {
    for (int chunk = 0; chunk < CHUNKS; ++chunk) fallback_previous[local_line * CHUNKS * 32 + chunk * 32 + lane] = scores[chunk];
    barrier();
  }
  if (winner > 0 && winner + 1 < DISPARITIES) {
    int lower = 0;
    int upper = 0;
    if (use_subgroup) {
      for (int chunk = 0; chunk < CHUNKS; ++chunk) {
        // Both shuffle calls execute in every active subgroup lane.
        ivec2 lo = subgroupShuffle(scores[chunk], uint((winner - 1) / 2) % 32);
        ivec2 hi = subgroupShuffle(scores[chunk], uint((winner + 1) / 2) % 32);
        if ((winner - 1) / 64 == chunk) lower = lo[(winner - 1) % 2];
        if ((winner + 1) / 64 == chunk) upper = hi[(winner + 1) % 2];
      }
    } else {
      uint base = local_line * CHUNKS * 32;
      lower = fallback_previous[base + (winner - 1) / 2][(winner - 1) % 2];
      upper = fallback_previous[base + (winner + 1) / 2][(winner + 1) % 2];
    }
    float lo = float(decode_score(lower));
    float hi = float(decode_score(upper));
    float denominator = lo + hi - 2.0 * float(best);
    if (denominator > 0.0) result += clamp((lo - hi) / (2.0 * denominator), -0.5, 0.5);
  }
  // Every valid score <= 20664; even its 1.1 threshold is below an invalid
  // marker (32768), so packed scores preserve the uniqueness comparison.
  if (lane == 0 && valid_pixel) disparity[pixel] = float(second) <= float(best) * 1.1 ? -1.0 : result;
  if (!use_subgroup) barrier();
}
