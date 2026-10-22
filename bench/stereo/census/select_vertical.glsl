// Copyright 2016 Fixstars Corporation
// SPDX-License-Identifier: Apache-2.0
// Select from the last path's register-held costs, as in select_path.glsl.
// The two horizontal volumes and the vertical-only recurrence stay distinct.
void select_pixel(uint vertical[BLOCK], int x, int y, uint lane, uint base) {
  int pixel = y * WIDTH + x;
  uint values[BLOCK];
  for (int k = 0; k < BLOCK; ++k) values[k] = vertical[k];
  for (int p = 0; p < 2; ++p) {
    int offset = (p * HEIGHT * WIDTH + pixel) * (DISPARITIES / PATH_PACK) + int(lane) * (BLOCK / PATH_PACK);
    for (int k = 0; k < BLOCK; k += PATH_PACK) {
#if PATH_PACK == 8
      uvec2 words = paths[offset + k / PATH_PACK];
      uint word = words.x;
      values[k + 4] += words.y & 255u;
      values[k + 5] += (words.y >> 8) & 255u;
      values[k + 6] += (words.y >> 16) & 255u;
      values[k + 7] += words.y >> 24;
#else
      uint word = paths[offset + k / PATH_PACK];
#endif
      values[k] += word & 255u;
      values[k + 1] += (word >> 8) & 255u;
      values[k + 2] += (word >> 16) & 255u;
      values[k + 3] += word >> 24;
    }
  }
  uint best = 0xffffffffu;
  for (int k = 0; k < BLOCK; ++k) {
    int d = int(lane) * BLOCK + k;
    best = min(best,mix(0xffffffffu,(values[k] << 16) | uint(d),d <= x));
  }
  best = group_min(best);
  int winner = int(best & 65535u);
  uint minimum = best >> 16;
  uint second = 0xffffffffu;
  uint lower = 0, upper = 0;
  int lower_d = max(winner - 1,0), upper_d = min(winner + 1,DISPARITIES - 1);
  for (int k = 0; k < BLOCK; ++k) {
    int d = int(lane) * BLOCK + k;
    uint candidate = mix(0xffffffffu,values[k],abs(d - winner) > 1);
    second = min(second,mix(0xffffffffu,candidate,d <= x));
    // Constant indexing avoids local-memory lowering for a dynamic array read.
    lower = mix(lower,values[k],k == lower_d % BLOCK);
    upper = mix(upper,values[k],k == upper_d % BLOCK);
  }
  second = group_min(second);
  // Every active lane executes the shuffles before the single-lane store.
  lower = subgroupShuffle(lower,base + uint(lower_d / BLOCK));
  upper = subgroupShuffle(upper,base + uint(upper_d / BLOCK));
  bool accepted = float(second) > float(minimum) * UNIQUENESS;
  float lo = float(lower), hi = float(upper);
  float denominator = lo + hi - 2.0 * float(minimum);
  float correction = clamp((lo-hi)/(2.0*max(denominator,1.0)),-0.5,0.5);
  correction = mix(0.0,correction,denominator > 0.0);
  correction = mix(0.0,correction,winner > 0);
  correction = mix(0.0,correction,winner < min(DISPARITIES - 1,x));
  float result = float(winner) + correction;
  if (lane == 0) {
    left_disparity[pixel] = mix(-1.0,result,accepted);
    winners[pixel] = mix(0xffffffffu,best,accepted);
  }
}
