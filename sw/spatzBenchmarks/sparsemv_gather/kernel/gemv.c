// Copyright 2025 ETH Zurich and University of Bologna.
//
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Author: Navaneeth Kunhi Purayil, ETH Zurich <nkunhi@iis.ee.ethz.ch>

#include "gemv.h"

// 23.0 % util
void sparsemv_v64b(double *a, double* b, int *idx_a, double* c, int M, int N) {
  unsigned int vl, vl_idx, vl_compressed;
  unsigned int avl = N;
  double  *a_ = a;
  int *idx_a_ = idx_a;
  double  *b_ = b;
  double *result = c; 

  for (int r = 0; r < M/4; r++) {
    // Stripmine and accumulate a partial reduced vector
    do {
      // Load indices
      asm volatile("vsetvli %0, %1, e32, m1, ta, ma" : "=r"(vl_idx) : "r"(1));
      asm volatile("vle32.v v4, (%0)" ::"r"(idx_a_));
      idx_a_ += vl_idx;
      asm volatile("vle32.v v5, (%0)" ::"r"(idx_a_));
      idx_a_ += vl_idx;
      asm volatile("vle32.v v6, (%0)" ::"r"(idx_a_));
      idx_a_ += vl_idx;
      asm volatile("vle32.v v7, (%0)" ::"r"(idx_a_));
      idx_a_ += vl_idx;

      // Load B, A1 & A2, and gather
      asm volatile("vsetvli %0, %1, e64, m4, ta, ma" : "=r"(vl) : "r"(2*avl));
      asm volatile("vle64.v v0, (%0)" ::"r"(b_));
      b_ += vl;
      asm volatile("vle64.v v8, (%0)" ::"r"(a_));
      a_ += vl;
      asm volatile("vrgather.vv v12, v0, v4");
      asm volatile("vrgather.vv v14, v0, v5");

      // Multiply and accumulate
      asm volatile("vsetvli %0, %1, e64, m2, ta, ma" : "=r"(vl_compressed) : "r"(vl/2));
      if (avl == N) {
        asm volatile("vfmul.vv v16, v12, v8");
      } else {
        asm volatile("vfmacc.vv v16, v12, v8");
      }
      if (avl == N) {
        asm volatile("vfmul.vv v18, v14, v10");
      } else {
        asm volatile("vfmacc.vv v18, v14, v10");
      }
      
      // Load A3 & A4, and gather
      asm volatile("vsetvli %0, %1, e64, m4, ta, ma" : "=r"(vl) : "r"(vl));
      asm volatile("vle64.v v20, (%0)" ::"r"(a_));
      a_ += vl;
      asm volatile("vrgather.vv v24, v0, v6");
      asm volatile("vrgather.vv v26, v0, v7");

      // Multiply and accumulate
      asm volatile("vsetvli %0, %1, e64, m2, ta, ma" : "=r"(vl_compressed) : "r"(vl/2));
      if (avl == N) {
        asm volatile("vfmul.vv v28, v24, v20");
      } else {
        asm volatile("vfmacc.vv v28, v24, v20");
      }
      if (avl == N) {
        asm volatile("vfmul.vv v30, v26, v22");
      } else {
        asm volatile("vfmacc.vv v30, v26, v22");
      }

      avl -= vl/2;
    } while (avl > 0);
    
    asm volatile("vfmv.s.f v0, %0" ::"f"(0.0));
    asm volatile("vfmv.s.f v4, %0" ::"f"(0.0));
    asm volatile("vfmv.s.f v8, %0" ::"f"(0.0));
    asm volatile("vfmv.s.f v12, %0" ::"f"(0.0));

    // Reduce and return
    asm volatile("vfredusum.vs v0, v16, v0");
    asm volatile("vfmv.f.s %0, v0" : "=f"(*result));
    result += 1;
    asm volatile("vfredusum.vs v4, v18, v4");
    asm volatile("vfmv.f.s %0, v4" : "=f"(*result));
    result += 1;
    asm volatile("vfredusum.vs v8, v28, v8");
    asm volatile("vfmv.f.s %0, v8" : "=f"(*result));
    result += 1;
    asm volatile("vfredusum.vs v12, v30, v12");
    asm volatile("vfmv.f.s %0, v12" : "=f"(*result));
    result += 1;

    b_ = b;
    avl = N;
  }
}
