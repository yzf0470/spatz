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

// 44 % util
// main.c: sparsemv_v64b(a_core, b, idx_a_core, result_core, m_core, sparsemv_l.N/2);
void sparsemv_m2_v64b(double *a, double* b, short *idx_a, double* c, int M, int N) {
  unsigned int vl, vl_idx, vl_compressed;
  unsigned int avl = N;
  double  *a_ = a     + (M-2) * N;
  double  *a2_;
  short *idx_a_ = idx_a + (M-2) * N/8;
  short *idx_a2_;
  double  *b_ = b;
  double  *c_ = c;

  asm volatile("vmv.s.x v2, zero");
  asm volatile("vmv.s.x v6, zero");

  for (int r=0; r < M/2; r++) {
    // Stripmine and accumulate a partial reduced vector
    do {
      a2_ = a_ + N;
      idx_a2_ = idx_a_ + N/8;

      // Set the vl
      asm volatile("vsetvli %0, %1, e64, m2, ta, ma" : "=r"(vl) : "r"(2*avl));
      asm volatile("vle64.v v0, (%0)" ::"r"(b_));
      b_ += vl;

      asm volatile("vsetvli %0, %1, e16, m2, ta, ma" : "=r"(vl_idx) : "r"(vl/16));
      asm volatile("vle16.v v4, (%0)" ::"r"(idx_a_)); // idx1
      idx_a_ += vl_idx;
      
      asm volatile("vsetvli %0, %1, e64, m2, ta, ma" : "=r"(vl) : "r"(vl));
      asm volatile("vrgather.vv v8, v0, v4");

      asm volatile("vsetvli %0, %1, e64, m2, ta, ma" : "=r"(vl_compressed) : "r"(vl/2));
      asm volatile("vle64.v v12, (%0)" ::"r"(a_));
      a_ += vl_compressed;

      // Multiply and accumulate
      if (avl == N) {
        asm volatile("vfmul.vv v16, v12, v8");
      } else {
        asm volatile("vfmacc.vv v16, v12, v8");
      }

      asm volatile("vsetvli %0, %1, e16, m2, ta, ma" : "=r"(vl_idx) : "r"(vl_idx));
      asm volatile("vle16.v v20, (%0)" ::"r"(idx_a2_)); // idx1
      idx_a2_ += vl_idx;
      
      asm volatile("vsetvli %0, %1, e64, m2, ta, ma" : "=r"(vl) : "r"(vl));
      asm volatile("vrgather.vv v24, v0, v20");

      asm volatile("vsetvli %0, %1, e64, m2, ta, ma" : "=r"(vl_compressed) : "r"(vl_compressed));
      asm volatile("vle64.v v28, (%0)" ::"r"(a2_));
      a2_ += vl_compressed;

      // Multiply and accumulate
      if (avl == N) {
        asm volatile("vfmul.vv v30, v24, v0");
      } else {
        asm volatile("vfmacc.vv v30, v24, v0");
      }

      avl -= vl_compressed;

    } while (avl > 0);

    // Reduce and return
    asm volatile("vfredusum.vs v2, v30, v2");
    asm volatile("vfslide1up.vf v6, v2, %0" ::"f"(0.0));
    asm volatile("vfredusum.vs v6, v16, v6");
    asm volatile("vfslide1up.vf v2, v6, %0" ::"f"(0.0));
    b_ = b;
    a_ -= (3*N);
    idx_a_ -= 3*N/8;
    avl = N;
  }
  
  asm volatile("vsetvli %0, %1, e64, m2, ta, ma" : "=r"(vl) : "r"(M));
  asm volatile("vse64.v v6,  (%0)" ::"r"(c));

}

// 29 % util
// main.c: sparsemv_v64b(a_core, b, idx_a_core, result_core, m_core, sparsemv_l.N/2);
//
void sparsemv_m4_v64b(double *a, double* b, int *idx_a, double* c, int M, int N) {
  unsigned int vl, vl_idx, vl_compressed;
  int flag = 1;
  unsigned int avl = N;
  double  *a_ = a     + (M-1) * N;
  int *idx_a_ = idx_a + (M-1) * N/16;
  double  *b_ = b;

  asm volatile("vmv.s.x v20, zero");
  asm volatile("vmv.s.x v24, zero");

  for (int r=0; r < M; r++) {

    // Stripmine and accumulate a partial reduced vector
    do {
      // Set the vl
      asm volatile("vsetvli %0, %1, e64, m4, ta, ma" : "=r"(vl) : "r"(2*avl));
      asm volatile("vle64.v v0, (%0)" ::"r"(b_));
      b_ += vl;

      asm volatile("vsetvli %0, %1, e32, m4, ta, ma" : "=r"(vl_idx) : "r"(vl/32));
      asm volatile("vle32.v v4, (%0)" ::"r"(idx_a_)); // idx1
      idx_a_ += vl_idx;
      
      asm volatile("vsetvli %0, %1, e64, m4, ta, ma" : "=r"(vl) : "r"(vl));
      asm volatile("vrgather.vv v8, v0, v4");

      asm volatile("vsetvli %0, %1, e64, m4, ta, ma" : "=r"(vl_compressed) : "r"(vl/2));
      asm volatile("vle64.v v12, (%0)" ::"r"(a_));
      a_ += vl_compressed;

      // Multiply and accumulate
      if (avl == N) {
        asm volatile("vfmul.vv v16, v12, v8");
      } else {
        asm volatile("vfmacc.vv v16, v12, v8");
      }

      avl -= vl_compressed;

    } while (avl > 0);

    if (flag) {
    // Reduce and return
      asm volatile("vfredusum.vs v20, v16, v20");
      asm volatile("vfslide1up.vf v24, v20, %0" ::"f"(0.0));
      flag = 0;
    } else {
      asm volatile("vfredusum.vs v24, v16, v24");
      asm volatile("vfslide1up.vf v20, v24, %0" ::"f"(0.0));
      flag = 1; 
    }
    b_      = b  ;
    a_     -= 2*N;
    idx_a_ -= N/8;
    avl = N;
  }
  
  asm volatile("vsetvli %0, %1, e64, m4, ta, ma" : "=r"(vl) : "r"(M));
  if (flag) {
    asm volatile("vse64.v v24, (%0)" ::"r"(c));
  } else {
    asm volatile("vse64.v v20, (%0)" ::"r"(c));
  }
}

void sparsemv_m4_v64b_v2(double *a, double* b, int *idx_a, double* c, int M, int N) {
  unsigned int vl, vl_idx, vl_compressed;
  unsigned int avl = N;
  double  *a_ = a     + (M-2) * N;
  double  *a2_;
  int *idx_a_ = idx_a + (M-2) * N/16;
  int *idx_a2_;
  double  *b_ = b;
  double  *c_ = c;

  for (int r=0; r < M/2; r++) {
    // Stripmine and accumulate a partial reduced vector
    do {
      a2_ = a_ + N;
      idx_a2_ = idx_a_ + N/16;

      // Set the vl
      asm volatile("vsetvli %0, %1, e64, m4, ta, ma" : "=r"(vl) : "r"(2*avl));
      asm volatile("vle64.v v0, (%0)" ::"r"(b_));
      b_ += vl;

      asm volatile("vsetvli %0, %1, e32, m4, ta, ma" : "=r"(vl_idx) : "r"(vl/32));
      asm volatile("vle32.v v4, (%0)" ::"r"(idx_a_)); // idx1
      idx_a_ += vl_idx;
      
      asm volatile("vsetvli %0, %1, e64, m4, ta, ma" : "=r"(vl) : "r"(vl));
      asm volatile("vrgather.vv v8, v0, v4");

      asm volatile("vsetvli %0, %1, e64, m4, ta, ma" : "=r"(vl_compressed) : "r"(vl/2));
      asm volatile("vle64.v v12, (%0)" ::"r"(a_));
      a_ += vl_compressed;

      // Multiply and accumulate
      if (avl == N) {
        asm volatile("vfmul.vv v16, v12, v8");
      } else {
        asm volatile("vfmacc.vv v16, v12, v8");
      }

      asm volatile("vsetvli %0, %1, e32, m4, ta, ma" : "=r"(vl_idx) : "r"(vl_idx));
      asm volatile("vle32.v v20, (%0)" ::"r"(idx_a2_)); // idx1
      idx_a2_ += vl_idx;
      
      asm volatile("vsetvli %0, %1, e64, m4, ta, ma" : "=r"(vl) : "r"(vl));
      asm volatile("vrgather.vv v24, v0, v20");

      asm volatile("vsetvli %0, %1, e64, m4, ta, ma" : "=r"(vl_compressed) : "r"(vl_compressed));
      asm volatile("vle64.v v28, (%0)" ::"r"(a2_));
      a2_ += vl_compressed;

      // Multiply and accumulate
      if (avl == N) {
        asm volatile("vfmul.vv v4, v24, v0");
      } else {
        asm volatile("vfmacc.vv v4, v24, v0");
      }

      avl -= vl_compressed;

    } while (avl > 0);

    asm volatile("vmv.s.x v20, zero");
    asm volatile("vmv.s.x v24, zero");

    // Reduce and return
    asm volatile("vfredusum.vs v20, v4, v20");
    asm volatile("vfslide1up.vf v24, v20, %0" ::"f"(0.0));
    asm volatile("vfredusum.vs v24, v16, v24");
    asm volatile("vfslide1up.vf v20, v24, %0" ::"f"(0.0));
    b_ = b;
    a_ -= (3*N);
    idx_a_ -= 3*N/16;
    avl = N;
  }
  
  asm volatile("vsetvli %0, %1, e64, m4, ta, ma" : "=r"(vl) : "r"(M));
  asm volatile("vse64.v v20,  (%0)" ::"r"(c));

}
