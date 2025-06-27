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
        asm volatile("vfmul.vv v24, v12, v8");
      } else {
        asm volatile("vfmacc.vv v24, v12, v8");
      }

      asm volatile("vsetvli %0, %1, e32, m4, ta, ma" : "=r"(vl_idx) : "r"(vl_idx));
      asm volatile("vle32.v v16, (%0)" ::"r"(idx_a2_)); // idx1
      idx_a2_ += vl_idx;
      
      asm volatile("vsetvli %0, %1, e64, m4, ta, ma" : "=r"(vl) : "r"(vl));
      asm volatile("vrgather.vv v20, v0, v16"); //??????????!!

      asm volatile("vsetvli %0, %1, e64, m4, ta, ma" : "=r"(vl_compressed) : "r"(vl_compressed));
      asm volatile("vle64.v v12, (%0)" ::"r"(a2_));
      a2_ += vl_compressed;

      // Multiply and accumulate
      if (avl == N) {
        asm volatile("vfmul.vv v28, v12, v20");
      } else {
        asm volatile("vfmacc.vv v28, v12, v20");
      }

      avl -= vl_compressed;

    } while (avl > 0);
    
    asm volatile("vfmv.s.f v4, %0" ::"f"(0.0));
    asm volatile("vfmv.s.f v16, %0" ::"f"(0.0));

    // Reduce and return
    asm volatile("vfredusum.vs v4, v28, v4");
    asm volatile("vfslide1up.vf v16, v4, %0" ::"f"(0.0));
    asm volatile("vfredusum.vs v16, v24, v16");
    asm volatile("vfslide1up.vf v4, v16, %0" ::"f"(0.0));
    b_ = b;
    a_ -= (3*N);
    idx_a_ -= 3*N/16;
    avl = N;
  }
  
  asm volatile("vsetvli %0, %1, e64, m4, ta, ma" : "=r"(vl) : "r"(M));
  asm volatile("vse64.v v16,  (%0)" ::"r"(c));
}

void sparsemv_m4_v64b_diyou(double *a, double* b, int *idx_a, double* c, int M, int N) {
  unsigned int vl, vl_idx, vl_compressed;
  unsigned int avl = N;
  double  *a_ = a;
  double  *a2_, *a3_, *a4_;
  int *idx_a_ = idx_a;
  int *idx_a2_, *idx_a3_, *idx_a4_;
  double  *b_ = b;
  double *result = c; 

  for (int r=0; r < M/4; r++) {
    // Stripmine and accumulate a partial reduced vector
    do {
      a2_ = a_ + N;
      a3_ = a2_ + N;
      a4_ = a3_ + N;
      idx_a2_ = idx_a_ + N/16;
      idx_a3_ = idx_a2_ + N/16;
      idx_a4_ = idx_a3_ + N/16;

      // Set the vl
      asm volatile("vsetvli %0, %1, e64, m4, ta, ma" : "=r"(vl) : "r"(2*avl));
      asm volatile("vle64.v v0, (%0)" ::"r"(b_));
      b_ += vl;

      asm volatile("vsetvli %0, %1, e32, m4, ta, ma" : "=r"(vl_idx) : "r"(vl/32));
      asm volatile("vle32.v v4, (%0)" ::"r"(idx_a_));
      idx_a_ += vl_idx;

      asm volatile("vsetvli %0, %1, e64, m4, ta, ma" : "=r"(vl) : "r"(vl));
      asm volatile("vrgather.vv v12, v0, v4");

      asm volatile("vsetvli %0, %1, e64, m4, ta, ma" : "=r"(vl_compressed) : "r"(vl/2));
      asm volatile("vle64.v v8, (%0)" ::"r"(a_));
      a_ += vl_compressed;

      // Multiply and accumulate
      if (avl == N) {
        asm volatile("vfmul.vv v16, v12, v8");
      } else {
        asm volatile("vfmacc.vv v16, v12, v8");
      }

      asm volatile("vsetvli %0, %1, e32, m4, ta, ma" : "=r"(vl_idx) : "r"(vl_idx));
      asm volatile("vle32.v v4, (%0)" ::"r"(idx_a2_));
      idx_a2_ += vl_idx;

      asm volatile("vsetvli %0, %1, e64, m4, ta, ma" : "=r"(vl) : "r"(vl));
      asm volatile("vrgather.vv v12, v0, v4");

      asm volatile("vsetvli %0, %1, e64, m4, ta, ma" : "=r"(vl_compressed) : "r"(vl_compressed));
      asm volatile("vle64.v v8, (%0)" ::"r"(a2_));
      a2_ += vl_compressed;

      // Multiply and accumulate
      if (avl == N) {
        asm volatile("vfmul.vv v20, v12, v8");
      } else {
        asm volatile("vfmacc.vv v20, v12, v8");
      }

      asm volatile("vsetvli %0, %1, e32, m4, ta, ma" : "=r"(vl_idx) : "r"(vl_idx));
      asm volatile("vle32.v v4, (%0)" ::"r"(idx_a3_));
      idx_a3_ += vl_idx;

      asm volatile("vsetvli %0, %1, e64, m4, ta, ma" : "=r"(vl) : "r"(vl));
      asm volatile("vrgather.vv v12, v0, v4");

      asm volatile("vsetvli %0, %1, e64, m4, ta, ma" : "=r"(vl_compressed) : "r"(vl_compressed));
      asm volatile("vle64.v v8, (%0)" ::"r"(a3_));
      a3_ += vl_compressed;

      // Multiply and accumulate
      if (avl == N) {
        asm volatile("vfmul.vv v24, v12, v8");
      } else {
        asm volatile("vfmacc.vv v24, v12, v8");
      }

      asm volatile("vsetvli %0, %1, e32, m4, ta, ma" : "=r"(vl_idx) : "r"(vl_idx));
      asm volatile("vle32.v v4, (%0)" ::"r"(idx_a4_));
      idx_a4_ += vl_idx;

      asm volatile("vsetvli %0, %1, e64, m4, ta, ma" : "=r"(vl) : "r"(vl));
      asm volatile("vrgather.vv v12, v0, v4");

      asm volatile("vsetvli %0, %1, e64, m4, ta, ma" : "=r"(vl_compressed) : "r"(vl_compressed));
      asm volatile("vle64.v v8, (%0)" ::"r"(a4_));
      a4_ += vl_compressed;

      // Multiply and accumulate
      if (avl == N) {
        asm volatile("vfmul.vv v28, v12, v8");
      } else {
        asm volatile("vfmacc.vv v28, v12, v8");
      }

      avl -= vl_compressed;

    } while (avl > 0);
    
    asm volatile("vfmv.s.f v0, %0" ::"f"(0.0));
    asm volatile("vfmv.s.f v4, %0" ::"f"(0.0));
    asm volatile("vfmv.s.f v8, %0" ::"f"(0.0));
    asm volatile("vfmv.s.f v12, %0" ::"f"(0.0));

    // Reduce and return
    asm volatile("vfredusum.vs v0, v16, v0");
    asm volatile("vfmv.f.s %0, v0" : "=f"(*result));
    result += 1;
    asm volatile("vfredusum.vs v4, v20, v4");
    asm volatile("vfmv.f.s %0, v4" : "=f"(*result));
    result += 1;
    asm volatile("vfredusum.vs v8, v24, v8");
    asm volatile("vfmv.f.s %0, v8" : "=f"(*result));
    result += 1;
    asm volatile("vfredusum.vs v12, v28, v12");
    asm volatile("vfmv.f.s %0, v12" : "=f"(*result));
    result += 1;

    b_ = b;
    a_ += (3*N);
    idx_a_ += 3*N/16;
    avl = N;
  }
}
