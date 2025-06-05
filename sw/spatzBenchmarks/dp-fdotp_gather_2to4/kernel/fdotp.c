// Copyright 2022 ETH Zurich and University of Bologna.
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

// Author: Matteo Perotti <mperotti@iis.ee.ethz.ch>

#include "fdotp.h"

// 64-bit dot-product: a * b
double fdotp_v64b(const double *a, const double *b, const int *idx_a, unsigned int avl) {
  const unsigned int orig_avl = avl;
  unsigned int vl_idx, vl_dense, vl_compressed;
  double red;

  // Clean the accumulator
  asm volatile("vmv.s.x v0, zero");

  // Stripmine and accumulate a partial reduced vector
  do {

    // Set the vl, load chunk a, b and idx_a
    asm volatile("vsetvli %0, %1, e64, m4, ta, ma" : "=r"(vl_dense) : "r"(2*avl));
    asm volatile("vle64.v v8, (%0)" ::"r"(b)); // src2, originally dense

    asm volatile("vsetvli %0, %1, e32, m4, ta, ma" : "=r"(vl_idx) : "r"(vl_dense/32));
    asm volatile("vle32.v v12, (%0)" ::"r"(idx_a)); // idx of src1
    
    asm volatile("vsetvli %0, %1, e64, m4, ta, ma" : "=r"(vl_compressed) : "r"(vl_dense));
    asm volatile("vle64.v v4, (%0)" ::"r"(a)); // src1, compressed

    // Gather b 
    asm volatile("vrgather.vv v16, v8, v12");
    
    // Multiply and accumulate
    if (avl == orig_avl) {
      asm volatile("vfmul.vv v20, v16, v4");
    } else {
      asm volatile("vfmacc.vv v20, v16, v4");
    }
    
    // Bump pointers
    a += vl_dense/2;
    b += vl_dense;
    idx_a += vl_idx;
    avl -= vl_dense/2;
  } while (avl > 0);

  // Reduce and return
  asm volatile("vfredusum.vs v0, v20, v0");
  asm volatile("vfmv.f.s %0, v0" : "=f"(red));

  return red;
}

double fdotp_slide_v64b(const double *a, const double *b, const int *idx_a, unsigned int avl) {
  const unsigned int orig_avl = avl;
  unsigned int vl_idx, vl_dense, vl_compressed;
  double red;
  unsigned int idx_counter = 0;

  // Clean the accumulator
  asm volatile("vmv.s.x v0, zero");

  // Stripmine and accumulate a partial reduced vector
  do {
    // Set the vl, load chunk a, b and idx_a
    asm volatile("vsetvli %0, %1, e64, m4, ta, ma" : "=r"(vl_dense) : "r"(2*avl));
    asm volatile("vle64.v v8, (%0)" ::"r"(b)); // src2, originally dense

    if (idx_counter == 0) {
      asm volatile("vsetvli %0, %1, e32, m4, ta, ma" : "=r"(vl_idx) : "r"(avl/16));
      asm volatile("vle32.v v12, (%0)" ::"r"(idx_a)); // idx of src1
      idx_counter = vl_idx;
      idx_a += vl_idx;
    } else { // slide
      asm volatile("vsetvli %0, %1, e32, m4, ta, ma" : "=r"(vl_idx) : "r"(vl_idx));
      asm volatile("vfslide1down.vf v12, v12, %0" ::"f"(0.0));
    }
    
    asm volatile("vsetvli %0, %1, e64, m4, ta, ma" : "=r"(vl_compressed) : "r"(vl_dense));

    // Gather b 
    asm volatile("vrgather.vv v16, v8, v12");
    idx_counter -= 1;

    // Gather a
    asm volatile("vle64.v v4, (%0)" ::"r"(a)); // src1, compressed
    
    // Multiply and accumulate
    if (avl == orig_avl) {
      asm volatile("vfmul.vv v20, v16, v4");
    } else {
      asm volatile("vfmacc.vv v20, v16, v4");
    }
    
    // Bump pointers
    a += vl_dense/2;
    b += vl_dense;
    avl -= vl_dense/2;
  } while (avl > 0);

  // Reduce and return
  asm volatile("vfredusum.vs v0, v20, v0");
  asm volatile("vfmv.f.s %0, v0" : "=f"(red));

  return red;
}
