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

#include <benchmark.h>
#include <snrt.h>
#include <stdio.h>

#include "kernel/gemv.c"
#include DATAHEADER

#if (PREC == 64)
#define T double
#elif (PREC == 32)
#define T float
#elif (PREC == 16)
#define T _Float16
#else
#define T double
#endif

T *a;
T *b;
short *idx_a;
T *result;

static inline int fp_check(const T a, const T b) {
  const T threshold = 0.001;

  // Absolute value
  T comp = a - b;
  if (comp < 0)
    comp = -comp;

  return comp > threshold;
}

int main() {
  const unsigned int num_cores = snrt_cluster_core_num();
  const unsigned int cid = snrt_cluster_core_idx();
  
  // Reset timer
  unsigned int timer = (unsigned int)-1;
  const unsigned int m_core = sparsemv_l.M / num_cores;

  // Allocate the matrices
  if (cid == 0) {
    a = (T *)snrt_l1alloc(sparsemv_l.M * sparsemv_l.N/2 * sizeof(T));
    b = (T *)snrt_l1alloc(sparsemv_l.N * sizeof(T));
    idx_a  = (short *)snrt_l1alloc(sparsemv_l.M * sparsemv_l.N/16 * sizeof(short));
    result = (T *)snrt_l1alloc(sparsemv_l.M * sizeof(T));
  }

  // Initialize the matrices
  if (cid == 0) {
    snrt_dma_start_1d(a, sparsemv_compressed_A_dram, sparsemv_l.M * sparsemv_l.N/2 * sizeof(T));
    snrt_dma_start_1d(b, sparsemv_B_dram, sparsemv_l.N * sizeof(T));
    snrt_dma_start_1d(idx_a, sparsemv_packed_idx_A_dram, sparsemv_l.M * sparsemv_l.N/16 * sizeof(short));
    snrt_dma_wait_all();
  }

  // Wait for all cores to finish
  snrt_cluster_hw_barrier();
  
  // Calculate internal pointers
  T *a_core = a + sparsemv_l.N/2 * m_core * cid;
  T *idx_a_core = idx_a + sparsemv_l.N/16 * m_core * cid;
  T *result_core = result + m_core * cid;

  // Wait for all cores to finish
  snrt_cluster_hw_barrier();

  // Start dump
  if (cid == 0)
    start_kernel();

  // Start timer
  if (cid == 0)
    timer = benchmark_get_cycle();

  // Calculate gemv
  if (sizeof(T) == 8)
    sparsemv_m2_v64b(a_core, b, idx_a_core, result_core, m_core, sparsemv_l.N/2);
  else if (sizeof(T) == 4)
    sparsemv_v32b(a_core, b, idx_a_core, result_core, m_core, sparsemv_l.N/2);
  else 
    sparsemv_v16b(a_core, b, idx_a_core, result_core, m_core, sparsemv_l.N/2);

  // Wait for all cores to finish
  snrt_cluster_hw_barrier();

  // End dump
  if (cid == 0)
  stop_kernel();

  // End timer and check if new best runtime
  if (cid == 0)
    timer = benchmark_get_cycle() - timer;

  // Check and display results
  if (cid == 0) {
    long unsigned int performance = 1000 * 2 * sparsemv_l.M * sparsemv_l.N / timer;
    long unsigned int utilization =
        performance / (2 * num_cores * SNRT_NFPU_PER_CORE * (8 / sizeof(T)));

    printf("\n----- (%d x %d) x (%d x 1) sparsemv -----\n", sparsemv_l.M, sparsemv_l.N, sparsemv_l.N);
    printf("The execution took %u cycles.\n", timer);
    printf("The performance is %ld OP/1000cycle (%ld%%o utilization).\n",
           performance, utilization);
  }

  if (cid == 0) {
    for (int i = 0; i < sparsemv_l.M; i++) {
      if (fp_check(result[i], sparsemv_result[i])) {
        printf("Error: ID: %i Result = %f, Golden = %f\n", i, result[i], sparsemv_result[i]);
        //return -1;
      }
    }
  }

  // Wait for core 0 to finish displaying results
  snrt_cluster_hw_barrier();
  return 0;
}
