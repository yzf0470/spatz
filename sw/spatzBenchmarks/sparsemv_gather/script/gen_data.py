#!/usr/bin/env python3
# Copyright 2022 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# Author: Matheus Cavalcante <matheusd@iis.ee.ethz.ch>

import numpy as np
import torch
import argparse
import pathlib
import hjson
import functools 
import operator

np.random.seed(42)
torch.manual_seed(42)

global verbose


def array_to_cstr(a, fmt=float):
    out = "{"
    if fmt == float:
        if isinstance(a, np.ndarray):
            a = a.flat
        if isinstance(a, torch.Tensor):
            a = a.numpy().flat
        for el in a:
            out += "{}, ".format(el)
    else:
        for sign, exp, mant in zip(
            a["sign"].numpy().flat,
            a["exponent"].numpy().flat,
            a["mantissa"].numpy().flat,
        ):
            value = sign * 2**7 + exp * 2**2 + mant
            out += "0x{:02x}, ".format(value)
    out = out[:-2] + "}"
    return out


def emit_header_file(layer_type: str, **kwargs):

    file_path = pathlib.Path(__file__).parent.parent / "data"
    emit_str = (
        "// Copyright 2025 ETH Zurich and University of Bologna.\n"
        + "// Licensed under the Apache License, Version 2.0, see LICENSE for details.\n"
        + "// SPDX-License-Identifier: Apache-2.0\n\n"
        + "// This file was generated automatically.\n\n"
    )

    file = file_path / ("data_" + str(kwargs["M"]) + "_"  + str(kwargs["N"]) + "_"  + str(kwargs["prec"]) + ".h")
    emit_str += emit_sparsemv_layer(**kwargs)
    with file.open("w") as f:
        f.write(emit_str)

# kwargs = {
#         "A": mat_A,
#         "B": vec_B,
#         "compressed_A": compressed_mat_A,
#         "idx_mat_A": idx_mat_A,
#         "idx_vec_B": idx_vec_B,
#         "packed_idx_mat_A_origin": packed_idx_mat_A_origin,
#         "packed_idx_mat_A": packed_idx_mat_A,
#         "packed_idx_vec_B": packed_idx_vec_B,
#         "result": result,
#         "M": param["M"],
#         "N": param["N"],
#         "prec": param["prec"],
#         "expand": param["expand"],
#         "bits_A": bits_A
#     }

def emit_sparsemv_layer(name="sparsemv", **kwargs):
    mat_A = kwargs["A"]
    vec_B = kwargs["B"]
    compressed_mat_A_origin = kwargs["compressed_A_origin"]
    compressed_mat_A = kwargs["compressed_A"]
    idx_mat_A = kwargs["idx_mat_A"]
    idx_vec_B = kwargs["idx_vec_B"]
    packed_idx_mat_A_origin = kwargs["packed_idx_mat_A_origin"]
    packed_idx_mat_A = kwargs["packed_idx_mat_A"]
    packed_idx_vec_B = kwargs["packed_idx_vec_B"]
    result = kwargs["result"]

    m = kwargs["M"]
    n = kwargs["N"]

    layer_str = ""
    layer_str += '#include "layer.h"\n\n'
    layer_str += f"sparsemv_layer {name}_l = {{\n"
    layer_str += f"\t.M = {m},\n"
    layer_str += f"\t.N = {n},\n"
    layer_str += f'\t.dtype = FP{kwargs["prec"]},\n'
    layer_str += "};\n\n\n"

    ctypes = {"64": "double", "32": "float", "16": "_Float16", "8": "char"}

    dtype = ctypes[str(kwargs["prec"])]
    if dtype != "char":
        layer_str += (
            f'static {dtype} {name}_B_dram [{n}] __attribute__((section(".data"))) = '
            + array_to_cstr(vec_B)
            + ";\n\n\n"
        )
        layer_str += (
            f'static {dtype} {name}_compressed_A_dram_origin [{m} * {n//2}] __attribute__((section(".data"))) = '
            + array_to_cstr(compressed_mat_A_origin)
            + ";\n\n\n"
        )
        layer_str += (
            f'static {dtype} {name}_compressed_A_dram [{m} * {n//2}] __attribute__((section(".data"))) = '
            + array_to_cstr(compressed_mat_A)
            + ";\n\n\n"
        )
        layer_str += (
            f'static {"int"} {name}_packed_idx_A_dram [{m} * {n//32}] __attribute__((section(".data"))) = '
            + array_to_cstr(packed_idx_mat_A)
            + ";\n\n\n"
        )
        layer_str += (
            f'static {dtype} {name}_result [{m}] __attribute__((section(".data"))) = '
            + array_to_cstr(result)
            + ";\n\n\n"
        )
    else:
        layer_str += (
            f"static {dtype} {name}_A_dram [{m} * {n}] = "
            + array_to_cstr(kwargs["bits_A"], fmt="char")
            + ";\n\n\n"
        )
        layer_str += (
            f"static {dtype} {name}_B_dram [{n}] = "
            + array_to_cstr(kwargs["bits_B"], fmt="char")
            + ";\n\n\n"
        )
        layer_str += (
            f"static {dtype} {name}_result [{m}] = "
            + array_to_cstr(kwargs["result"], fmt="char")
            + ";\n\n\n"
        )

    return layer_str

def rand_data_generator(shape, prec, alt=False):
    total_elems = functools.reduce(operator.mul, shape, 1)
    rand_int = torch.randint(
        low=1,
        high=8,
        size=(total_elems, 1),
        dtype=torch.int64,
        requires_grad=False
    )
    if prec == 64:
        return rand_int.to(torch.float64), {}
    elif prec == 32:
        return rand_int.to(torch.float32), {}
    elif prec == 16:
        if alt:
            return torch.randn(shape, requires_grad=False, dtype=torch.bfloat16), {}
        else:
            return torch.randn(shape, requires_grad=False, dtype=torch.float16), {}
    elif prec == 8:
        sign = torch.randint(
            0, 2, shape, requires_grad=False, dtype=torch.uint8
        )  # -1 or 1
        exponent = torch.randint(
            0, 16, shape, requires_grad=False, dtype=torch.uint8
        )  # < 0b01111
        mantissa = torch.randint(
            0, 4, shape, requires_grad=False, dtype=torch.uint8
        )  # can be arbitrary
        bits = {"sign": sign, "exponent": exponent, "mantissa": mantissa}
        # TODO: not actually correct
        return ((-1.0) ** sign.double()) * (2.0 ** (exponent.double() - 15.0)) * (
            1.0 + mantissa.double() / (2**2)
        ), bits

def sparse_rand_data_generator(effective_ele_num, unit_len, shape, prec, alt=False): #generate sparse vector and its index
    total_elems = functools.reduce(operator.mul, shape, 1)
    if total_elems % unit_len != 0:
        raise ValueError("shape must be mutiple of unit_len")
    if effective_ele_num > unit_len:
        raise ValueError("effective_ele_num should not be more than unit_len")
    
    num_groups = total_elems // unit_len
    mask = torch.zeros((num_groups, unit_len), dtype=torch.uint8)
    result_idx_list = []

    for i in range(num_groups):
        indices = torch.randperm(unit_len)[:effective_ele_num]
        indices, _ = indices.sort()
        mask[i, indices] = 1
        result_idx_list.append(indices)

    mask = mask.view(-1,1).to(torch.float32)
    rand_vals = rand_data_generator(shape, prec, alt)[0]
    result = mask*rand_vals 

    result_idx = torch.cat(result_idx_list, dim=0).reshape(len(result_idx_list)*effective_ele_num, 1)
    result_idx_flat = torch.cat(result_idx_list, dim=0).view(-1)
    chunks = result_idx_flat.view(-1, 16).to(torch.int32)
    shifts = torch.arange(16, dtype=torch.int32, device=chunks.device) * 2
    result_idx_packed = ((chunks & 0x3) << shifts).sum(dim=1)
    result_idx_packed = result_idx_packed.view(-1, 1)

    return result, result_idx, result_idx_packed, {}

def reorder_result_idx_packed(result_idx_packed: torch.Tensor, M:int, N:int) -> torch.Tensor:
    flat = result_idx_packed.view(-1)
    units = flat.view(M//4, 4, N)
    reordered_units = []
    for unit in units:
        reordered_units.append(unit.transpose(0,1).reshape(-1))
    new_flat = torch.cat(reordered_units, dim=0)

    return new_flat.view(-1,1)

def reorder_result(result: torch.Tensor, M:int, N:int) -> torch.Tensor:
    flat = result.view(-1)
    units = flat.view(M//4, 4, N//16, 16)
    reordered_units = []
    for unit in units:
        block = unit.permute(1,0,2).reshape(-1)
        reordered_units.append(block)
    new_flat = torch.cat(reordered_units, dim=0)

    return new_flat.view(-1,1)

def sparse_rand_data_generator_int16(effective_ele_num, unit_len, shape, prec, alt=False): #generate sparse vector and its index
    total_elems = functools.reduce(operator.mul, shape, 1)
    if total_elems % unit_len != 0:
        raise ValueError("shape must be mutiple of unit_len")
    if effective_ele_num > unit_len:
        raise ValueError("effective_ele_num should not be more than unit_len")
    
    num_groups = total_elems // unit_len
    mask = torch.zeros((num_groups, unit_len), dtype=torch.uint8)
    result_idx_list = []

    for i in range(num_groups):
        indices = torch.randperm(unit_len)[:effective_ele_num]
        indices, _ = indices.sort()
        mask[i, indices] = 1
        result_idx_list.append(indices)

    mask = mask.view(-1,1).to(torch.float32)
    rand_vals = rand_data_generator(shape, prec, alt)[0]
    result = mask*rand_vals 

    result_idx = torch.cat(result_idx_list, dim=0).reshape(len(result_idx_list)*effective_ele_num, 1)
    result_idx_flat = torch.cat(result_idx_list, dim=0).view(-1)
    chunks = result_idx_flat.view(-1, 8).to(torch.int32)
    shifts = torch.arange(8, dtype=torch.int32, device=chunks.device) * 2
    packed = ((chunks & 0x3) << shifts).sum(dim=1)
    result_idx_packed = packed.to(torch.int16).view(-1, 1)

    return result, result_idx, result_idx_packed, {}

def extract_nonzero(result: torch.Tensor) -> torch.Tensor:
    flat = result.view(-1)
    nonzero_vals = flat[flat != 0]
    return nonzero_vals.view(-1,1)

def gemv(a, b, shape):
    print(a.shape , b.shape)
    return torch.matmul(a.reshape(shape), b)

def main():

    parser = argparse.ArgumentParser(description="Generate data for kernels")
    parser.add_argument(
        "-c",
        "--cfg",
        type=pathlib.Path,
        required=True,
        help="Select param config file kernel",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Set verbose")

    args = parser.parse_args()

    global verbose
    verbose = args.verbose

    with args.cfg.open() as f:
        param = hjson.loads(f.read())

    mat_A, idx_mat_A, packed_idx_mat_A_origin, bits_A = sparse_rand_data_generator(2, 4, (param["M"], param["N"]), param["prec"])
    packed_idx_mat_A = reorder_result_idx_packed(packed_idx_mat_A_origin, param["M"], param["N"]//32)
    vec_B, idx_vec_B, packed_idx_vec_B         = mat_A[0:param["N"]], idx_mat_A[0:(param["N"])//2], packed_idx_mat_A[0:(param["N"])//32]
    compressed_mat_A_origin                    = extract_nonzero(mat_A)
    compressed_mat_A                           = reorder_result(compressed_mat_A_origin, param["M"], param["N"]//2)
    result = gemv(mat_A, vec_B, (param["M"], param["N"]))

    kwargs = {
        "A": mat_A,
        "B": vec_B,
        "compressed_A_origin": compressed_mat_A_origin,
        "compressed_A": compressed_mat_A,
        "idx_mat_A": idx_mat_A,
        "idx_vec_B": idx_vec_B,
        "packed_idx_mat_A_origin": packed_idx_mat_A_origin,
        "packed_idx_mat_A": packed_idx_mat_A,
        "packed_idx_vec_B": packed_idx_vec_B,
        "result": result,
        "M": param["M"],
        "N": param["N"],
        "prec": param["prec"],
        "expand": param["expand"],
        "bits_A": bits_A
    }

    emit_header_file("sparsemv", **kwargs)


if __name__ == "__main__":
    main()
