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
from functools import reduce

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


def emit_header_file(name, **kwargs):

    file_path = pathlib.Path(__file__).parent.parent / "data"
    emit_str = (
        "// Copyright 2023 ETH Zurich and University of Bologna.\n"
        + "// Licensed under the Apache License, Version 2.0, see LICENSE for details.\n"
        + "// SPDX-License-Identifier: Apache-2.0\n\n"
        + "// This file was generated automatically.\n\n"
    )

    file = file_path / ("data_" + str(kwargs["M"]) + ".h")
    emit_str += emit_dotp_layer(name, **kwargs)
    with file.open("w") as f:
        f.write(emit_str)

def emit_dotp_layer(name = "sparse", **kwargs):
    vec_A = kwargs["A"]
    vec_B = kwargs["B"]
    if name == "sparse":
        compressed_vec_A = kwargs["compressed_A"]
        idx_A = kwargs["idx_A"]
        idx_B = kwargs["idx_B"]
        packed_idx_A = kwargs["packed_idx_A"]
        packed_idx_B = kwargs["packed_idx_B"]
    result = kwargs["result"]

    m = kwargs["M"]

    layer_str = ""
    layer_str += '#include "layer.h"\n\n'
    layer_str += f"dotp_layer {name}_l = {{\n"
    layer_str += f"\t.M = {m},\n"
    layer_str += f'\t.dtype = FP{kwargs["prec"]},\n'
    layer_str += "};\n\n\n"

    ctypes = {"64": "double", "32": "float", "16": "__fp16", "8": "char"}

    dtype = ctypes[str(kwargs["prec"])]
    if dtype != "char":
        layer_str += (
            f'static {dtype} {name}_A_dram [{2*m}] __attribute__((section(".data"))) = '
            + array_to_cstr(vec_A)
            + ";\n\n\n"
        )
        layer_str += (
            f'static {dtype} {name}_B_dram [{2*m}] __attribute__((section(".data"))) = '
            + array_to_cstr(vec_B)
            + ";\n\n\n"
        )
        if name == "sparse":
            layer_str += (
                f'static {dtype} compressed_A_dram [{m}] __attribute__((section(".data"))) = '
                + array_to_cstr(compressed_vec_A)
                + ";\n\n\n"
            )
            layer_str += (
                f'static {"int"} idx_A_dram [{m}] __attribute__((section(".data"))) = '
                + array_to_cstr(idx_A)
                + ";\n\n\n"
            )
            layer_str += (
                f'static {"int"} idx_B_dram [{m}] __attribute__((section(".data"))) = '
                + array_to_cstr(idx_B)
                + ";\n\n\n"
            )
            layer_str += (
                f'static {"int"} packed_idx_A_dram [{int(m/16)}] __attribute__((section(".data"))) = '
                + array_to_cstr(packed_idx_A)
                + ";\n\n\n"
            )
            layer_str += (
                f'static {"int"} packed_idx_B_dram [{int(m/16)}] __attribute__((section(".data"))) = '
                + array_to_cstr(packed_idx_B)
                + ";\n\n\n"
            )
        layer_str += (
            f'static {dtype} {name}_result __attribute__((section(".data"))) = '
            + array_to_cstr(result)
            + ";\n\n\n"
        )
    else:
        layer_str += (
            f"static {dtype} {name}_A_dram [{m}] = "
            + array_to_cstr(kwargs["bits_A"], fmt="char")
            + ";\n\n\n"
        )
        layer_str += (
            f"static {dtype} {name}_B_dram [{m}] = "
            + array_to_cstr(kwargs["bits_B"], fmt="char")
            + ";\n\n\n"
        )
        layer_str += (
            f"static {dtype} {name}_result = "
            + array_to_cstr(kwargs["result"], fmt="char")
            + ";\n\n\n"
        )

    return layer_str

def rand_data_generator(shape, prec, alt=False):
    rand_int = torch.randint(
        low=1,
        high=8,
        size=shape,
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
    if shape[0] % unit_len != 0:
        raise ValueError("shape must be mutiple of unit_len")
    if effective_ele_num > unit_len:
        raise ValueError("effective_ele_num should not be more than unit_len")
    num_groups = shape[0] // unit_len
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

def extract_nonzero(result: torch.Tensor) -> torch.Tensor:
    flat = result.view(-1)
    nonzero_vals = flat[flat != 0]
    return nonzero_vals.view(-1,1)

def dotp(a, b):
    return reduce(lambda a, b: a + b, np.multiply(a, b))

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

    vec_A, idx_A, packed_idx_A, bits_A = sparse_rand_data_generator(2, 4, (2*param["M"], 1), param["prec"])
    # vec_A, bits_A = rand_data_generator((param["M"], 1), param["prec"])
    vec_B, idx_B, packed_idx_B, bits_B = sparse_rand_data_generator(2, 4, (2*param["M"], 1), param["prec"])
    # vec_B, bits_B = rand_data_generator((param["M"], 1), param["prec"])
    compressed_vec_A = extract_nonzero(vec_A)
    # gather function
    
    result = dotp(vec_A, vec_A)

    kwargs = {
        "A": vec_A,
        "compressed_A": compressed_vec_A,
        "idx_A": idx_A, 
        "packed_idx_A": packed_idx_A,
        "B": vec_B,
        "idx_B": idx_B,
        "packed_idx_B": packed_idx_B,
        "result": result,
        "M": param["M"],
        "prec": param["prec"],
        "expand": param["expand"],
        "bits_A": bits_A,
        "bits_B": bits_B,
    }

    emit_header_file("sparse", **kwargs)


if __name__ == "__main__":
    main()