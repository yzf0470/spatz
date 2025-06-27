// Copyright 2023 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Author: Matheus Cavalcante, ETH Zurich
//
// The vector slide unit executes all slide instructions

module spatz_vsldu
  import spatz_pkg::*;
  import rvv_pkg::*;
  import cf_math_pkg::idx_width; (
    input  logic             clk_i,
    input  logic             rst_ni,
    // Spatz request
    input  spatz_req_t       spatz_req_i,
    input  logic             spatz_req_valid_i,
    output logic             spatz_req_ready_o,
    // VSLDU response
    output logic             vsldu_rsp_valid_o,
    output vsldu_rsp_t       vsldu_rsp_o,
    // VRF interface
    output vrf_addr_t        vrf_waddr_o,
    output vrf_data_t        vrf_wdata_o,
    output logic             vrf_we_o,
    output vrf_be_t          vrf_wbe_o,
    input  logic             vrf_wvalid_i,
    
    output spatz_id_t  [2:0] vrf_id_o,
    output vrf_addr_t  [1:0] vrf_raddr_o, // source: 0, index: 1
    output logic       [1:0] vrf_re_o,
    input  vrf_data_t  [1:0] vrf_rdata_i,
    input  logic       [1:0] vrf_rvalid_i
  );

// Include FF
`include "common_cells/registers.svh"

  ///////////////////////
  //  Operation queue  //
  ///////////////////////

  spatz_req_t spatz_req_d;

  spatz_req_t spatz_req;
  logic       spatz_req_valid;
  logic       spatz_req_ready;

  spill_register #(
    .T(spatz_req_t)
  ) i_operation_queue (
    .clk_i  (clk_i                                          ),
    .rst_ni (rst_ni                                         ),
    .data_i (spatz_req_d                                    ),
    .valid_i(spatz_req_valid_i && spatz_req_i.ex_unit == SLD),
    .ready_o(spatz_req_ready_o                              ),
    .data_o (spatz_req                                      ),
    .valid_o(spatz_req_valid                                ),
    .ready_i(spatz_req_ready                                )
  );

  logic [1:0] sparse_unit_effective_element_num;
  logic [2:0] sparse_unit_total_element_num;
  assign sparse_unit_effective_element_num = 2'd2;
  assign sparse_unit_total_element_num     = 3'd4;

  // Convert the vl to number of bytes for all element widths
  always_comb begin: proc_spatz_req
    spatz_req_d = spatz_req_i;
    // spatz_req_d: vl counts by Bytes
    // spatz_req_i: vl counts by elements;
    unique case (spatz_req_i.vtype.vsew)
      EW_8: begin
        spatz_req_d.vl     = spatz_req_i.vl;
        spatz_req_d.vstart = spatz_req_i.vstart;
        if (spatz_req_i.op_sld.vmv && spatz_req_i.op_sld.insert)
          spatz_req_d.rs1 = MAXEW == EW_32 ? {4{spatz_req_i.rs1[7:0]}} : {8{spatz_req_i.rs1[7:0]}};
      end
      EW_16: begin
        spatz_req_d.vl     = spatz_req_i.vl << 1;
        spatz_req_d.vstart = spatz_req_i.vstart << 1;
        if (spatz_req_i.op_sld.vmv && spatz_req_i.op_sld.insert)
          spatz_req_d.rs1 = MAXEW == EW_32 ? {2{spatz_req_i.rs1[15:0]}} : {4{spatz_req_i.rs1[15:0]}};
      end
      EW_32: begin
        spatz_req_d.vl     = spatz_req_i.vl << 2;
        spatz_req_d.vstart = spatz_req_i.vstart << 2;
        if (spatz_req_i.op_sld.vmv && spatz_req_i.op_sld.insert)
          spatz_req_d.rs1 = MAXEW == EW_32 ? {1{spatz_req_i.rs1[31:0]}} : {2{spatz_req_i.rs1[31:0]}};
      end
      default: begin
        spatz_req_d.vl     = spatz_req_i.vl << MAXEW;
        spatz_req_d.vstart = spatz_req_i.vstart << MAXEW;
        if (spatz_req_i.op_sld.vmv && spatz_req_i.op_sld.insert)
          spatz_req_d.rs1 = spatz_req_i.rs1;
      end
    endcase
  end: proc_spatz_req

  ///////////////////////
  //  Output Register  //
  ///////////////////////
  
  // Added
  // Is the instruction gather
  logic is_gather;
  assign is_gather = spatz_req_valid && (spatz_req.op == VRGATHER);

  vrf_data_t gather_data_d,  gather_data_q;
  vrf_data_t gather_partial_data;
  logic      gather_valid_d, gather_valid_q;
  logic      gather_finished_d, gather_finished_q;
  logic      half_gathered_d, half_gathered_q;
  `FF(half_gathered_q, half_gathered_d, '0)
  
  typedef struct packed {
    vrf_addr_t waddr;
    vrf_data_t wdata;
    vrf_be_t wbe;
  } vrf_req_t;

  vrf_req_t vrf_req_d, vrf_req_q;
  logic     vrf_req_valid_d, vrf_req_ready_d;
  logic     vrf_req_valid_q, vrf_req_ready_q;

  always_comb begin: gather_output_comb
    
    if (spatz_req_valid && is_gather) begin
      if (vrf_rvalid_i[0] && vrf_rvalid_i[1]) begin
        gather_data_d  = gather_valid_q ? gather_partial_data : (gather_data_q | gather_partial_data);
        gather_valid_d = half_gathered_q || gather_finished_d;
      end else if (half_gathered_d) begin
        gather_data_d  = gather_data_q;
        gather_valid_d = '0;
      end else begin
        gather_data_d  = '0;
        gather_valid_d = '0;
      end
    end else if (gather_finished_q) begin
      gather_data_d  = '0;
      gather_valid_d = '0;
    end else begin
      gather_data_d  = gather_data_q;
      gather_valid_d = gather_valid_q;
    end
  end: gather_output_comb

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      gather_data_q     <= '0;
      gather_valid_q    <= '0;
      gather_finished_q <= '0; 
    end else begin
      gather_data_q     <= gather_data_d;
      gather_valid_q    <= gather_valid_d;
      gather_finished_q <= gather_finished_d;
    end
  end

  spill_register #(
    .T(vrf_req_t)
  ) i_vrf_req_register (
    .clk_i  (clk_i          ),
    .rst_ni (rst_ni         ),
    .data_i (vrf_req_d      ),
    .valid_i(vrf_req_valid_d),
    .ready_o(vrf_req_ready_d),
    .data_o (vrf_req_q      ),
    .valid_o(vrf_req_valid_q),
    .ready_i(vrf_req_ready_q)
  );

  assign vrf_waddr_o     = vrf_req_q.waddr;
  assign vrf_wdata_o     = vrf_req_q.wdata; 
  assign vrf_wbe_o       = vrf_req_q.wbe;
  assign vrf_we_o        = vrf_req_valid_q; 
  assign vrf_id_o[0]     = spatz_req.id; // ID of the instruction currently reading elements
  assign vrf_id_o[1]     = (spatz_req.op == VRGATHER) ? spatz_req.id : '0; 
                                         // If gather, ID of the instruction currently reading elements
  assign vrf_req_ready_q = vrf_wvalid_i;

  /////////////
  // Signals //
  /////////////

  // Is the register file operation valid?
  logic vreg_operations_finished;

  // Is the vector length zero (no active instruction)
  logic is_vl_zero;
  assign is_vl_zero = spatz_req.vl == 'd0;

  // Is the instruction slide up
  logic is_slide_up;
  assign is_slide_up = spatz_req.op == VSLIDEUP;

  // Instruction currently committing results
  spatz_id_t op_id_q, op_id_d;
  `FF(op_id_q, op_id_d, '0)

  // Number of bytes we slide up or down
  vlen_t slide_amount_q, slide_amount_d;
  `FF(slide_amount_q, slide_amount_d, '0)

  // Are we doing a vregfile read prefetch (when we slide down)
  logic prefetch_q, prefetch_d;
  `FF(prefetch_q, prefetch_d, 1'b0);

  ///////////////////
  // State Handler //
  ///////////////////

  // Currently running instructions
  logic [NrParallelInstructions-1:0] running_d, running_q;
  `FF(running_q, running_d, '0)

  // Respond to controller if we are finished executing
  typedef enum logic {
    VSLDU_RUNNING,    // Running an instruction
    VSLDU_WAIT_WVALID // Waiting for the last wvalid to acknowledge the instruction
   } state_t;
   state_t state_q, state_d;
  `FF(state_q, state_d, VSLDU_RUNNING)

  // New instruction
  // Initialize the internal state one cycle in advance
  logic new_vsldu_request, new_vsldu_request_q;
  assign new_vsldu_request = spatz_req_valid && !running_q[spatz_req.id];

  `FF(new_vsldu_request_q, new_vsldu_request, '0)

  // Accept a new operation or clear req register if we are finished
  always_comb begin
    slide_amount_d = slide_amount_q;
    prefetch_d     = prefetch_q;
    running_d      = running_q;

    // Spatz SLDU ready when empty
    spatz_req_ready = !spatz_req_valid;

    // New request?
    if (new_vsldu_request) begin
      // Mark the instruction as running
      running_d[spatz_req.id] = 1'b1;

      slide_amount_d = spatz_req.op_sld.insert ? (spatz_req.op_sld.vmv ? 'd0 : 'd1) : spatz_req.rs1;
      slide_amount_d <<= spatz_req.vtype.vsew;

      prefetch_d = spatz_req.op == VSLIDEUP ? spatz_req.vstart >= VRFWordBWidth : 1'b1;
    end

    // Finished an instruction
    if (vreg_operations_finished) begin
      // We are handling an instruction
      spatz_req_ready = 1'b1;

      // No longer running this instruction
      running_d[spatz_req.id] = 1'b0;
    end

    // Clear the prefetch register
    if (prefetch_q && vrf_re_o[0] && vrf_rvalid_i[0]) // specify vrf_re_o->vrf_re_o[0]
      prefetch_d = 1'b0;
  end

  /////////////////////
  //  Slide control  //
  /////////////////////

  // Vector register file counter signals
  logic  vreg_counter_en;
  vlen_t vreg_counter_delta;
  logic [$bits(vlen_t)-1 + $clog2(VRFWordBWidth)+6 : 0] vreg_counter_d;
  logic [$bits(vlen_t)-1 + $clog2(VRFWordBWidth)+6 : 0] vreg_counter_q;
  `FF(vreg_counter_q, vreg_counter_d, '0)

  // Are we on the first/last VRF operation?
  logic vreg_operation_first;
  logic vreg_operation_last;

  // FSM to decide whether we are on the first operation or not
  typedef enum logic {
    VREG_IDLE,
    VREG_WAIT_FIRST_WRITE
  } vreg_operation_first_t;
  vreg_operation_first_t vreg_operation_first_q, vreg_operation_first_d;
  `FF(vreg_operation_first_q, vreg_operation_first_d, VREG_IDLE)

  always_comb begin: vsldu_vreg_counter_proc
    // How many bytes are left to do
    automatic int unsigned delta = spatz_req.vl - vreg_counter_q; // value is in bytes

    // Default assignments
    vreg_counter_en        = 1'b0;
    vreg_counter_d         = vreg_counter_q;
    vreg_counter_delta     = '0;
    vreg_operation_first_d = vreg_operation_first_q;
    vreg_operation_first   = '0;

    half_gathered_d          = half_gathered_q;

    if(!is_gather) begin
      // Do we have a new request?
      if (new_vsldu_request) begin
        // Load vstart into the counter
        vreg_counter_d = spatz_req.vstart;
        if (!spatz_req.op_sld.insert && spatz_req.vstart < slide_amount_d && is_slide_up)
          vreg_counter_d = slide_amount_d;
      end

      // Is this the first/last operation?
      case (vreg_operation_first_q)
        VREG_IDLE: begin
          // Wait until our first write operation
          vreg_operation_first = spatz_req_valid && !prefetch_q && new_vsldu_request_q;
          if (spatz_req_valid && (vreg_counter_q <= slide_amount_q)) // maybe some problem with the last "()"
            vreg_operation_first_d = VREG_WAIT_FIRST_WRITE;

          if (vrf_req_valid_d && vrf_req_ready_d)
            vreg_operation_first_d = VREG_IDLE;
        end
        VREG_WAIT_FIRST_WRITE: begin
          vreg_operation_first = spatz_req_valid && !prefetch_q;
          if (vrf_req_valid_d && vrf_req_ready_d)
            vreg_operation_first_d = VREG_IDLE;
        end
        default:;
      endcase
    end

    vreg_operation_last  = spatz_req_valid && (is_gather || !prefetch_q) 
                        && (delta <= (VRFWordBWidth - vreg_counter_q[idx_width(VRFWordBWidth)-1:0]));

    // How many operations are we calculating now?
    if (spatz_req_valid) begin
      if (!is_gather) begin
        if (vreg_operation_last)
          vreg_counter_delta = delta;
        else if (vreg_operation_first)
          vreg_counter_delta = VRFWordBWidth - vreg_counter_d[idx_width(VRFWordBWidth)-1:0];
        else
          vreg_counter_delta = VRFWordBWidth;
      end else begin
          vreg_counter_delta = VRFWordBWidth; 
      end
    end

    // Do we have to increment the counter?
    vreg_counter_en =
                   ((spatz_req.use_vs2 && vrf_re_o[0] && vrf_rvalid_i[0]) || !spatz_req.use_vs2) 
                   && (is_gather ? ((spatz_req.use_vs1 && vrf_re_o[1] && vrf_rvalid_i[1]) || !spatz_req.use_vs1) : 1'b1)
                   && ((spatz_req.use_vd && ((is_gather && ~gather_valid_d) || (vrf_req_valid_d && vrf_req_ready_d))) || !spatz_req.use_vd);
    
    if (vreg_counter_en) begin
      if (vreg_operation_last)
        // Reset the counter
        vreg_counter_d = '0;
      else
        // Increment the counter
        vreg_counter_d = vreg_counter_q + vreg_counter_delta;
    end

    if (is_gather && vreg_counter_en) begin
      if (vreg_operation_last) begin
        // Reset the counter
        half_gathered_d = '0;
      end else begin
        // Increment the counter
        half_gathered_d = ~half_gathered_q;
      end
    end

    // Did we finish?
    vreg_operations_finished = vreg_operation_last && vreg_counter_en;
    gather_finished_d        = spatz_req_valid && is_gather && vreg_operation_last
                            && ((spatz_req.use_vs2 && vrf_re_o[0] && vrf_rvalid_i[0]) || !spatz_req.use_vs2) 
                            && ((spatz_req.use_vs1 && vrf_re_o[1] && vrf_rvalid_i[1]) || !spatz_req.use_vs1);
  end: vsldu_vreg_counter_proc

  always_comb begin: vsldu_rsp
    // Maintain state
    state_d = state_q;
    op_id_d = op_id_q;

    // Do not acknowledge anything
    vsldu_rsp_valid_o = 1'b0;
    vsldu_rsp_o       = '0;

    // ID of the instruction currently writing elements
    vrf_id_o[2] = spatz_req.id;

    case (state_q)
      VSLDU_RUNNING: begin
        // Did we finish the execution of an instruction?
        if (!is_vl_zero && vreg_operations_finished && spatz_req_valid) begin
          op_id_d = spatz_req.id;
          state_d = VSLDU_WAIT_WVALID;
        end
      end

      VSLDU_WAIT_WVALID: begin
        vrf_id_o[2] = op_id_q; // ID of the instruction currently writing to the VRF

        if (vrf_wvalid_i) begin
          vsldu_rsp_valid_o = 1'b1;
          vsldu_rsp_o.id    = op_id_q;
          state_d           = VSLDU_RUNNING;

          // Did we finish *another* instruction?
          if (!is_vl_zero && vreg_operations_finished && spatz_req_valid) begin
            op_id_d = spatz_req.id;
            state_d = VSLDU_WAIT_WVALID;
          end
        end
      end // case: VSLDU_WAIT_WVALID

      default:;
    endcase
  end: vsldu_rsp

  ////////////
  // Slider //
  ////////////

  // Shift overflow register
  vrf_data_t shift_overflow_q, shift_overflow_d;
  `FF(shift_overflow_q, shift_overflow_d, '0)

  // Number of bytes we have to shift the elements around
  // inside the register element
  logic [$clog2(VRFWordBWidth)-1:0] in_elem_offset, in_elem_flipped_offset;
  assign in_elem_offset         = slide_amount_d[$clog2(VRFWordBWidth)-1:0];
  assign in_elem_flipped_offset = VRFWordBWidth - in_elem_offset;

  // Data signals for different stages of the shift
  vrf_data_t data_in, data_out, data_low, data_high;
  logic [$bits(vlen_t)-1 + $clog2(VRFWordBWidth)+2:0] sparse_idx;
  
  always_comb begin

    shift_overflow_d = shift_overflow_q;
    gather_partial_data  = '0;

    if (!is_gather) begin

      data_in   = '0;
      data_out  = '0;
      data_high = '0;
      data_low  = '0;

      vrf_req_d.wbe   = '0;
      vrf_req_d.wdata = '0;

      // Is there a vector instruction executing now?
      if (!is_vl_zero) begin
        if (is_slide_up && spatz_req.op_sld.insert && spatz_req.op_sld.vmv) begin
          for (int b_src = 0; b_src < VRFWordBWidth; b_src++)
            data_in[(VRFWordBWidth-b_src-1)*8 +: 8] = spatz_req.rs1[b_src*8%ELEN +: 8];
        end else if (is_slide_up) begin
          // If we have a slide up operation, flip all bytes around (d[-i] = d[i])
          for (int b_src = 0; b_src < VRFWordBWidth; b_src++)
            data_in[(VRFWordBWidth-b_src-1)*8 +: 8] = vrf_rdata_i[0][b_src*8 +: 8];
        end else begin
          data_in = vrf_rdata_i[0];

          // If we are already over the MAXVL, all continuing elements are zero
          if ((vreg_counter_q >= MAXVL - slide_amount_q) || (vreg_operation_last && spatz_req.op_sld.insert))
            data_in = '0;
        end

        // Shift direct elements into the correct position
        for (int b_src = 0; b_src < VRFWordBWidth; b_src++) begin
          if (b_src >= in_elem_offset) begin
            // high elements
            for (int b_dst = 0; b_dst <= b_src; b_dst++)
              if (b_src-b_dst == in_elem_offset)
                data_high[b_dst*8 +: 8] = data_in[b_src*8 +: 8];
          end else begin
            // low elements
            for (int b_dst = b_src; b_dst < VRFWordBWidth; b_dst++)
              if (b_dst-b_src == in_elem_flipped_offset)
                data_low[b_dst*8 +: 8] = data_in[b_src*8 +: 8];
          end
        end
      
        // Combine overflow and direct elements together
        if (is_slide_up) begin
          if (vreg_counter_en || prefetch_q)
            shift_overflow_d = data_low;
          data_out = data_high | shift_overflow_q;
        end else begin
          if (vreg_counter_en || prefetch_q)
            shift_overflow_d = data_high;
          data_out = data_low | shift_overflow_q;

          // Insert rs1 element at the last position
          if (spatz_req.op_sld.insert && vreg_operation_last) begin
            for (int b = 0; b < VRFWordBWidth; b++)
              if (b >= (vreg_counter_q[$clog2(VRFWordBWidth)-1:0] + vreg_counter_delta - (4'b0001<<spatz_req.vtype.vsew)))
                data_out[b*8 +: 8] = data_low[b*8 +: 8];
            data_out = data_out | (vrf_data_t'(spatz_req.rs1) << 8*(vreg_counter_q[$clog2(VRFWordBWidth)-1:0]+vreg_counter_delta-(4'b0001<<spatz_req.vtype.vsew)));
          end
        end

        // If we have a slide up operation, flip all bytes back around (d[i] = d[-i])
        if (is_slide_up) begin
          for (int b_src = 0; b_src < VRFWordBWidth; b_src++)
            vrf_req_d.wdata[(VRFWordBWidth-b_src-1)*8 +: 8] = data_out[b_src*8 +: 8];

        // Insert rs1 element at the first position
        if (spatz_req.op_sld.insert && !spatz_req.op_sld.vmv && vreg_operation_first && spatz_req.vstart == 'd0)
          vrf_req_d.wdata = vrf_req_d.wdata | vrf_data_t'(spatz_req.rs1);
        end else begin
          vrf_req_d.wdata = data_out;
        end
        
        // Create byte enable mask
        for (int i = 0; i < VRFWordBWidth; i++)
          vrf_req_d.wbe[i] = i < vreg_counter_delta;

        // Special byte enable mask case when we are operating on the first register element.
        if (vreg_operation_first && is_slide_up)
          for (int i = 0; i < VRFWordBWidth; i++)
            vrf_req_d.wbe[i] = (spatz_req.op_sld.insert || (i >= slide_amount_d[$clog2(VRFWordBWidth)-1:0])) & (i < (vreg_counter_q[$clog2(VRFWordBWidth)-1:0] + vreg_counter_delta));
      end

      // Reset overflow register when finished
      if (vreg_operations_finished)
        shift_overflow_d = '0;

    end else begin

      data_high            = '0;
      data_low             = '0;
      vrf_req_d.wbe        = '0;
      vrf_req_d.wdata      = '0;
      sparse_idx           = vreg_counter_q[$bits(vlen_t)-1 + $clog2(VRFWordBWidth)+6:4];


      if (!is_vl_zero) begin

        if ((sparse_unit_effective_element_num == 1) && (sparse_unit_total_element_num == 2)) begin
          unique case (vrf_rdata_i[1][sparse_idx * 2 +: 2]) 
            2'd0: data_low[63:0]  = vrf_rdata_i[0][63:0];
            2'd1: data_low[63:0]  = vrf_rdata_i[0][127:64];
            default:;
          endcase
          unique case (vrf_rdata_i[1][(sparse_idx+1) * 2 +: 2]) 
            2'd0: data_high[63:0] = vrf_rdata_i[0][191:128];
            2'd1: data_high[63:0] = vrf_rdata_i[0][255:192];
            default:;
          endcase
        end else if ((sparse_unit_effective_element_num == 2) && (sparse_unit_total_element_num == 4)) begin
          unique case (vrf_rdata_i[1][sparse_idx * 2 +: 2]) 
            2'd0: data_low[63:0]  = vrf_rdata_i[0][63:0];
            2'd1: data_low[63:0]  = vrf_rdata_i[0][127:64];
            2'd2: data_low[63:0]  = vrf_rdata_i[0][191:128];
            2'd3: data_low[63:0]  = vrf_rdata_i[0][255:192];
            default:;
          endcase
          unique case (vrf_rdata_i[1][(sparse_idx+1) * 2 +: 2]) 
            2'd0: data_high[63:0] = vrf_rdata_i[0][63:0];
            2'd1: data_high[63:0] = vrf_rdata_i[0][127:64];
            2'd2: data_high[63:0] = vrf_rdata_i[0][191:128];
            2'd3: data_high[63:0] = vrf_rdata_i[0][255:192];
            default:;
          endcase
        end
        
        case (sparse_idx[1:0])
          2'd0 : gather_partial_data[127:0]   = {data_high[63:0], data_low[63:0]};
          2'd2 : gather_partial_data[255:128] = {data_high[63:0], data_low[63:0]};
        endcase
        
        vrf_req_d.wdata = gather_data_d; 

        for (int i = 0; i < VRFWordBWidth; i++) 
          vrf_req_d.wbe[i] = i < vreg_counter_delta;
        
      end
    end
  end

  // VRF signals

  assign vrf_re_o[0]       = spatz_req.use_vs2 && (spatz_req_valid || (!is_gather && prefetch_q)) && running_q[spatz_req.id];
  assign vrf_re_o[1]       = spatz_req.use_vs1 && spatz_req_valid && running_q[spatz_req.id];
  assign vrf_req_valid_d   = spatz_req_valid
                             && spatz_req.use_vd
                             && (is_gather ? gather_valid_d : 
                             (vrf_re_o[0] || !spatz_req.use_vs2) && (vrf_rvalid_i[0] || !spatz_req.use_vs2) && !prefetch_q);

  ////////////////////////
  // Address Generation //
  ////////////////////////

  vlen_t sld_offset_rd;

  always_comb begin
    sld_offset_rd   = is_slide_up ? (prefetch_q ? -slide_amount_q[$bits(vlen_t)-1:$clog2(VRFWordBWidth)] - 1 
                                                : -slide_amount_q[$bits(vlen_t)-1:$clog2(VRFWordBWidth)]) 
                                  :  prefetch_q ? slide_amount_q[$bits(vlen_t)-1:$clog2(VRFWordBWidth)] 
                                                : slide_amount_q[$bits(vlen_t)-1:$clog2(VRFWordBWidth)] + 1;
    vrf_raddr_o[0]  = {spatz_req.vs2, $clog2(NrWordsPerVector)'(1'b0)} + vreg_counter_q[$bits(vlen_t)-1:$clog2(VRFWordBWidth)] 
                    + (is_gather ? '0 : sld_offset_rd);
    vrf_raddr_o[1]  = {spatz_req.vs1, $clog2(NrWordsPerVector)'(1'b0)} + vreg_counter_q[$bits(vlen_t)-1 + $clog2(VRFWordBWidth)+6:$clog2(VRFWordBWidth)+6];
    // vreg_counter_q[$bits(vlen_t)-1:$clog2(VRFWordBWidth)+6] is the same as vreg_counter_q[$bits(vlen_t)-1:$clog2(VRFWordBWidth)]/64
    vrf_req_d.waddr = {spatz_req.vd,  $clog2(NrWordsPerVector)'(1'b0)} 
                    + (is_gather ? vreg_counter_q[$bits(vlen_t)-1:$clog2(VRFWordBWidth)+1] 
                                 : vreg_counter_q[$bits(vlen_t)-1:$clog2(VRFWordBWidth)]);
  end

endmodule : spatz_vsldu
