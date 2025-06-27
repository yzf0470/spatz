onerror {resume}
quietly WaveActivateNextPane {} 0
add wave -noupdate /tb_bin/i_dut/cluster_probe
add wave -noupdate -expand -group {core[0]} -group Params {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/BootAddr}
add wave -noupdate -expand -group {core[0]} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/clk_i}
add wave -noupdate -expand -group {core[0]} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/rst_i}
add wave -noupdate -expand -group {core[0]} -radix unsigned {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/hart_id_i}
add wave -noupdate -expand -group {core[0]} -divider Instructions
add wave -noupdate -expand -group {core[0]} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/inst_addr_o}
add wave -noupdate -expand -group {core[0]} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/inst_data_i}
add wave -noupdate -expand -group {core[0]} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/inst_valid_o}
add wave -noupdate -expand -group {core[0]} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/inst_ready_i}
add wave -noupdate -expand -group {core[0]} -divider Load/Store
add wave -noupdate -expand -group {core[0]} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/data_req_o}
add wave -noupdate -expand -group {core[0]} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/data_rsp_i}
add wave -noupdate -expand -group {core[0]} -divider Accelerator
add wave -noupdate -expand -group {core[0]} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/acc_qreq_o}
add wave -noupdate -expand -group {core[0]} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/acc_qrsp_i}
add wave -noupdate -expand -group {core[0]} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/acc_qvalid_o}
add wave -noupdate -expand -group {core[0]} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/acc_qready_i}
add wave -noupdate -expand -group {core[0]} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/acc_prsp_i}
add wave -noupdate -expand -group {core[0]} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/acc_pvalid_i}
add wave -noupdate -expand -group {core[0]} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/acc_pready_o}
add wave -noupdate -expand -group {core[0]} -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/illegal_inst}
add wave -noupdate -expand -group {core[0]} -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/stall}
add wave -noupdate -expand -group {core[0]} -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/lsu_stall}
add wave -noupdate -expand -group {core[0]} -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/acc_stall}
add wave -noupdate -expand -group {core[0]} -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/zero_lsb}
add wave -noupdate -expand -group {core[0]} -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/pc_d}
add wave -noupdate -expand -group {core[0]} -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/pc_q}
add wave -noupdate -expand -group {core[0]} -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/wfi_d}
add wave -noupdate -expand -group {core[0]} -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/wfi_q}
add wave -noupdate -expand -group {core[0]} -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/fcsr_d}
add wave -noupdate -expand -group {core[0]} -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/fcsr_q}
add wave -noupdate -expand -group {core[0]} -group Snitch -divider LSU
add wave -noupdate -expand -group {core[0]} -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/ls_size}
add wave -noupdate -expand -group {core[0]} -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/ls_amo}
add wave -noupdate -expand -group {core[0]} -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/ld_result}
add wave -noupdate -expand -group {core[0]} -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/lsu_qready}
add wave -noupdate -expand -group {core[0]} -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/lsu_qvalid}
add wave -noupdate -expand -group {core[0]} -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/lsu_pvalid}
add wave -noupdate -expand -group {core[0]} -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/lsu_pready}
add wave -noupdate -expand -group {core[0]} -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/lsu_rd}
add wave -noupdate -expand -group {core[0]} -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/retire_load}
add wave -noupdate -expand -group {core[0]} -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/retire_i}
add wave -noupdate -expand -group {core[0]} -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/retire_acc}
add wave -noupdate -expand -group {core[0]} -group Snitch -divider ALU
add wave -noupdate -expand -group {core[0]} -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/opa}
add wave -noupdate -expand -group {core[0]} -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/opb}
add wave -noupdate -expand -group {core[0]} -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/iimm}
add wave -noupdate -expand -group {core[0]} -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/uimm}
add wave -noupdate -expand -group {core[0]} -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/jimm}
add wave -noupdate -expand -group {core[0]} -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/bimm}
add wave -noupdate -expand -group {core[0]} -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/simm}
add wave -noupdate -expand -group {core[0]} -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/adder_result}
add wave -noupdate -expand -group {core[0]} -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/alu_result}
add wave -noupdate -expand -group {core[0]} -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/rd}
add wave -noupdate -expand -group {core[0]} -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/rs1}
add wave -noupdate -expand -group {core[0]} -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/rs2}
add wave -noupdate -expand -group {core[0]} -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/gpr_raddr}
add wave -noupdate -expand -group {core[0]} -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/gpr_rdata}
add wave -noupdate -expand -group {core[0]} -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/gpr_waddr}
add wave -noupdate -expand -group {core[0]} -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/gpr_wdata}
add wave -noupdate -expand -group {core[0]} -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/gpr_we}
add wave -noupdate -expand -group {core[0]} -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/consec_pc}
add wave -noupdate -expand -group {core[0]} -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/sb_d}
add wave -noupdate -expand -group {core[0]} -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/sb_q}
add wave -noupdate -expand -group {core[0]} -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/is_load}
add wave -noupdate -expand -group {core[0]} -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/is_store}
add wave -noupdate -expand -group {core[0]} -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/is_signed}
add wave -noupdate -expand -group {core[0]} -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/ls_misaligned}
add wave -noupdate -expand -group {core[0]} -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/ld_addr_misaligned}
add wave -noupdate -expand -group {core[0]} -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/st_addr_misaligned}
add wave -noupdate -expand -group {core[0]} -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/valid_instr}
add wave -noupdate -expand -group {core[0]} -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/exception}
add wave -noupdate -expand -group {core[0]} -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/alu_op}
add wave -noupdate -expand -group {core[0]} -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/opa_select}
add wave -noupdate -expand -group {core[0]} -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/opb_select}
add wave -noupdate -expand -group {core[0]} -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/write_rd}
add wave -noupdate -expand -group {core[0]} -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/uses_rd}
add wave -noupdate -expand -group {core[0]} -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/next_pc}
add wave -noupdate -expand -group {core[0]} -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/rd_select}
add wave -noupdate -expand -group {core[0]} -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/rd_bypass}
add wave -noupdate -expand -group {core[0]} -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/is_branch}
add wave -noupdate -expand -group {core[0]} -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/csr_rvalue}
add wave -noupdate -expand -group {core[0]} -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/csr_en}
add wave -noupdate -expand -group {core[0]} -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/acc_register_rd}
add wave -noupdate -expand -group {core[0]} -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/operands_ready}
add wave -noupdate -expand -group {core[0]} -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/dst_ready}
add wave -noupdate -expand -group {core[0]} -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/opa_ready}
add wave -noupdate -expand -group {core[0]} -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/opb_ready}
add wave -noupdate -expand -group {core[0]} -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/shift_opa}
add wave -noupdate -expand -group {core[0]} -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/shift_opa_reversed}
add wave -noupdate -expand -group {core[0]} -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/shift_right_result}
add wave -noupdate -expand -group {core[0]} -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/shift_left_result}
add wave -noupdate -expand -group {core[0]} -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/shift_opa_ext}
add wave -noupdate -expand -group {core[0]} -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/shift_right_result_ext}
add wave -noupdate -expand -group {core[0]} -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/shift_left}
add wave -noupdate -expand -group {core[0]} -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/shift_arithmetic}
add wave -noupdate -expand -group {core[0]} -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/alu_opa}
add wave -noupdate -expand -group {core[0]} -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/alu_opb}
add wave -noupdate -expand -group {core[0]} -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/alu_writeback}
add wave -noupdate -expand -group {core[0]} -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/acc_mem_cnt_d}
add wave -noupdate -expand -group {core[0]} -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/acc_mem_cnt_q}
add wave -noupdate -expand -group {core[0]} -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/acc_mem_str_cnt_d}
add wave -noupdate -expand -group {core[0]} -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/acc_mem_str_cnt_q}
add wave -noupdate -expand -group {core[0]} -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/core_events_o}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal -group RF {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/i_snitch_regfile/clk_i}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal -group RF {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/i_snitch_regfile/raddr_i}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal -group RF {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/i_snitch_regfile/rdata_o}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal -group RF {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/i_snitch_regfile/waddr_i}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal -group RF {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/i_snitch_regfile/wdata_i}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal -group RF {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/i_snitch_regfile/we_i}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal -group RF {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/i_snitch_regfile/mem}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal -group RF {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/i_snitch_regfile/we_dec}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/clk_i}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/rst_i}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/hart_id_i}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/irq_i}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/flush_i_valid_o}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/flush_i_ready_i}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/inst_addr_o}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/inst_cacheable_o}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/inst_data_i}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/inst_valid_o}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/inst_ready_i}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/acc_qreq_o}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/acc_qrsp_i}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/acc_qvalid_o}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/acc_qready_i}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/acc_prsp_i}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/acc_pvalid_i}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/acc_pready_o}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/acc_mem_finished_i}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/acc_mem_str_finished_i}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/data_req_o}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/data_rsp_i}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/ptw_valid_o}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/ptw_ready_i}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/ptw_va_o}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/ptw_ppn_o}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/ptw_pte_i}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/ptw_is_4mega_i}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/fpu_rnd_mode_o}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/fpu_fmt_mode_o}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/fpu_status_i}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/core_events_o}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/illegal_inst}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/illegal_csr}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/interrupt}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/ecall}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/ebreak}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/zero_lsb}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/meip}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/mtip}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/msip}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/mcip}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/seip}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/stip}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/ssip}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/scip}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/interrupts_enabled}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/any_interrupt_pending}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/pc_d}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/pc_q}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/wfi_d}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/wfi_q}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/consec_pc}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/iimm}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/uimm}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/jimm}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/bimm}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/simm}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/opa}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/opb}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/adder_result}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/alu_result}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/rd}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/rs1}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/rs2}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/stall}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/lsu_stall}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/gpr_raddr}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/gpr_rdata}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/gpr_waddr}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/gpr_wdata}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/gpr_we}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/sb_d}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/sb_q}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/is_load}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/is_store}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/is_signed}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/is_fp_load}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/is_fp_store}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/ls_misaligned}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/ld_addr_misaligned}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/st_addr_misaligned}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/inst_addr_misaligned}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/itlb_valid}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/itlb_ready}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/itlb_va}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/itlb_page_fault}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/itlb_pa}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/dtlb_valid}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/dtlb_ready}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/dtlb_va}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/dtlb_page_fault}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/dtlb_pa}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/trans_ready}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/trans_active}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/itlb_trans_valid}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/dtlb_trans_valid}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/trans_active_exp}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/tlb_flush}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/ls_size}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/ls_amo}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/ld_result}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/lsu_qready}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/lsu_qvalid}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/lsu_tlb_qready}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/lsu_tlb_qvalid}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/lsu_pvalid}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/lsu_pready}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/lsu_empty}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/ls_paddr}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/lsu_rd}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/retire_load}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/retire_i}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/retire_acc}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/acc_stall}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/valid_instr}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/exception}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/alu_op}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/opa_select}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/opb_select}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/write_rd}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/uses_rd}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/next_pc}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/rd_select}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/rd_bypass}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/is_branch}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/csr_rvalue}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/csr_en}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/scratch_d}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/scratch_q}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/epc_d}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/epc_q}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/tvec_d}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/tvec_q}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/cause_d}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/cause_q}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/cause_irq_d}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/cause_irq_q}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/spp_d}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/spp_q}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/mpp_d}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/mpp_q}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/ie_d}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/ie_q}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/pie_d}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/pie_q}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/eie_d}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/eie_q}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/tie_d}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/tie_q}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/sie_d}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/sie_q}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/cie_d}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/cie_q}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/seip_d}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/seip_q}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/stip_d}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/stip_q}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/ssip_d}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/ssip_q}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/scip_d}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/scip_q}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/priv_lvl_d}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/priv_lvl_q}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/satp_d}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/satp_q}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/dcsr_d}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/dcsr_q}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/dpc_d}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/dpc_q}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/dscratch_d}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/dscratch_q}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/debug_d}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/debug_q}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/fcsr_d}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/fcsr_q}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/read_fcsr}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/cycle_q}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/instret_q}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/retired_instr_q}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/retired_load_q}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/retired_i_q}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/retired_acc_q}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/mseg_q}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/mseg_d}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/acc_register_rd}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/acc_mem_stall}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/acc_mem_store}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/operands_ready}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/dst_ready}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/opa_ready}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/opb_ready}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/npc}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/shift_opa}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/shift_opa_reversed}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/shift_right_result}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/shift_left_result}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/shift_opa_ext}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/shift_right_result_ext}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/shift_left}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/shift_arithmetic}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/alu_opa}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/alu_opb}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/lsu_qdata}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/acc_mem_cnt_q}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/acc_mem_cnt_d}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/acc_mem_str_cnt_q}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/acc_mem_str_cnt_d}
add wave -noupdate -expand -group {core[0]} -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/alu_writeback}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/issue_valid_i}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/issue_ready_o}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/issue_req_i}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/issue_rsp_o}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/rsp_valid_o}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/rsp_ready_i}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/rsp_o}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/spatz_mem_req_o}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/spatz_mem_req_valid_o}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/spatz_mem_req_ready_i}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/spatz_mem_rsp_i}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/spatz_mem_rsp_valid_i}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/clk_i}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/rst_ni}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/issue_req_i}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/issue_valid_i}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/issue_ready_o}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/issue_rsp_o}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/resp_o}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/resp_valid_o}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/resp_ready_i}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/issue_req_o}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/issue_valid_o}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/issue_ready_i}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/issue_rsp_i}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/resp_i}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/resp_valid_i}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/resp_ready_o}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/fp_lsu_mem_req_o}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/fp_lsu_mem_rsp_i}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/fp_lsu_mem_finished_o}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/fp_lsu_mem_str_finished_o}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/spatz_mem_finished_i}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/spatz_mem_str_finished_i}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/fd}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/fs1}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/fs2}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/fs3}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/fpr_raddr}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/fpr_rdata}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/fpr_waddr}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/fpr_wdata}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/fpr_we}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/sb_d}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/sb_q}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/use_fs1}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/use_fs2}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/use_fs3}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/use_fd}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/use_rd}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/operands_available}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/is_move}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/is_load}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/is_store}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/is_local}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/stall}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/lsu_stall}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/vlsu_stall}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/move_stall}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/illegal_inst}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/retire}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/ls_size}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/fp_lsu_qtag}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/fp_lsu_qwrite}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/fp_lsu_qsigned}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/fp_lsu_qaddr}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/fp_lsu_qdata}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/fp_lsu_qsize}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/fp_lsu_qamo}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/fp_lsu_qvalid}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/fp_lsu_qready}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/fp_lsu_pdata}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/fp_lsu_ptag}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/fp_lsu_pvalid}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/fp_lsu_pready}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/mem_qaddr}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/mem_qwrite}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/mem_qdata}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/mem_qstrb}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/mem_qid}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/mem_pdata}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/mem_perror}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/mem_pid}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/acc_mem_cnt_q}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/acc_mem_cnt_d}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/acc_mem_str_cnt_q}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/acc_mem_str_cnt_d}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/is_vector_load}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/is_vector_store}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/fp_move_result_i}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/fp_move_result_valid_i}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/fp_move_result_ready_o}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/fp_move_result_o}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/fp_move_result_valid_o}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/fp_move_result_ready_i}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/outstanding_store_q}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/outstanding_store_d}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} -group FPR {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/i_fpr/clk_i}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} -group FPR {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/i_fpr/raddr_i}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} -group FPR {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/i_fpr/rdata_o}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} -group FPR {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/i_fpr/waddr_i}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} -group FPR {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/i_fpr/wdata_i}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} -group FPR {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/i_fpr/we_i}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} -group FPR {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/i_fpr/mem}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} -group FPR {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/i_fpr/we_dec}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} -group LSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/i_fp_lsu/clk_i}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} -group LSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/i_fp_lsu/rst_i}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} -group LSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/i_fp_lsu/lsu_qtag_i}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} -group LSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/i_fp_lsu/lsu_qwrite_i}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} -group LSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/i_fp_lsu/lsu_qsigned_i}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} -group LSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/i_fp_lsu/lsu_qaddr_i}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} -group LSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/i_fp_lsu/lsu_qdata_i}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} -group LSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/i_fp_lsu/lsu_qsize_i}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} -group LSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/i_fp_lsu/lsu_qamo_i}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} -group LSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/i_fp_lsu/lsu_qvalid_i}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} -group LSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/i_fp_lsu/lsu_qready_o}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} -group LSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/i_fp_lsu/lsu_pdata_o}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} -group LSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/i_fp_lsu/lsu_ptag_o}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} -group LSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/i_fp_lsu/lsu_perror_o}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} -group LSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/i_fp_lsu/lsu_pvalid_o}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} -group LSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/i_fp_lsu/lsu_pready_i}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} -group LSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/i_fp_lsu/lsu_empty_o}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} -group LSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/i_fp_lsu/data_req_o}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} -group LSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/i_fp_lsu/data_rsp_i}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} -group LSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/i_fp_lsu/ld_result}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} -group LSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/i_fp_lsu/lsu_qdata}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} -group LSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/i_fp_lsu/data_qdata}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} -group LSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/i_fp_lsu/laq_in}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} -group LSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/i_fp_lsu/laq_out}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} -group LSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/i_fp_lsu/mem_out}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} -group LSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/i_fp_lsu/laq_full}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} -group LSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/i_fp_lsu/mem_full}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} -group LSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/i_fp_lsu/laq_push}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group {FPU Sequencer} -group LSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/i_fp_lsu/shifted_data}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_controller/clk_i}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_controller/rst_ni}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_controller/issue_valid_i}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_controller/issue_ready_o}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_controller/issue_req_i}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_controller/issue_rsp_o}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_controller/rsp_valid_o}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_controller/rsp_ready_i}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_controller/rsp_o}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_controller/fpu_rnd_mode_i}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_controller/fpu_fmt_mode_i}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_controller/spatz_req_valid_o}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_controller/spatz_req_o}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_controller/vfu_req_ready_i}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_controller/vfu_rsp_valid_i}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_controller/vfu_rsp_ready_o}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_controller/vfu_rsp_i}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_controller/vlsu_req_ready_i}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_controller/vlsu_rsp_valid_i}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_controller/vlsu_rsp_i}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_controller/vsldu_req_ready_i}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_controller/vsldu_rsp_valid_i}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_controller/vsldu_rsp_i}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_controller/sb_enable_i}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_controller/sb_wrote_result_i}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_controller/sb_enable_o}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_controller/sb_id_i}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_controller/spatz_req}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_controller/spatz_req_valid}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_controller/spatz_req_illegal}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_controller/vstart_d}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_controller/vstart_q}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_controller/vl_d}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_controller/vl_q}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_controller/vtype_d}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_controller/vtype_q}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_controller/decoder_req}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_controller/decoder_req_valid}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_controller/decoder_rsp}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_controller/decoder_rsp_valid}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_controller/buffer_spatz_req}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_controller/req_buffer_ready}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_controller/req_buffer_valid}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_controller/req_buffer_pop}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_controller/read_table_d}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_controller/read_table_q}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_controller/write_table_d}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_controller/write_table_q}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_controller/scoreboard_q}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_controller/scoreboard_d}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_controller/wrote_result_q}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_controller/wrote_result_d}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_controller/narrow_wide_q}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_controller/narrow_wide_d}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_controller/wrote_result_narrowing_q}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_controller/wrote_result_narrowing_d}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_controller/retire_csr}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_controller/stall}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_controller/vfu_stall}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_controller/vlsu_stall}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_controller/vsldu_stall}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_controller/running_insn_d}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_controller/running_insn_q}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_controller/next_insn_id}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_controller/running_insn_full}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_controller/vfu_rsp}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_controller/vfu_rsp_valid}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_controller/vfu_rsp_ready}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_controller/rsp_valid_d}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_controller/rsp_d}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group VRF -divider RegisterWrite
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group VRF {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vrf/waddr_i}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group VRF {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vrf/wdata_i}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group VRF {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vrf/we_i}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group VRF {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vrf/wbe_i}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group VRF {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vrf/wvalid_o}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group VRF -divider RegisterRead
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group VRF {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vrf/raddr_i}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group VRF {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vrf/rdata_o}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group VRF {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vrf/re_i}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group VRF {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vrf/rvalid_o}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group VRF -divider Internal
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group VRF {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vrf/waddr}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group VRF {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vrf/wdata}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group VRF {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vrf/we}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group VRF {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vrf/wbe}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group VRF {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vrf/raddr}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -group VRF {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vrf/rdata}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/clk_i}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/rst_ni}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/spatz_req_i}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/spatz_req_valid_i}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/spatz_req_ready_o}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/vlsu_rsp_valid_o}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/vlsu_rsp_o}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/vrf_waddr_o}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/vrf_wdata_o}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/vrf_we_o}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/vrf_wbe_o}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/vrf_wvalid_i}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/vrf_id_o}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/vrf_raddr_o}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/vrf_re_o}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/vrf_rdata_i}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/vrf_rvalid_i}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/spatz_mem_req_o}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/spatz_mem_req_valid_o}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/spatz_mem_req_ready_i}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/spatz_mem_rsp_i}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/spatz_mem_rsp_valid_i}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/spatz_mem_finished_o}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/spatz_mem_str_finished_o}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/spatz_req_d}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/mem_spatz_req}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/mem_spatz_req_valid}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/mem_spatz_req_ready}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/mem_is_strided}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/mem_is_indexed}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/state_d}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/state_q}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/store_count_q}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/store_count_d}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/rob_wdata}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/rob_wid}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/rob_push}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/rob_rvalid}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/rob_rdata}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/rob_pop}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/rob_rid}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/rob_req_id}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/rob_id}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/rob_full}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/rob_empty}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/mem_operation_valid}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/mem_operation_last}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/mem_counter_max}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/mem_counter_en}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/mem_counter_load}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/mem_counter_delta}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/mem_counter_d}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/mem_counter_q}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/mem_port_finished_q}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/mem_idx_counter_delta}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/mem_idx_counter_d}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/mem_idx_counter_q}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/mem_insn_finished_q}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/mem_insn_finished_d}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/mem_insn_pending_q}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/mem_insn_pending_d}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/commit_insn_d}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/commit_insn_push}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/commit_insn_q}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/commit_insn_pop}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/commit_insn_empty}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/commit_insn_valid}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/commit_counter_max}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/commit_counter_en}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/commit_counter_load}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/commit_counter_delta}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/commit_counter_d}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/commit_counter_q}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/commit_finished_q}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/commit_finished_d}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/mem_req_addr}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/vd_vreg_addr}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/vs2_vreg_addr}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/vd_elem_id}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/vs2_elem_id_d}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/vs2_elem_id_q}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/pending_index}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/mem_req_addr_offset}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/busy_q}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/busy_d}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/vlsu_finished_req}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/spatz_mem_req}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/spatz_mem_req_valid}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/spatz_mem_req_ready}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/commit_operation_valid}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/commit_operation_last}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/mem_is_load}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/mem_is_vstart_zero}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/mem_is_addr_unaligned}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/mem_is_single_element_operation}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/mem_single_element_size}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/mem_idx_single_element_size}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/commit_is_addr_unaligned}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/commit_is_single_element_operation}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/commit_single_element_size}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/vreg_addr_offset}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/offset_queue_full}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/vrf_req_d}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/vrf_req_q}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/vrf_req_valid_d}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/vrf_req_ready_d}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/vrf_req_valid_q}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/vrf_req_ready_q}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/vreg_start_0}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/catchup}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/mem_req_id}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/mem_req_data}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/mem_req_svalid}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/mem_req_strb}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/mem_req_lvalid}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/mem_req_last}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/mem_pending_d}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/mem_pending_q}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vlsu/mem_pending}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vsldu/clk_i}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vsldu/rst_ni}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vsldu/spatz_req_i}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vsldu/spatz_req_valid_i}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vsldu/spatz_req_ready_o}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vsldu/vsldu_rsp_valid_o}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vsldu/vsldu_rsp_o}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vsldu/vrf_waddr_o}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vsldu/vrf_wdata_o}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vsldu/vrf_we_o}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vsldu/vrf_wbe_o}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vsldu/vrf_wvalid_i}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vsldu/vrf_id_o}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vsldu/vrf_raddr_o}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vsldu/vrf_re_o}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vsldu/vrf_rdata_i}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vsldu/vrf_rvalid_i}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vsldu/spatz_req_d}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vsldu/spatz_req}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vsldu/spatz_req_valid}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vsldu/spatz_req_ready}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vsldu/vrf_req_d}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vsldu/vrf_req_q}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vsldu/vrf_req_valid_d}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vsldu/vrf_req_ready_d}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vsldu/vrf_req_valid_q}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vsldu/vrf_req_ready_q}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vsldu/vreg_operations_finished}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vsldu/is_vl_zero}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vsldu/is_slide_up}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vsldu/op_id_q}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vsldu/op_id_d}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vsldu/slide_amount_q}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vsldu/slide_amount_d}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vsldu/prefetch_q}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vsldu/prefetch_d}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vsldu/running_d}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vsldu/running_q}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vsldu/state_q}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vsldu/state_d}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vsldu/new_vsldu_request}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vsldu/new_vsldu_request_q}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vsldu/vreg_counter_en}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vsldu/vreg_counter_delta}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vsldu/vreg_counter_d}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vsldu/vreg_counter_q}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vsldu/vreg_operation_first}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vsldu/vreg_operation_last}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vsldu/vreg_operation_first_q}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vsldu/vreg_operation_first_d}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vsldu/shift_overflow_q}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vsldu/shift_overflow_d}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vsldu/in_elem_offset}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vsldu/in_elem_flipped_offset}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vsldu/data_in}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vsldu/data_out}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vsldu/data_low}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vsldu/data_high}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vsldu/sld_offset_rd}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/clk_i}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/rst_ni}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/hart_id_i}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/spatz_req_i}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/spatz_req_valid_i}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/spatz_req_ready_o}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/vfu_rsp_valid_o}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/vfu_rsp_ready_i}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/vfu_rsp_o}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/vrf_waddr_o}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/vrf_wdata_o}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/vrf_we_o}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/vrf_wbe_o}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/vrf_wvalid_i}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/vrf_id_o}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/vrf_raddr_o}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/vrf_re_o}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/vrf_rdata_i}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/vrf_rvalid_i}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/fpu_status_o}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/spatz_req}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/spatz_req_valid}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/spatz_req_ready}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/vl_q}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/vl_d}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/busy_q}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/busy_d}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/nr_elem_word}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/state_d}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/state_q}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/ipu_result_tag}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/fpu_result_tag}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/result_tag}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/input_tag}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/vstart}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/stall}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/reduction_operand_ready_d}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/reduction_operand_ready_q}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/op1_is_ready}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/op2_is_ready}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/op3_is_ready}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/operands_ready}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/valid_operations}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/pending_results}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/word_issued}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/running_d}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/running_q}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/is_fpu_insn}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/is_fpu_busy}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/is_ipu_busy}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/scalar_result}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/last_request}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/reduction_state_d}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/reduction_state_q}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/reduction_done}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/narrowing_upper_d}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/narrowing_upper_q}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/widening_upper_d}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/widening_upper_q}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/result}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/result_valid}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/result_ready}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/reduction_q}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/reduction_d}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/ipu_result}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/ipu_result_valid}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/ipu_in_ready}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/fpu_result}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/fpu_result_valid}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/fpu_in_ready}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/operand1}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/operand2}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/operand3}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/in_ready}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/reduction_pointer_d}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/reduction_pointer_q}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/reduction_operand_request}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/vreg_wbe}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/vreg_we}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/vreg_r_req}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/vreg_addr_q}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/vreg_addr_d}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/vreg_wdata}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/int_ipu_in_ready}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/int_ipu_operand1}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/int_ipu_operand2}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/int_ipu_operand3}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/int_ipu_result}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/int_ipu_result_tag}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/int_ipu_result_valid}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/int_ipu_result_ready}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/int_ipu_busy}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/ipu_wide_operand1}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/ipu_wide_operand2}
add wave -noupdate -expand -group {core[0]} -expand -group Spatz -expand -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/ipu_wide_operand3}
add wave -noupdate -expand -group {core[0]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/clk_i}
add wave -noupdate -expand -group {core[0]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/clk_d2_i}
add wave -noupdate -expand -group {core[0]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/rst_ni}
add wave -noupdate -expand -group {core[0]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/testmode_i}
add wave -noupdate -expand -group {core[0]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/hart_id_i}
add wave -noupdate -expand -group {core[0]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/irq_i}
add wave -noupdate -expand -group {core[0]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/hive_req_o}
add wave -noupdate -expand -group {core[0]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/hive_rsp_i}
add wave -noupdate -expand -group {core[0]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/data_req_o}
add wave -noupdate -expand -group {core[0]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/data_rsp_i}
add wave -noupdate -expand -group {core[0]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/tcdm_req_o}
add wave -noupdate -expand -group {core[0]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/tcdm_rsp_i}
add wave -noupdate -expand -group {core[0]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/axi_dma_req_o}
add wave -noupdate -expand -group {core[0]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/axi_dma_res_i}
add wave -noupdate -expand -group {core[0]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/axi_dma_busy_o}
add wave -noupdate -expand -group {core[0]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/axi_dma_perf_o}
add wave -noupdate -expand -group {core[0]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/axi_dma_events_o}
add wave -noupdate -expand -group {core[0]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/core_events_o}
add wave -noupdate -expand -group {core[0]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/tcdm_addr_base_i}
add wave -noupdate -expand -group {core[0]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/acc_snitch_req}
add wave -noupdate -expand -group {core[0]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/acc_snitch_demux}
add wave -noupdate -expand -group {core[0]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/acc_snitch_resp}
add wave -noupdate -expand -group {core[0]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/acc_demux_snitch}
add wave -noupdate -expand -group {core[0]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/acc_resp}
add wave -noupdate -expand -group {core[0]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/dma_resp}
add wave -noupdate -expand -group {core[0]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/acc_snitch_demux_qvalid}
add wave -noupdate -expand -group {core[0]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/acc_snitch_demux_qready}
add wave -noupdate -expand -group {core[0]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/acc_qvalid}
add wave -noupdate -expand -group {core[0]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/acc_qready}
add wave -noupdate -expand -group {core[0]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/dma_qvalid}
add wave -noupdate -expand -group {core[0]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/dma_qready}
add wave -noupdate -expand -group {core[0]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/acc_pvalid}
add wave -noupdate -expand -group {core[0]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/acc_pready}
add wave -noupdate -expand -group {core[0]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/dma_pvalid}
add wave -noupdate -expand -group {core[0]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/dma_pready}
add wave -noupdate -expand -group {core[0]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/acc_demux_snitch_valid}
add wave -noupdate -expand -group {core[0]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/acc_demux_snitch_ready}
add wave -noupdate -expand -group {core[0]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/fpu_rnd_mode}
add wave -noupdate -expand -group {core[0]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/fpu_fmt_mode}
add wave -noupdate -expand -group {core[0]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/fpu_status}
add wave -noupdate -expand -group {core[0]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/snitch_events}
add wave -noupdate -expand -group {core[0]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/snitch_dreq_d}
add wave -noupdate -expand -group {core[0]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/snitch_dreq_q}
add wave -noupdate -expand -group {core[0]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/merged_dreq}
add wave -noupdate -expand -group {core[0]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/snitch_drsp_d}
add wave -noupdate -expand -group {core[0]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/snitch_drsp_q}
add wave -noupdate -expand -group {core[0]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/merged_drsp}
add wave -noupdate -expand -group {core[0]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/spatz_mem_finished}
add wave -noupdate -expand -group {core[0]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/spatz_mem_str_finished}
add wave -noupdate -expand -group {core[0]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/fp_lsu_mem_req}
add wave -noupdate -expand -group {core[0]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/fp_lsu_mem_rsp}
add wave -noupdate -expand -group {core[0]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/spatz_mem_req}
add wave -noupdate -expand -group {core[0]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/spatz_mem_req_valid}
add wave -noupdate -expand -group {core[0]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/spatz_mem_req_ready}
add wave -noupdate -expand -group {core[0]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/spatz_mem_rsp}
add wave -noupdate -expand -group {core[0]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/spatz_mem_rsp_valid}
add wave -noupdate -expand -group {core[0]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/data_tcdm_req}
add wave -noupdate -expand -group {core[0]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/data_tcdm_rsp}
add wave -noupdate -expand -group {core[0]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/slave_select}
add wave -noupdate -expand -group {core[0]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/addr_map}
add wave -noupdate -expand -group {core[0]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/f}
add wave -noupdate -expand -group {core[0]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/cycle}
add wave -noupdate -group {core[1]} -group Params {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/BootAddr}
add wave -noupdate -group {core[1]} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/clk_i}
add wave -noupdate -group {core[1]} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/rst_i}
add wave -noupdate -group {core[1]} -radix unsigned {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/hart_id_i}
add wave -noupdate -group {core[1]} -divider Instructions
add wave -noupdate -group {core[1]} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/inst_addr_o}
add wave -noupdate -group {core[1]} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/inst_data_i}
add wave -noupdate -group {core[1]} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/inst_valid_o}
add wave -noupdate -group {core[1]} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/inst_ready_i}
add wave -noupdate -group {core[1]} -divider Load/Store
add wave -noupdate -group {core[1]} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/data_req_o}
add wave -noupdate -group {core[1]} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/data_rsp_i}
add wave -noupdate -group {core[1]} -divider Accelerator
add wave -noupdate -group {core[1]} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/acc_qreq_o}
add wave -noupdate -group {core[1]} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/acc_qrsp_i}
add wave -noupdate -group {core[1]} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/acc_qvalid_o}
add wave -noupdate -group {core[1]} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/acc_qready_i}
add wave -noupdate -group {core[1]} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/acc_prsp_i}
add wave -noupdate -group {core[1]} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/acc_pvalid_i}
add wave -noupdate -group {core[1]} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/acc_pready_o}
add wave -noupdate -group {core[1]} -expand -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/illegal_inst}
add wave -noupdate -group {core[1]} -expand -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/stall}
add wave -noupdate -group {core[1]} -expand -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/lsu_stall}
add wave -noupdate -group {core[1]} -expand -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/acc_stall}
add wave -noupdate -group {core[1]} -expand -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/zero_lsb}
add wave -noupdate -group {core[1]} -expand -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/pc_d}
add wave -noupdate -group {core[1]} -expand -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/pc_q}
add wave -noupdate -group {core[1]} -expand -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/wfi_d}
add wave -noupdate -group {core[1]} -expand -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/wfi_q}
add wave -noupdate -group {core[1]} -expand -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/fcsr_d}
add wave -noupdate -group {core[1]} -expand -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/fcsr_q}
add wave -noupdate -group {core[1]} -expand -group Snitch -divider LSU
add wave -noupdate -group {core[1]} -expand -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/ls_size}
add wave -noupdate -group {core[1]} -expand -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/ls_amo}
add wave -noupdate -group {core[1]} -expand -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/ld_result}
add wave -noupdate -group {core[1]} -expand -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/lsu_qready}
add wave -noupdate -group {core[1]} -expand -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/lsu_qvalid}
add wave -noupdate -group {core[1]} -expand -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/lsu_pvalid}
add wave -noupdate -group {core[1]} -expand -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/lsu_pready}
add wave -noupdate -group {core[1]} -expand -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/lsu_rd}
add wave -noupdate -group {core[1]} -expand -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/retire_load}
add wave -noupdate -group {core[1]} -expand -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/retire_i}
add wave -noupdate -group {core[1]} -expand -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/retire_acc}
add wave -noupdate -group {core[1]} -expand -group Snitch -divider ALU
add wave -noupdate -group {core[1]} -expand -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/opa}
add wave -noupdate -group {core[1]} -expand -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/opb}
add wave -noupdate -group {core[1]} -expand -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/iimm}
add wave -noupdate -group {core[1]} -expand -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/uimm}
add wave -noupdate -group {core[1]} -expand -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/jimm}
add wave -noupdate -group {core[1]} -expand -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/bimm}
add wave -noupdate -group {core[1]} -expand -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/simm}
add wave -noupdate -group {core[1]} -expand -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/adder_result}
add wave -noupdate -group {core[1]} -expand -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/alu_result}
add wave -noupdate -group {core[1]} -expand -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/rd}
add wave -noupdate -group {core[1]} -expand -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/rs1}
add wave -noupdate -group {core[1]} -expand -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/rs2}
add wave -noupdate -group {core[1]} -expand -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/gpr_raddr}
add wave -noupdate -group {core[1]} -expand -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/gpr_rdata}
add wave -noupdate -group {core[1]} -expand -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/gpr_waddr}
add wave -noupdate -group {core[1]} -expand -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/gpr_wdata}
add wave -noupdate -group {core[1]} -expand -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/gpr_we}
add wave -noupdate -group {core[1]} -expand -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/consec_pc}
add wave -noupdate -group {core[1]} -expand -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/sb_d}
add wave -noupdate -group {core[1]} -expand -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/sb_q}
add wave -noupdate -group {core[1]} -expand -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/is_load}
add wave -noupdate -group {core[1]} -expand -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/is_store}
add wave -noupdate -group {core[1]} -expand -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/is_signed}
add wave -noupdate -group {core[1]} -expand -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/ls_misaligned}
add wave -noupdate -group {core[1]} -expand -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/ld_addr_misaligned}
add wave -noupdate -group {core[1]} -expand -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/st_addr_misaligned}
add wave -noupdate -group {core[1]} -expand -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/valid_instr}
add wave -noupdate -group {core[1]} -expand -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/exception}
add wave -noupdate -group {core[1]} -expand -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/alu_op}
add wave -noupdate -group {core[1]} -expand -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/opa_select}
add wave -noupdate -group {core[1]} -expand -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/opb_select}
add wave -noupdate -group {core[1]} -expand -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/write_rd}
add wave -noupdate -group {core[1]} -expand -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/uses_rd}
add wave -noupdate -group {core[1]} -expand -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/next_pc}
add wave -noupdate -group {core[1]} -expand -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/rd_select}
add wave -noupdate -group {core[1]} -expand -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/rd_bypass}
add wave -noupdate -group {core[1]} -expand -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/is_branch}
add wave -noupdate -group {core[1]} -expand -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/csr_rvalue}
add wave -noupdate -group {core[1]} -expand -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/csr_en}
add wave -noupdate -group {core[1]} -expand -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/acc_register_rd}
add wave -noupdate -group {core[1]} -expand -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/operands_ready}
add wave -noupdate -group {core[1]} -expand -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/dst_ready}
add wave -noupdate -group {core[1]} -expand -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/opa_ready}
add wave -noupdate -group {core[1]} -expand -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/opb_ready}
add wave -noupdate -group {core[1]} -expand -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/shift_opa}
add wave -noupdate -group {core[1]} -expand -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/shift_opa_reversed}
add wave -noupdate -group {core[1]} -expand -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/shift_right_result}
add wave -noupdate -group {core[1]} -expand -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/shift_left_result}
add wave -noupdate -group {core[1]} -expand -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/shift_opa_ext}
add wave -noupdate -group {core[1]} -expand -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/shift_right_result_ext}
add wave -noupdate -group {core[1]} -expand -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/shift_left}
add wave -noupdate -group {core[1]} -expand -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/shift_arithmetic}
add wave -noupdate -group {core[1]} -expand -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/alu_opa}
add wave -noupdate -group {core[1]} -expand -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/alu_opb}
add wave -noupdate -group {core[1]} -expand -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/alu_writeback}
add wave -noupdate -group {core[1]} -expand -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/acc_mem_cnt_d}
add wave -noupdate -group {core[1]} -expand -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/acc_mem_cnt_q}
add wave -noupdate -group {core[1]} -expand -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/acc_mem_str_cnt_d}
add wave -noupdate -group {core[1]} -expand -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/acc_mem_str_cnt_q}
add wave -noupdate -group {core[1]} -expand -group Snitch {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/core_events_o}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal -group RF {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/i_snitch_regfile/clk_i}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal -group RF {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/i_snitch_regfile/raddr_i}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal -group RF {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/i_snitch_regfile/rdata_o}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal -group RF {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/i_snitch_regfile/waddr_i}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal -group RF {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/i_snitch_regfile/wdata_i}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal -group RF {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/i_snitch_regfile/we_i}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal -group RF {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/i_snitch_regfile/mem}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal -group RF {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/i_snitch_regfile/we_dec}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/clk_i}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/rst_i}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/hart_id_i}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/irq_i}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/flush_i_valid_o}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/flush_i_ready_i}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/inst_addr_o}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/inst_cacheable_o}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/inst_data_i}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/inst_valid_o}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/inst_ready_i}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/acc_qreq_o}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/acc_qrsp_i}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/acc_qvalid_o}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/acc_qready_i}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/acc_prsp_i}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/acc_pvalid_i}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/acc_pready_o}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/acc_mem_finished_i}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/acc_mem_str_finished_i}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/data_req_o}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/data_rsp_i}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/ptw_valid_o}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/ptw_ready_i}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/ptw_va_o}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/ptw_ppn_o}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/ptw_pte_i}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/ptw_is_4mega_i}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/fpu_rnd_mode_o}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/fpu_fmt_mode_o}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/fpu_status_i}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/core_events_o}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/illegal_inst}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/illegal_csr}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/interrupt}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/ecall}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/ebreak}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/zero_lsb}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/meip}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/mtip}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/msip}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/mcip}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/seip}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/stip}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/ssip}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/scip}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/interrupts_enabled}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/any_interrupt_pending}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/pc_d}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/pc_q}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/wfi_d}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/wfi_q}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/consec_pc}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/iimm}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/uimm}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/jimm}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/bimm}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/simm}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/opa}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/opb}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/adder_result}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/alu_result}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/rd}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/rs1}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/rs2}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/stall}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/lsu_stall}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/gpr_raddr}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/gpr_rdata}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/gpr_waddr}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/gpr_wdata}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/gpr_we}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/sb_d}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/sb_q}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/is_load}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/is_store}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/is_signed}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/is_fp_load}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/is_fp_store}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/ls_misaligned}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/ld_addr_misaligned}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/st_addr_misaligned}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/inst_addr_misaligned}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/itlb_valid}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/itlb_ready}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/itlb_va}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/itlb_page_fault}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/itlb_pa}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/dtlb_valid}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/dtlb_ready}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/dtlb_va}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/dtlb_page_fault}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/dtlb_pa}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/trans_ready}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/trans_active}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/itlb_trans_valid}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/dtlb_trans_valid}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/trans_active_exp}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/tlb_flush}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/ls_size}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/ls_amo}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/ld_result}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/lsu_qready}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/lsu_qvalid}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/lsu_tlb_qready}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/lsu_tlb_qvalid}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/lsu_pvalid}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/lsu_pready}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/lsu_empty}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/ls_paddr}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/lsu_rd}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/retire_load}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/retire_i}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/retire_acc}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/acc_stall}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/valid_instr}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/exception}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/alu_op}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/opa_select}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/opb_select}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/write_rd}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/uses_rd}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/next_pc}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/rd_select}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/rd_bypass}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/is_branch}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/csr_rvalue}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/csr_en}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/scratch_d}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/scratch_q}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/epc_d}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/epc_q}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/tvec_d}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/tvec_q}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/cause_d}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/cause_q}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/cause_irq_d}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/cause_irq_q}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/spp_d}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/spp_q}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/mpp_d}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/mpp_q}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/ie_d}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/ie_q}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/pie_d}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/pie_q}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/eie_d}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/eie_q}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/tie_d}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/tie_q}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/sie_d}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/sie_q}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/cie_d}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/cie_q}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/seip_d}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/seip_q}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/stip_d}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/stip_q}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/ssip_d}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/ssip_q}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/scip_d}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/scip_q}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/priv_lvl_d}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/priv_lvl_q}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/satp_d}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/satp_q}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/dcsr_d}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/dcsr_q}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/dpc_d}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/dpc_q}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/dscratch_d}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/dscratch_q}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/debug_d}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/debug_q}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/fcsr_d}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/fcsr_q}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/read_fcsr}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/cycle_q}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/instret_q}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/retired_instr_q}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/retired_load_q}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/retired_i_q}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/retired_acc_q}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/mseg_q}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/mseg_d}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/acc_register_rd}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/acc_mem_stall}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/acc_mem_store}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/operands_ready}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/dst_ready}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/opa_ready}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/opb_ready}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/npc}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/shift_opa}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/shift_opa_reversed}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/shift_right_result}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/shift_left_result}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/shift_opa_ext}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/shift_right_result_ext}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/shift_left}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/shift_arithmetic}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/alu_opa}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/alu_opb}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/lsu_qdata}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/acc_mem_cnt_q}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/acc_mem_cnt_d}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/acc_mem_str_cnt_q}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/acc_mem_str_cnt_d}
add wave -noupdate -group {core[1]} -expand -group Snitch -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/alu_writeback}
add wave -noupdate -group {core[1]} -expand -group Spatz {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/issue_valid_i}
add wave -noupdate -group {core[1]} -expand -group Spatz {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/issue_ready_o}
add wave -noupdate -group {core[1]} -expand -group Spatz {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/issue_req_i}
add wave -noupdate -group {core[1]} -expand -group Spatz {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/issue_rsp_o}
add wave -noupdate -group {core[1]} -expand -group Spatz {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/rsp_valid_o}
add wave -noupdate -group {core[1]} -expand -group Spatz {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/rsp_ready_i}
add wave -noupdate -group {core[1]} -expand -group Spatz {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/rsp_o}
add wave -noupdate -group {core[1]} -expand -group Spatz {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/spatz_mem_req_o}
add wave -noupdate -group {core[1]} -expand -group Spatz {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/spatz_mem_req_valid_o}
add wave -noupdate -group {core[1]} -expand -group Spatz {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/spatz_mem_req_ready_i}
add wave -noupdate -group {core[1]} -expand -group Spatz {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/spatz_mem_rsp_i}
add wave -noupdate -group {core[1]} -expand -group Spatz {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/spatz_mem_rsp_valid_i}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/clk_i}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/rst_ni}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/issue_req_i}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/issue_valid_i}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/issue_ready_o}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/issue_rsp_o}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/resp_o}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/resp_valid_o}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/resp_ready_i}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/issue_req_o}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/issue_valid_o}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/issue_ready_i}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/issue_rsp_i}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/resp_i}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/resp_valid_i}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/resp_ready_o}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/fp_lsu_mem_req_o}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/fp_lsu_mem_rsp_i}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/fp_lsu_mem_finished_o}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/fp_lsu_mem_str_finished_o}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/spatz_mem_finished_i}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/spatz_mem_str_finished_i}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/fd}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/fs1}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/fs2}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/fs3}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/fpr_raddr}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/fpr_rdata}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/fpr_waddr}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/fpr_wdata}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/fpr_we}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/sb_d}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/sb_q}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/use_fs1}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/use_fs2}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/use_fs3}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/use_fd}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/use_rd}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/operands_available}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/is_move}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/is_load}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/is_store}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/is_local}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/stall}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/lsu_stall}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/vlsu_stall}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/move_stall}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/illegal_inst}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/retire}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/ls_size}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/fp_lsu_qtag}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/fp_lsu_qwrite}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/fp_lsu_qsigned}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/fp_lsu_qaddr}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/fp_lsu_qdata}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/fp_lsu_qsize}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/fp_lsu_qamo}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/fp_lsu_qvalid}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/fp_lsu_qready}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/fp_lsu_pdata}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/fp_lsu_ptag}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/fp_lsu_pvalid}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/fp_lsu_pready}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/mem_qaddr}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/mem_qwrite}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/mem_qdata}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/mem_qstrb}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/mem_qid}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/mem_pdata}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/mem_perror}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/mem_pid}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/acc_mem_cnt_q}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/acc_mem_cnt_d}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/acc_mem_str_cnt_q}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/acc_mem_str_cnt_d}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/is_vector_load}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/is_vector_store}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/fp_move_result_i}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/fp_move_result_valid_i}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/fp_move_result_ready_o}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/fp_move_result_o}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/fp_move_result_valid_o}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/fp_move_result_ready_i}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/outstanding_store_q}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/outstanding_store_d}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} -group FPR {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/i_fpr/clk_i}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} -group FPR {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/i_fpr/raddr_i}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} -group FPR {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/i_fpr/rdata_o}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} -group FPR {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/i_fpr/waddr_i}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} -group FPR {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/i_fpr/wdata_i}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} -group FPR {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/i_fpr/we_i}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} -group FPR {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/i_fpr/mem}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} -group FPR {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/i_fpr/we_dec}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} -group LSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/i_fp_lsu/clk_i}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} -group LSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/i_fp_lsu/rst_i}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} -group LSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/i_fp_lsu/lsu_qtag_i}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} -group LSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/i_fp_lsu/lsu_qwrite_i}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} -group LSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/i_fp_lsu/lsu_qsigned_i}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} -group LSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/i_fp_lsu/lsu_qaddr_i}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} -group LSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/i_fp_lsu/lsu_qdata_i}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} -group LSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/i_fp_lsu/lsu_qsize_i}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} -group LSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/i_fp_lsu/lsu_qamo_i}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} -group LSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/i_fp_lsu/lsu_qvalid_i}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} -group LSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/i_fp_lsu/lsu_qready_o}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} -group LSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/i_fp_lsu/lsu_pdata_o}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} -group LSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/i_fp_lsu/lsu_ptag_o}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} -group LSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/i_fp_lsu/lsu_perror_o}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} -group LSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/i_fp_lsu/lsu_pvalid_o}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} -group LSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/i_fp_lsu/lsu_pready_i}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} -group LSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/i_fp_lsu/lsu_empty_o}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} -group LSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/i_fp_lsu/data_req_o}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} -group LSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/i_fp_lsu/data_rsp_i}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} -group LSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/i_fp_lsu/ld_result}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} -group LSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/i_fp_lsu/lsu_qdata}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} -group LSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/i_fp_lsu/data_qdata}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} -group LSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/i_fp_lsu/laq_in}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} -group LSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/i_fp_lsu/laq_out}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} -group LSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/i_fp_lsu/mem_out}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} -group LSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/i_fp_lsu/laq_full}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} -group LSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/i_fp_lsu/mem_full}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} -group LSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/i_fp_lsu/laq_push}
add wave -noupdate -group {core[1]} -expand -group Spatz -group {FPU Sequencer} -group LSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/gen_fpu_sequencer/i_fpu_sequencer/i_fp_lsu/shifted_data}
add wave -noupdate -group {core[1]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_controller/clk_i}
add wave -noupdate -group {core[1]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_controller/rst_ni}
add wave -noupdate -group {core[1]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_controller/issue_valid_i}
add wave -noupdate -group {core[1]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_controller/issue_ready_o}
add wave -noupdate -group {core[1]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_controller/issue_req_i}
add wave -noupdate -group {core[1]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_controller/issue_rsp_o}
add wave -noupdate -group {core[1]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_controller/rsp_valid_o}
add wave -noupdate -group {core[1]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_controller/rsp_ready_i}
add wave -noupdate -group {core[1]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_controller/rsp_o}
add wave -noupdate -group {core[1]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_controller/fpu_rnd_mode_i}
add wave -noupdate -group {core[1]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_controller/fpu_fmt_mode_i}
add wave -noupdate -group {core[1]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_controller/spatz_req_valid_o}
add wave -noupdate -group {core[1]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_controller/spatz_req_o}
add wave -noupdate -group {core[1]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_controller/vfu_req_ready_i}
add wave -noupdate -group {core[1]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_controller/vfu_rsp_valid_i}
add wave -noupdate -group {core[1]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_controller/vfu_rsp_ready_o}
add wave -noupdate -group {core[1]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_controller/vfu_rsp_i}
add wave -noupdate -group {core[1]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_controller/vlsu_req_ready_i}
add wave -noupdate -group {core[1]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_controller/vlsu_rsp_valid_i}
add wave -noupdate -group {core[1]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_controller/vlsu_rsp_i}
add wave -noupdate -group {core[1]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_controller/vsldu_req_ready_i}
add wave -noupdate -group {core[1]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_controller/vsldu_rsp_valid_i}
add wave -noupdate -group {core[1]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_controller/vsldu_rsp_i}
add wave -noupdate -group {core[1]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_controller/sb_enable_i}
add wave -noupdate -group {core[1]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_controller/sb_wrote_result_i}
add wave -noupdate -group {core[1]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_controller/sb_enable_o}
add wave -noupdate -group {core[1]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_controller/sb_id_i}
add wave -noupdate -group {core[1]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_controller/spatz_req}
add wave -noupdate -group {core[1]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_controller/spatz_req_valid}
add wave -noupdate -group {core[1]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_controller/spatz_req_illegal}
add wave -noupdate -group {core[1]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_controller/vstart_d}
add wave -noupdate -group {core[1]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_controller/vstart_q}
add wave -noupdate -group {core[1]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_controller/vl_d}
add wave -noupdate -group {core[1]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_controller/vl_q}
add wave -noupdate -group {core[1]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_controller/vtype_d}
add wave -noupdate -group {core[1]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_controller/vtype_q}
add wave -noupdate -group {core[1]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_controller/decoder_req}
add wave -noupdate -group {core[1]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_controller/decoder_req_valid}
add wave -noupdate -group {core[1]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_controller/decoder_rsp}
add wave -noupdate -group {core[1]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_controller/decoder_rsp_valid}
add wave -noupdate -group {core[1]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_controller/buffer_spatz_req}
add wave -noupdate -group {core[1]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_controller/req_buffer_ready}
add wave -noupdate -group {core[1]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_controller/req_buffer_valid}
add wave -noupdate -group {core[1]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_controller/req_buffer_pop}
add wave -noupdate -group {core[1]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_controller/read_table_d}
add wave -noupdate -group {core[1]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_controller/read_table_q}
add wave -noupdate -group {core[1]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_controller/write_table_d}
add wave -noupdate -group {core[1]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_controller/write_table_q}
add wave -noupdate -group {core[1]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_controller/scoreboard_q}
add wave -noupdate -group {core[1]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_controller/scoreboard_d}
add wave -noupdate -group {core[1]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_controller/wrote_result_q}
add wave -noupdate -group {core[1]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_controller/wrote_result_d}
add wave -noupdate -group {core[1]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_controller/narrow_wide_q}
add wave -noupdate -group {core[1]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_controller/narrow_wide_d}
add wave -noupdate -group {core[1]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_controller/wrote_result_narrowing_q}
add wave -noupdate -group {core[1]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_controller/wrote_result_narrowing_d}
add wave -noupdate -group {core[1]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_controller/retire_csr}
add wave -noupdate -group {core[1]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_controller/stall}
add wave -noupdate -group {core[1]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_controller/vfu_stall}
add wave -noupdate -group {core[1]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_controller/vlsu_stall}
add wave -noupdate -group {core[1]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_controller/vsldu_stall}
add wave -noupdate -group {core[1]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_controller/running_insn_d}
add wave -noupdate -group {core[1]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_controller/running_insn_q}
add wave -noupdate -group {core[1]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_controller/next_insn_id}
add wave -noupdate -group {core[1]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_controller/running_insn_full}
add wave -noupdate -group {core[1]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_controller/vfu_rsp}
add wave -noupdate -group {core[1]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_controller/vfu_rsp_valid}
add wave -noupdate -group {core[1]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_controller/vfu_rsp_ready}
add wave -noupdate -group {core[1]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_controller/rsp_valid_d}
add wave -noupdate -group {core[1]} -expand -group Spatz -group Controller {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_controller/rsp_d}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VRF -divider RegisterWrite
add wave -noupdate -group {core[1]} -expand -group Spatz -group VRF {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vrf/waddr_i}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VRF {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vrf/wdata_i}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VRF {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vrf/we_i}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VRF {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vrf/wbe_i}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VRF {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vrf/wvalid_o}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VRF -divider RegisterRead
add wave -noupdate -group {core[1]} -expand -group Spatz -group VRF {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vrf/raddr_i}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VRF {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vrf/rdata_o}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VRF {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vrf/re_i}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VRF {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vrf/rvalid_o}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VRF -divider Internal
add wave -noupdate -group {core[1]} -expand -group Spatz -group VRF {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vrf/waddr}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VRF {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vrf/wdata}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VRF {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vrf/we}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VRF {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vrf/wbe}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VRF {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vrf/raddr}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VRF {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vrf/rdata}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/clk_i}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/rst_ni}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/spatz_req_i}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/spatz_req_valid_i}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/spatz_req_ready_o}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/vlsu_rsp_valid_o}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/vlsu_rsp_o}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/vrf_waddr_o}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/vrf_wdata_o}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/vrf_we_o}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/vrf_wbe_o}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/vrf_wvalid_i}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/vrf_id_o}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/vrf_raddr_o}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/vrf_re_o}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/vrf_rdata_i}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/vrf_rvalid_i}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/spatz_mem_req_o}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/spatz_mem_req_valid_o}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/spatz_mem_req_ready_i}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/spatz_mem_rsp_i}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/spatz_mem_rsp_valid_i}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/spatz_mem_finished_o}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/spatz_mem_str_finished_o}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/spatz_req_d}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/mem_spatz_req}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/mem_spatz_req_valid}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/mem_spatz_req_ready}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/mem_is_strided}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/mem_is_indexed}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/state_d}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/state_q}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/store_count_q}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/store_count_d}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/rob_wdata}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/rob_wid}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/rob_push}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/rob_rvalid}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/rob_rdata}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/rob_pop}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/rob_rid}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/rob_req_id}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/rob_id}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/rob_full}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/rob_empty}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/mem_operation_valid}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/mem_operation_last}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/mem_counter_max}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/mem_counter_en}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/mem_counter_load}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/mem_counter_delta}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/mem_counter_d}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/mem_counter_q}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/mem_port_finished_q}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/mem_idx_counter_delta}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/mem_idx_counter_d}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/mem_idx_counter_q}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/mem_insn_finished_q}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/mem_insn_finished_d}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/mem_insn_pending_q}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/mem_insn_pending_d}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/commit_insn_d}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/commit_insn_push}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/commit_insn_q}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/commit_insn_pop}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/commit_insn_empty}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/commit_insn_valid}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/commit_counter_max}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/commit_counter_en}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/commit_counter_load}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/commit_counter_delta}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/commit_counter_d}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/commit_counter_q}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/commit_finished_q}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/commit_finished_d}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/mem_req_addr}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/vd_vreg_addr}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/vs2_vreg_addr}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/vd_elem_id}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/vs2_elem_id_d}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/vs2_elem_id_q}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/pending_index}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/mem_req_addr_offset}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/busy_q}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/busy_d}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/vlsu_finished_req}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/spatz_mem_req}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/spatz_mem_req_valid}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/spatz_mem_req_ready}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/commit_operation_valid}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/commit_operation_last}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/mem_is_load}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/mem_is_vstart_zero}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/mem_is_addr_unaligned}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/mem_is_single_element_operation}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/mem_single_element_size}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/mem_idx_single_element_size}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/commit_is_addr_unaligned}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/commit_is_single_element_operation}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/commit_single_element_size}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/vreg_addr_offset}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/offset_queue_full}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/vrf_req_d}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/vrf_req_q}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/vrf_req_valid_d}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/vrf_req_ready_d}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/vrf_req_valid_q}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/vrf_req_ready_q}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/vreg_start_0}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/catchup}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/mem_req_id}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/mem_req_data}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/mem_req_svalid}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/mem_req_strb}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/mem_req_lvalid}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/mem_req_last}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/mem_pending_d}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/mem_pending_q}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VLSU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vlsu/mem_pending}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vsldu/clk_i}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vsldu/rst_ni}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vsldu/spatz_req_i}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vsldu/spatz_req_valid_i}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vsldu/spatz_req_ready_o}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vsldu/vsldu_rsp_valid_o}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vsldu/vsldu_rsp_o}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vsldu/vrf_waddr_o}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vsldu/vrf_wdata_o}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vsldu/vrf_we_o}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vsldu/vrf_wbe_o}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vsldu/vrf_wvalid_i}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vsldu/vrf_id_o}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vsldu/vrf_raddr_o}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vsldu/vrf_re_o}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vsldu/vrf_rdata_i}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vsldu/vrf_rvalid_i}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vsldu/spatz_req_d}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vsldu/spatz_req}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vsldu/spatz_req_valid}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vsldu/spatz_req_ready}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vsldu/vrf_req_d}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vsldu/vrf_req_q}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vsldu/vrf_req_valid_d}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vsldu/vrf_req_ready_d}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vsldu/vrf_req_valid_q}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vsldu/vrf_req_ready_q}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vsldu/vreg_operations_finished}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vsldu/is_vl_zero}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vsldu/is_slide_up}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vsldu/op_id_q}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vsldu/op_id_d}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vsldu/slide_amount_q}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vsldu/slide_amount_d}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vsldu/prefetch_q}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vsldu/prefetch_d}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vsldu/running_d}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vsldu/running_q}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vsldu/state_q}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vsldu/state_d}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vsldu/new_vsldu_request}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vsldu/new_vsldu_request_q}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vsldu/vreg_counter_en}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vsldu/vreg_counter_delta}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vsldu/vreg_counter_d}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vsldu/vreg_counter_q}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vsldu/vreg_operation_first}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vsldu/vreg_operation_last}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vsldu/vreg_operation_first_q}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vsldu/vreg_operation_first_d}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vsldu/shift_overflow_q}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vsldu/shift_overflow_d}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vsldu/in_elem_offset}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vsldu/in_elem_flipped_offset}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vsldu/data_in}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vsldu/data_out}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vsldu/data_low}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vsldu/data_high}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VSLDU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vsldu/sld_offset_rd}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/clk_i}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/rst_ni}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/hart_id_i}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/spatz_req_i}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/spatz_req_valid_i}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/spatz_req_ready_o}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/vfu_rsp_valid_o}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/vfu_rsp_ready_i}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/vfu_rsp_o}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/vrf_waddr_o}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/vrf_wdata_o}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/vrf_we_o}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/vrf_wbe_o}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/vrf_wvalid_i}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/vrf_id_o}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/vrf_raddr_o}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/vrf_re_o}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/vrf_rdata_i}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/vrf_rvalid_i}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/fpu_status_o}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/spatz_req}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/spatz_req_valid}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/spatz_req_ready}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/vl_q}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/vl_d}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/busy_q}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/busy_d}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/nr_elem_word}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/state_d}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/state_q}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/ipu_result_tag}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/fpu_result_tag}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/result_tag}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/input_tag}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/vstart}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/stall}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/reduction_operand_ready_d}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/reduction_operand_ready_q}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/op1_is_ready}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/op2_is_ready}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/op3_is_ready}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/operands_ready}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/valid_operations}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/pending_results}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/word_issued}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/running_d}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/running_q}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/is_fpu_insn}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/is_fpu_busy}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/is_ipu_busy}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/scalar_result}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/last_request}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/reduction_state_d}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/reduction_state_q}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/reduction_done}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/narrowing_upper_d}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/narrowing_upper_q}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/widening_upper_d}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/widening_upper_q}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/result}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/result_valid}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/result_ready}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/reduction_q}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/reduction_d}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/ipu_result}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/ipu_result_valid}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/ipu_in_ready}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/fpu_result}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/fpu_result_valid}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/fpu_in_ready}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/operand1}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/operand2}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/operand3}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/in_ready}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/reduction_pointer_d}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/reduction_pointer_q}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/reduction_operand_request}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/vreg_wbe}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/vreg_we}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/vreg_r_req}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/vreg_addr_q}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/vreg_addr_d}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/vreg_wdata}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/int_ipu_in_ready}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/int_ipu_operand1}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/int_ipu_operand2}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/int_ipu_operand3}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/int_ipu_result}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/int_ipu_result_tag}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/int_ipu_result_valid}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/int_ipu_result_ready}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/int_ipu_busy}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/ipu_wide_operand1}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/ipu_wide_operand2}
add wave -noupdate -group {core[1]} -expand -group Spatz -group VFU {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/ipu_wide_operand3}
add wave -noupdate -group {core[1]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/clk_i}
add wave -noupdate -group {core[1]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/clk_d2_i}
add wave -noupdate -group {core[1]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/rst_ni}
add wave -noupdate -group {core[1]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/testmode_i}
add wave -noupdate -group {core[1]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/hart_id_i}
add wave -noupdate -group {core[1]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/irq_i}
add wave -noupdate -group {core[1]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/hive_req_o}
add wave -noupdate -group {core[1]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/hive_rsp_i}
add wave -noupdate -group {core[1]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/data_req_o}
add wave -noupdate -group {core[1]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/data_rsp_i}
add wave -noupdate -group {core[1]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/tcdm_req_o}
add wave -noupdate -group {core[1]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/tcdm_rsp_i}
add wave -noupdate -group {core[1]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/axi_dma_req_o}
add wave -noupdate -group {core[1]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/axi_dma_res_i}
add wave -noupdate -group {core[1]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/axi_dma_busy_o}
add wave -noupdate -group {core[1]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/axi_dma_perf_o}
add wave -noupdate -group {core[1]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/axi_dma_events_o}
add wave -noupdate -group {core[1]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/core_events_o}
add wave -noupdate -group {core[1]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/tcdm_addr_base_i}
add wave -noupdate -group {core[1]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/acc_snitch_req}
add wave -noupdate -group {core[1]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/acc_snitch_demux}
add wave -noupdate -group {core[1]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/acc_snitch_resp}
add wave -noupdate -group {core[1]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/acc_demux_snitch}
add wave -noupdate -group {core[1]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/acc_resp}
add wave -noupdate -group {core[1]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/dma_resp}
add wave -noupdate -group {core[1]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/acc_snitch_demux_qvalid}
add wave -noupdate -group {core[1]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/acc_snitch_demux_qready}
add wave -noupdate -group {core[1]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/acc_qvalid}
add wave -noupdate -group {core[1]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/acc_qready}
add wave -noupdate -group {core[1]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/dma_qvalid}
add wave -noupdate -group {core[1]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/dma_qready}
add wave -noupdate -group {core[1]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/acc_pvalid}
add wave -noupdate -group {core[1]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/acc_pready}
add wave -noupdate -group {core[1]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/dma_pvalid}
add wave -noupdate -group {core[1]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/dma_pready}
add wave -noupdate -group {core[1]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/acc_demux_snitch_valid}
add wave -noupdate -group {core[1]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/acc_demux_snitch_ready}
add wave -noupdate -group {core[1]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/fpu_rnd_mode}
add wave -noupdate -group {core[1]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/fpu_fmt_mode}
add wave -noupdate -group {core[1]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/fpu_status}
add wave -noupdate -group {core[1]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/snitch_events}
add wave -noupdate -group {core[1]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/snitch_dreq_d}
add wave -noupdate -group {core[1]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/snitch_dreq_q}
add wave -noupdate -group {core[1]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/merged_dreq}
add wave -noupdate -group {core[1]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/snitch_drsp_d}
add wave -noupdate -group {core[1]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/snitch_drsp_q}
add wave -noupdate -group {core[1]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/merged_drsp}
add wave -noupdate -group {core[1]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/spatz_mem_finished}
add wave -noupdate -group {core[1]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/spatz_mem_str_finished}
add wave -noupdate -group {core[1]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/fp_lsu_mem_req}
add wave -noupdate -group {core[1]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/fp_lsu_mem_rsp}
add wave -noupdate -group {core[1]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/spatz_mem_req}
add wave -noupdate -group {core[1]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/spatz_mem_req_valid}
add wave -noupdate -group {core[1]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/spatz_mem_req_ready}
add wave -noupdate -group {core[1]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/spatz_mem_rsp}
add wave -noupdate -group {core[1]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/spatz_mem_rsp_valid}
add wave -noupdate -group {core[1]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/data_tcdm_req}
add wave -noupdate -group {core[1]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/data_tcdm_rsp}
add wave -noupdate -group {core[1]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/slave_select}
add wave -noupdate -group {core[1]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/addr_map}
add wave -noupdate -group {core[1]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/f}
add wave -noupdate -group {core[1]} -group Internal {/tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/cycle}
add wave -noupdate -expand -group Cluster /tb_bin/i_dut/i_cluster_wrapper/i_cluster/clk_i
add wave -noupdate -expand -group Cluster /tb_bin/i_dut/i_cluster_wrapper/i_cluster/rst_ni
add wave -noupdate -expand -group Cluster /tb_bin/i_dut/i_cluster_wrapper/i_cluster/debug_req_i
add wave -noupdate -expand -group Cluster /tb_bin/i_dut/i_cluster_wrapper/i_cluster/meip_i
add wave -noupdate -expand -group Cluster /tb_bin/i_dut/i_cluster_wrapper/i_cluster/mtip_i
add wave -noupdate -expand -group Cluster /tb_bin/i_dut/i_cluster_wrapper/i_cluster/msip_i
add wave -noupdate -expand -group Cluster /tb_bin/i_dut/i_cluster_wrapper/i_cluster/hart_base_id_i
add wave -noupdate -expand -group Cluster /tb_bin/i_dut/i_cluster_wrapper/i_cluster/cluster_base_addr_i
add wave -noupdate -expand -group Cluster /tb_bin/i_dut/i_cluster_wrapper/i_cluster/axi_core_default_user_i
add wave -noupdate -expand -group Cluster /tb_bin/i_dut/i_cluster_wrapper/i_cluster/cluster_probe_o
add wave -noupdate -expand -group Cluster /tb_bin/i_dut/i_cluster_wrapper/i_cluster/axi_in_req_i
add wave -noupdate -expand -group Cluster /tb_bin/i_dut/i_cluster_wrapper/i_cluster/axi_in_resp_o
add wave -noupdate -expand -group Cluster /tb_bin/i_dut/i_cluster_wrapper/i_cluster/axi_out_req_o
add wave -noupdate -expand -group Cluster /tb_bin/i_dut/i_cluster_wrapper/i_cluster/axi_out_resp_i
add wave -noupdate -expand -group Cluster /tb_bin/i_dut/i_cluster_wrapper/i_cluster/tcdm_start_address
add wave -noupdate -expand -group Cluster /tb_bin/i_dut/i_cluster_wrapper/i_cluster/tcdm_end_address
add wave -noupdate -expand -group Cluster /tb_bin/i_dut/i_cluster_wrapper/i_cluster/cluster_periph_start_address
add wave -noupdate -expand -group Cluster /tb_bin/i_dut/i_cluster_wrapper/i_cluster/cluster_periph_end_address
add wave -noupdate -expand -group Cluster /tb_bin/i_dut/i_cluster_wrapper/i_cluster/narrow_axi_slv_req
add wave -noupdate -expand -group Cluster /tb_bin/i_dut/i_cluster_wrapper/i_cluster/narrow_axi_slv_rsp
add wave -noupdate -expand -group Cluster /tb_bin/i_dut/i_cluster_wrapper/i_cluster/narrow_axi_mst_req
add wave -noupdate -expand -group Cluster /tb_bin/i_dut/i_cluster_wrapper/i_cluster/narrow_axi_mst_rsp
add wave -noupdate -expand -group Cluster /tb_bin/i_dut/i_cluster_wrapper/i_cluster/wide_axi_mst_req
add wave -noupdate -expand -group Cluster /tb_bin/i_dut/i_cluster_wrapper/i_cluster/wide_axi_mst_rsp
add wave -noupdate -expand -group Cluster /tb_bin/i_dut/i_cluster_wrapper/i_cluster/wide_axi_slv_req
add wave -noupdate -expand -group Cluster /tb_bin/i_dut/i_cluster_wrapper/i_cluster/wide_axi_slv_rsp
add wave -noupdate -expand -group Cluster /tb_bin/i_dut/i_cluster_wrapper/i_cluster/ic_req
add wave -noupdate -expand -group Cluster /tb_bin/i_dut/i_cluster_wrapper/i_cluster/ic_rsp
add wave -noupdate -expand -group Cluster /tb_bin/i_dut/i_cluster_wrapper/i_cluster/sb_dma_req
add wave -noupdate -expand -group Cluster /tb_bin/i_dut/i_cluster_wrapper/i_cluster/sb_dma_rsp
add wave -noupdate -expand -group Cluster /tb_bin/i_dut/i_cluster_wrapper/i_cluster/ext_dma_req
add wave -noupdate -expand -group Cluster /tb_bin/i_dut/i_cluster_wrapper/i_cluster/ext_dma_rsp
add wave -noupdate -expand -group Cluster /tb_bin/i_dut/i_cluster_wrapper/i_cluster/axi_soc_req
add wave -noupdate -expand -group Cluster /tb_bin/i_dut/i_cluster_wrapper/i_cluster/axi_soc_rsp
add wave -noupdate -expand -group Cluster /tb_bin/i_dut/i_cluster_wrapper/i_cluster/tcdm_req
add wave -noupdate -expand -group Cluster /tb_bin/i_dut/i_cluster_wrapper/i_cluster/tcdm_rsp
add wave -noupdate -expand -group Cluster /tb_bin/i_dut/i_cluster_wrapper/i_cluster/core_events
add wave -noupdate -expand -group Cluster /tb_bin/i_dut/i_cluster_wrapper/i_cluster/tcdm_events
add wave -noupdate -expand -group Cluster /tb_bin/i_dut/i_cluster_wrapper/i_cluster/dma_events
add wave -noupdate -expand -group Cluster /tb_bin/i_dut/i_cluster_wrapper/i_cluster/icache_events
add wave -noupdate -expand -group Cluster /tb_bin/i_dut/i_cluster_wrapper/i_cluster/core_req
add wave -noupdate -expand -group Cluster /tb_bin/i_dut/i_cluster_wrapper/i_cluster/filtered_core_req
add wave -noupdate -expand -group Cluster /tb_bin/i_dut/i_cluster_wrapper/i_cluster/core_rsp
add wave -noupdate -expand -group Cluster /tb_bin/i_dut/i_cluster_wrapper/i_cluster/filtered_core_rsp
add wave -noupdate -expand -group Cluster /tb_bin/i_dut/i_cluster_wrapper/i_cluster/reg_req
add wave -noupdate -expand -group Cluster /tb_bin/i_dut/i_cluster_wrapper/i_cluster/reg_rsp
add wave -noupdate -expand -group Cluster /tb_bin/i_dut/i_cluster_wrapper/i_cluster/bootrom_reg_req
add wave -noupdate -expand -group Cluster /tb_bin/i_dut/i_cluster_wrapper/i_cluster/bootrom_reg_rsp
add wave -noupdate -expand -group Cluster /tb_bin/i_dut/i_cluster_wrapper/i_cluster/icache_prefetch_enable
add wave -noupdate -expand -group Cluster /tb_bin/i_dut/i_cluster_wrapper/i_cluster/cl_interrupt
add wave -noupdate -expand -group Cluster /tb_bin/i_dut/i_cluster_wrapper/i_cluster/dma_xbar_default_port
add wave -noupdate -expand -group Cluster /tb_bin/i_dut/i_cluster_wrapper/i_cluster/dma_xbar_rule
add wave -noupdate -expand -group Cluster /tb_bin/i_dut/i_cluster_wrapper/i_cluster/ext_dma_req_q_addr_nontrunc
add wave -noupdate -expand -group Cluster /tb_bin/i_dut/i_cluster_wrapper/i_cluster/hive_req
add wave -noupdate -expand -group Cluster /tb_bin/i_dut/i_cluster_wrapper/i_cluster/hive_rsp
add wave -noupdate -expand -group Cluster /tb_bin/i_dut/i_cluster_wrapper/i_cluster/inst_addr
add wave -noupdate -expand -group Cluster /tb_bin/i_dut/i_cluster_wrapper/i_cluster/inst_cacheable
add wave -noupdate -expand -group Cluster /tb_bin/i_dut/i_cluster_wrapper/i_cluster/inst_data
add wave -noupdate -expand -group Cluster /tb_bin/i_dut/i_cluster_wrapper/i_cluster/inst_valid
add wave -noupdate -expand -group Cluster /tb_bin/i_dut/i_cluster_wrapper/i_cluster/inst_ready
add wave -noupdate -expand -group Cluster /tb_bin/i_dut/i_cluster_wrapper/i_cluster/inst_error
add wave -noupdate -expand -group Cluster /tb_bin/i_dut/i_cluster_wrapper/i_cluster/flush_valid
add wave -noupdate -expand -group Cluster /tb_bin/i_dut/i_cluster_wrapper/i_cluster/flush_ready
add wave -noupdate -expand -group Cluster /tb_bin/i_dut/i_cluster_wrapper/i_cluster/core_to_axi_req
add wave -noupdate -expand -group Cluster /tb_bin/i_dut/i_cluster_wrapper/i_cluster/core_to_axi_rsp
add wave -noupdate -expand -group Cluster /tb_bin/i_dut/i_cluster_wrapper/i_cluster/cluster_user
add wave -noupdate -expand -group Cluster /tb_bin/i_dut/i_cluster_wrapper/i_cluster/cluster_xbar_rules
add wave -noupdate -expand -group Cluster /tb_bin/i_dut/i_cluster_wrapper/i_cluster/narrow_axi_slv_req_soc
add wave -noupdate -expand -group Cluster /tb_bin/i_dut/i_cluster_wrapper/i_cluster/narrow_axi_slv_resp_soc
add wave -noupdate -expand -group Cluster /tb_bin/i_dut/i_cluster_wrapper/i_cluster/flat_acc
add wave -noupdate -expand -group Cluster /tb_bin/i_dut/i_cluster_wrapper/i_cluster/flat_con
TreeUpdate [SetDefaultTree]
WaveRestoreCursors {{Cursor 1} {2484000 ps} 0} {{Cursor 2} {2589463 ps} 0}
quietly wave cursor active 2
configure wave -namecolwidth 227
configure wave -valuecolwidth 100
configure wave -justifyvalue left
configure wave -signalnamewidth 1
configure wave -snapdistance 10
configure wave -datasetprefix 0
configure wave -rowmargin 4
configure wave -childrowmargin 2
configure wave -gridoffset 0
configure wave -gridperiod 1
configure wave -griddelta 40
configure wave -timeline 0
configure wave -timelineunits ps
update
WaveRestoreZoom {2585385 ps} {2596629 ps}
bookmark add wave bookmark0 {{0 ps} {2482000 ps}} 0
bookmark add wave bookmark1 {{2571173 ps} {2607901 ps}} 34
