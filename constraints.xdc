###############################################################################
# Neural Network Processing Element (PE) Timing Constraints
# 
# This XDC file defines timing and physical constraints for the Neural Network PE
# for implementation on FPGA using Vivado.
###############################################################################

# Clock definition
# 100 MHz system clock (10 ns period)
create_clock -period 10.000 -name clk -waveform {0.000 5.000} [get_ports clk]

# Clock uncertainty and jitter
set_clock_uncertainty 0.200 [get_clocks clk]

# Input delay constraints
# Assuming inputs arrive 2 ns after clock edge
set_input_delay -clock [get_clocks clk] -max 2.000 [get_ports {rst_n}]
set_input_delay -clock [get_clocks clk] -max 2.000 [get_ports {enable}]
set_input_delay -clock [get_clocks clk] -max 2.000 [get_ports {act_func_sel*}]
set_input_delay -clock [get_clocks clk] -max 2.000 [get_ports {load_weight}]
set_input_delay -clock [get_clocks clk] -max 2.000 [get_ports {clear_acc}]
set_input_delay -clock [get_clocks clk] -max 2.000 [get_ports {forward_output}]
set_input_delay -clock [get_clocks clk] -max 2.000 [get_ports {activation_in*}]
set_input_delay -clock [get_clocks clk] -max 2.000 [get_ports {weight_in*}]
set_input_delay -clock [get_clocks clk] -max 2.000 [get_ports {upstream_valid}]
set_input_delay -clock [get_clocks clk] -max 2.000 [get_ports {downstream_ready}]
set_input_delay -clock [get_clocks clk] -max 2.000 [get_ports {dataflow_mode*}]

# Output delay constraints
# Outputs must be valid within 2 ns after clock edge
set_output_delay -clock [get_clocks clk] -max 2.000 [get_ports {upstream_ready}]
set_output_delay -clock [get_clocks clk] -max 2.000 [get_ports {downstream_valid}]
set_output_delay -clock [get_clocks clk] -max 2.000 [get_ports {activation_out*}]
set_output_delay -clock [get_clocks clk] -max 2.000 [get_ports {result_out*}]

# False path constraints
# Asynchronous reset should be treated as a false path for timing analysis
set_false_path -from [get_ports rst_n] -to [all_registers]

# Maximum delay constraints for critical paths
# MAC operation has the critical path, allow 8 ns
set_max_delay 8.000 -from [get_pins */pipe_activation_in_reg*/C] -to [get_pins */accumulator_reg*/D]
set_max_delay 8.000 -from [get_pins */pipe_weight_reg*/C] -to [get_pins */accumulator_reg*/D]

# Multicycle paths
# Activation functions computed over multiple cycles
set_multicycle_path -setup 2 -from [get_pins */mac_result_reg*/C] -to [get_pins */activation_result_reg*/D]
set_multicycle_path -hold 1 -from [get_pins */mac_result_reg*/C] -to [get_pins */activation_result_reg*/D]

# Physical constraints (comment/adjust based on your FPGA board)
# Example for Xilinx Artix-7 on Basys3 board
# System clock
#set_property PACKAGE_PIN W5 [get_ports clk]
#set_property IOSTANDARD LVCMOS33 [get_ports clk]

# Reset signal 
#set_property PACKAGE_PIN U18 [get_ports rst_n]
#set_property IOSTANDARD LVCMOS33 [get_ports rst_n]

# Configuration and analysis constraints
# Report timing on critical paths
report_timing -from [get_pins */pipe_activation_in_reg*/C] -to [get_pins */accumulator_reg*/D] -max_paths 10
report_timing -from [get_pins */mac_result_reg*/C] -to [get_pins */activation_result_reg*/D] -max_paths 10

###############################################################################
# Note: Physical constraints should be uncommented and adjusted based on your
# specific FPGA board and pin assignment requirements.
###############################################################################