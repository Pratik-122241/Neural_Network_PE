/**
 * Systolic Array Top Module
 * 
 * This module implements a 2x2 systolic array using the Neural Network Processing
 * Elements (PEs). It demonstrates how the individual PEs can be connected to form
 * a larger computational structure for matrix multiplication acceleration.
 *
 * The systolic array uses weight-stationary dataflow, where weights are loaded
 * into each PE and remain fixed while activations flow through the array.
 * 
 * Author: Claude
 * Date: May 7, 2025
 */

module systolic_array_2x2 #(
    // Data width parameters
    parameter DATA_WIDTH = 8,              // Default data width is 8 bits
    parameter WEIGHT_WIDTH = 8,            // Default weight width is 8 bits
    parameter ACCUM_WIDTH = 24,            // Accumulator width for MAC operations
    parameter FRAC_BITS = 4,               // Number of fractional bits for fixed-point
    
    // Memory parameters
    parameter FIFO_DEPTH = 8,              // Depth of input/output FIFOs
    parameter WEIGHT_BUFFER_DEPTH = 16     // Depth of weight buffer
)(
    // Clock and reset
    input wire clk,                        // System clock
    input wire rst_n,                      // Active low reset
    
    // Control signals
    input wire enable,                     // Enable the array
    input wire [1:0] act_func_sel,         // Activation function selector
    input wire load_weights,               // Signal to load weights
    input wire clear_acc,                  // Clear accumulator signal
    
    // Input data - row activations
    input wire signed [DATA_WIDTH-1:0] act_in_row0,  // Row 0 input activation
    input wire signed [DATA_WIDTH-1:0] act_in_row1,  // Row 1 input activation
    
    // Input weights - 2x2 matrix
    input wire signed [WEIGHT_WIDTH-1:0] weight_00,  // Weight for PE(0,0)
    input wire signed [WEIGHT_WIDTH-1:0] weight_01,  // Weight for PE(0,1)
    input wire signed [WEIGHT_WIDTH-1:0] weight_10,  // Weight for PE(1,0)
    input wire signed [WEIGHT_WIDTH-1:0] weight_11,  // Weight for PE(1,1)
    
    // Handshaking signals
    input wire in_valid,                   // Input data valid
    output wire in_ready,                  // Ready to receive input
    input wire out_ready,                  // Output receiver is ready
    output wire out_valid,                 // Output data is valid
    
    // Output results - column results
    output wire signed [DATA_WIDTH-1:0] result_col0, // Column 0 result
    output wire signed [DATA_WIDTH-1:0] result_col1  // Column 1 result
);

    // Internal connections for activation propagation (horizontal)
    wire signed [DATA_WIDTH-1:0] act_row0_pe0_to_pe1; // Row 0: PE(0,0) to PE(0,1)
    wire signed [DATA_WIDTH-1:0] act_row1_pe0_to_pe1; // Row 1: PE(1,0) to PE(1,1)
    
    // Handshaking signals between PEs
    wire pe00_upstream_ready, pe00_downstream_valid;
    wire pe01_upstream_ready, pe01_downstream_valid;
    wire pe10_upstream_ready, pe10_downstream_valid;
    wire pe11_upstream_ready, pe11_downstream_valid;
    
    // Assign top-level handshaking signals
    assign in_ready = pe00_upstream_ready & pe10_upstream_ready;
    assign out_valid = pe01_downstream_valid & pe11_downstream_valid;
    
    // Dataflow mode - set to weight stationary (00)
    wire [1:0] dataflow_mode = 2'b00;

    // PE(0,0) - Top Left
    neural_network_pe #(
        .DATA_WIDTH(DATA_WIDTH),
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .ACCUM_WIDTH(ACCUM_WIDTH),
        .FRAC_BITS(FRAC_BITS),
        .FIFO_DEPTH(FIFO_DEPTH),
        .WEIGHT_BUFFER_DEPTH(WEIGHT_BUFFER_DEPTH)
    ) pe_0_0 (
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable),
        .act_func_sel(act_func_sel),
        .load_weight(load_weights),
        .clear_acc(clear_acc),
        .forward_output(1'b1),  // Always forward in systolic array
        
        // Data connections
        .activation_in(act_in_row0),
        .weight_in(weight_00),
        .activation_out(act_row0_pe0_to_pe1),
        .result_out(/* Not connected - middle PE */),
        
        // Handshaking
        .upstream_valid(in_valid),
        .upstream_ready(pe00_upstream_ready),
        .downstream_ready(pe01_upstream_ready),
        .downstream_valid(pe00_downstream_valid),
        
        // Configuration
        .dataflow_mode(dataflow_mode)
    );
    
    // PE(0,1) - Top Right
    neural_network_pe #(
        .DATA_WIDTH(DATA_WIDTH),
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .ACCUM_WIDTH(ACCUM_WIDTH),
        .FRAC_BITS(FRAC_BITS),
        .FIFO_DEPTH(FIFO_DEPTH),
        .WEIGHT_BUFFER_DEPTH(WEIGHT_BUFFER_DEPTH)
    ) pe_0_1 (
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable),
        .act_func_sel(act_func_sel),
        .load_weight(load_weights),
        .clear_acc(clear_acc),
        .forward_output(1'b0),  // End of row, no forwarding
        
        // Data connections
        .activation_in(act_row0_pe0_to_pe1),
        .weight_in(weight_01),
        .activation_out(/* Not connected - end of row */),
        .result_out(result_col0),  // Output column 0
        
        // Handshaking
        .upstream_valid(pe00_downstream_valid),
        .upstream_ready(pe01_upstream_ready),
        .downstream_ready(out_ready),
        .downstream_valid(pe01_downstream_valid),
        
        // Configuration
        .dataflow_mode(dataflow_mode)
    );
    
    // PE(1,0) - Bottom Left
    neural_network_pe #(
        .DATA_WIDTH(DATA_WIDTH),
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .ACCUM_WIDTH(ACCUM_WIDTH),
        .FRAC_BITS(FRAC_BITS),
        .FIFO_DEPTH(FIFO_DEPTH),
        .WEIGHT_BUFFER_DEPTH(WEIGHT_BUFFER_DEPTH)
    ) pe_1_0 (
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable),
        .act_func_sel(act_func_sel),
        .load_weight(load_weights),
        .clear_acc(clear_acc),
        .forward_output(1'b1),  // Always forward in systolic array
        
        // Data connections
        .activation_in(act_in_row1),
        .weight_in(weight_10),
        .activation_out(act_row1_pe0_to_pe1),
        .result_out(/* Not connected - middle PE */),
        
        // Handshaking
        .upstream_valid(in_valid),
        .upstream_ready(pe10_upstream_ready),
        .downstream_ready(pe11_upstream_ready),
        .downstream_valid(pe10_downstream_valid),
        
        // Configuration
        .dataflow_mode(dataflow_mode)
    );
    
    // PE(1,1) - Bottom Right
    neural_network_pe #(
        .DATA_WIDTH(DATA_WIDTH),
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .ACCUM_WIDTH(ACCUM_WIDTH),
        .FRAC_BITS(FRAC_BITS),
        .FIFO_DEPTH(FIFO_DEPTH),
        .WEIGHT_BUFFER_DEPTH(WEIGHT_BUFFER_DEPTH)
    ) pe_1_1 (
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable),
        .act_func_sel(act_func_sel),
        .load_weight(load_weights),
        .clear_acc(clear_acc),
        .forward_output(1'b0),  // End of row, no forwarding
        
        // Data connections
        .activation_in(act_row1_pe0_to_pe1),
        .weight_in(weight_11),
        .activation_out(/* Not connected - end of row */),
        .result_out(result_col1),  // Output column 1
        
        // Handshaking
        .upstream_valid(pe10_downstream_valid),
        .upstream_ready(pe11_upstream_ready),
        .downstream_ready(out_ready),
        .downstream_valid(pe11_downstream_valid),
        
        // Configuration
        .dataflow_mode(dataflow_mode)
    );

endmodule