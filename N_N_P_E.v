/**
 * Neural Network Processing Element (PE) for Systolic Array Architecture
 * 
 * This module implements a configurable neural network processing element
 * designed for use in a systolic array architecture for matrix multiplication
 * acceleration. The PE supports parameterized bit widths, signed fixed-point
 * arithmetic, and various activation functions.
 *
 * Key Features:
 * - Configurable precision MAC unit with parameterized bit width
 * - Pipelined architecture for high throughput
 * - Support for ReLU and sigmoid activation functions
 * - Input buffers for weights and activations
 * - Handshaking signals for system integration
 * - Scalable design for array configurations
 * - Support for weight stationary dataflow
 */

`timescale 1ns/1ps

module neural_network_pe #(
    // Data width parameters
    parameter DATA_WIDTH = 8,              // Default data width is 8 bits
    parameter WEIGHT_WIDTH = 8,            // Default weight width is 8 bits
    parameter ACCUM_WIDTH = 24,            // Accumulator width for MAC operations
    parameter FRAC_BITS = 4,               // Number of fractional bits for fixed-point
    
    // Memory parameters
    parameter FIFO_DEPTH = 8,              // Depth of input/output FIFOs
    parameter WEIGHT_BUFFER_DEPTH = 16,    // Depth of weight buffer
    
    // Control parameters
    parameter ACT_FUNC_SEL_WIDTH = 2       // Activation function selection width
)(
    // Clock and reset
    input wire clk,                        // System clock
    input wire rst_n,                      // Active low reset
    
    // Control signals
    input wire enable,                     // Enable the PE
    input wire [ACT_FUNC_SEL_WIDTH-1:0] act_func_sel, // Activation function selector
    input wire load_weight,                // Signal to load weight
    input wire clear_acc,                  // Clear accumulator signal
    input wire forward_output,             // Forward output to next PE
    
    // Data inputs
    input wire signed [DATA_WIDTH-1:0] activation_in,  // Input activation
    input wire signed [WEIGHT_WIDTH-1:0] weight_in,    // Input weight for loading
    
    // Handshaking signals
    input wire upstream_valid,             // Upstream data valid
    output reg upstream_ready,             // Ready to receive from upstream
    input wire downstream_ready,           // Downstream ready to receive
    output reg downstream_valid,           // Output data valid
    
    // Configuration signals
    input wire [1:0] dataflow_mode,        // 00: weight stationary, 01: output stationary, 10: input stationary
    
    // Data outputs
    output reg signed [DATA_WIDTH-1:0] activation_out, // Output activation to next PE
    output reg signed [DATA_WIDTH-1:0] result_out      // Output result after activation
);

    // Internal signals
    reg signed [WEIGHT_WIDTH-1:0] weight_reg;          // Stored weight
    reg signed [DATA_WIDTH-1:0] activation_reg;        // Stored activation
    reg signed [ACCUM_WIDTH-1:0] accumulator;          // MAC accumulator
    reg signed [ACCUM_WIDTH-1:0] mac_result;           // Result of MAC operation
    reg signed [DATA_WIDTH-1:0] activation_result;     // Result after activation function
    reg signed [DATA_WIDTH-1:0] fifo_write_data;       // Data to write to output FIFO
    
    // State machine states
    localparam IDLE = 2'b00;
    localparam COMPUTE = 2'b01;
    localparam ACTIVATE = 2'b10;
    localparam OUTPUT_DATA = 2'b11;
    
    reg [1:0] current_state;
    reg [1:0] next_state;
    reg [1:0] prev_state;                              // Track previous state for FIFO timing
    
    // Activation function selection
    localparam ACT_NONE = 2'b00;     // No activation (linear)
    localparam ACT_RELU = 2'b01;     // ReLU activation
    localparam ACT_SIGMOID = 2'b10;  // Sigmoid activation
    localparam ACT_TANH = 2'b11;     // Tanh activation (for future expansion)
    
    // Pipeline registers - ensure they're all properly signed
    reg signed [DATA_WIDTH-1:0] pipe_activation_in;
    reg signed [WEIGHT_WIDTH-1:0] pipe_weight;
    
    // Input activation FIFO buffer
    reg [3:0] act_fifo_count;                        // Count of elements in activation FIFO
    reg signed [DATA_WIDTH-1:0] act_fifo [0:FIFO_DEPTH-1]; // Activation FIFO storage
    reg [2:0] act_fifo_rd_ptr;                       // Read pointer for activation FIFO
    reg [2:0] act_fifo_wr_ptr;                       // Write pointer for activation FIFO
    
    // Weight buffer
    reg signed [WEIGHT_WIDTH-1:0] weight_buffer [0:WEIGHT_BUFFER_DEPTH-1]; // Weight buffer storage
    reg [3:0] weight_buffer_ptr;                     // Pointer for weight buffer
    
    // Output FIFO
    reg [3:0] out_fifo_count;                        // Count of elements in output FIFO
    reg signed [DATA_WIDTH-1:0] out_fifo [0:FIFO_DEPTH-1]; // Output FIFO storage
    reg [2:0] out_fifo_rd_ptr;                       // Read pointer for output FIFO
    reg [2:0] out_fifo_wr_ptr;                       // Write pointer for output FIFO
    
    // Flags
    reg activation_fifo_full;                        // Flag for full activation FIFO
    reg activation_fifo_empty;                       // Flag for empty activation FIFO
    reg output_fifo_full;                            // Flag for full output FIFO
    reg output_fifo_empty;                           // Flag for empty output FIFO
    
    // Debug flags
    reg signed [ACCUM_WIDTH-1:0] tmp_mult;           // Temporary storage for debug
    
    // Compute multiplication result - explicit signed expression to preserve sign
    wire signed [ACCUM_WIDTH-1:0] mult_result;
    assign mult_result = (pipe_activation_in * pipe_weight);
    
    // Sigmoid approximation LUT
    reg [DATA_WIDTH-1:0] sigmoid_lut [0:15];         // Sigmoid lookup table
    
    // Sigmoid index for lookup
    reg [3:0] sigmoid_idx;
    reg [3:0] sigmoid_idx_reg; // Pre-computed index for better timing
    
    // Initialize sigmoid LUT with approximated values
    initial begin
        // Approximated sigmoid values for inputs from -8 to +7 (scaled to DATA_WIDTH)
        sigmoid_lut[0] = 0;           // Very negative input -> ~0
        sigmoid_lut[1] = 13;          // ~0.05
        sigmoid_lut[2] = 26;          // ~0.1
        sigmoid_lut[3] = 51;          // ~0.2
        sigmoid_lut[4] = 77;          // ~0.3
        sigmoid_lut[5] = 102;         // ~0.4
        sigmoid_lut[6] = 128;         // ~0.5 (midpoint)
        sigmoid_lut[7] = 153;         // ~0.6
        sigmoid_lut[8] = 179;         // ~0.7
        sigmoid_lut[9] = 204;         // ~0.8
        sigmoid_lut[10] = 230;        // ~0.9
        sigmoid_lut[11] = 243;        // ~0.95
        sigmoid_lut[12] = 250;        // ~0.98
        sigmoid_lut[13] = 253;        // ~0.99
        sigmoid_lut[14] = 255;        // ~1.0
        sigmoid_lut[15] = 255;        // Very positive input -> ~1
    end
    
    // Initialize important registers to avoid X values
    initial begin
        current_state = IDLE;
        next_state = IDLE;
        prev_state = IDLE;
        pipe_activation_in = 0;
        pipe_weight = 0;
        accumulator = 0;
        mac_result = 0;
        activation_result = 0;
        fifo_write_data = 0;
        act_fifo_count = 0;
        act_fifo_rd_ptr = 0;
        act_fifo_wr_ptr = 0;
        weight_buffer_ptr = 0;
        out_fifo_count = 0;
        out_fifo_rd_ptr = 0;
        out_fifo_wr_ptr = 0;
        upstream_ready = 1;           // Start ready to receive
        downstream_valid = 0;
        activation_out = 0;
        result_out = 0;
        tmp_mult = 0;
        sigmoid_idx = 0;
        sigmoid_idx_reg = 0;
    end
    
    // State register with previous state tracking
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            current_state <= IDLE;
            prev_state <= IDLE;
        end else begin
            prev_state <= current_state;
            current_state <= next_state;
        end
    end
    
    // Next state logic
    always @(*) begin
        case (current_state)
            IDLE: begin
                if (enable && !activation_fifo_empty)
                    next_state = COMPUTE;
                else
                    next_state = IDLE;
            end
            
            COMPUTE: begin
                next_state = ACTIVATE;
            end
            
            ACTIVATE: begin
                if (!output_fifo_full)
                    next_state = OUTPUT_DATA;
                else
                    next_state = ACTIVATE;
            end
            
            OUTPUT_DATA: begin
                if (!activation_fifo_empty && enable)
                    next_state = COMPUTE;
                else
                    next_state = IDLE;
            end
            
            default: next_state = IDLE;
        endcase
    end
    
    // FIFO status flags
    always @(*) begin
        activation_fifo_full = (act_fifo_count == FIFO_DEPTH);
        activation_fifo_empty = (act_fifo_count == 0);
        output_fifo_full = (out_fifo_count == FIFO_DEPTH);
        output_fifo_empty = (out_fifo_count == 0);
        
        // Upstream ready when activation FIFO is not full
        upstream_ready = !activation_fifo_full;
        
        // Downstream valid when output FIFO is not empty
        downstream_valid = !output_fifo_empty;
    end
    
    // Input activation FIFO management
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            act_fifo_rd_ptr <= 0;
            act_fifo_wr_ptr <= 0;
            act_fifo_count <= 0;
        end else begin
            // Write to activation FIFO when upstream valid and not full
            if (upstream_valid && !activation_fifo_full) begin
                act_fifo[act_fifo_wr_ptr] <= activation_in;
                act_fifo_wr_ptr <= (act_fifo_wr_ptr == FIFO_DEPTH-1) ? 0 : act_fifo_wr_ptr + 1;
                act_fifo_count <= act_fifo_count + 1;
                
                // Debug message with explicit sign indication
                $display("Time %0t: Captured input data %0d (signed: %0d)", 
                         $time, activation_in, $signed(activation_in));
            end
        end
    end
    
    // IMPORTANT: Modified to pre-load data in transition to COMPUTE state
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pipe_activation_in <= 0;
        end else if (current_state == IDLE && next_state == COMPUTE && !activation_fifo_empty) begin
            // Preload the next activation value right before COMPUTE state
            pipe_activation_in <= act_fifo[act_fifo_rd_ptr];
            
            // Update read pointer and count
            act_fifo_rd_ptr <= (act_fifo_rd_ptr == FIFO_DEPTH-1) ? 0 : act_fifo_rd_ptr + 1;
            act_fifo_count <= act_fifo_count - 1;
            
            // Debug message
            $display("Time %0t: Processing input data %0d (signed: %0d)", 
                     $time, act_fifo[act_fifo_rd_ptr], $signed(act_fifo[act_fifo_rd_ptr]));
        end
    end
    
    // Weight management with improved debugging
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            weight_buffer_ptr <= 0;
            pipe_weight <= 0;
        end else begin
            // Load weight into buffer when requested
            if (load_weight) begin
                weight_buffer[weight_buffer_ptr] <= weight_in;
                $display("Time %0t: Loaded weight %0d at position %0d", 
                         $time, $signed(weight_in), weight_buffer_ptr);
                weight_buffer_ptr <= (weight_buffer_ptr == WEIGHT_BUFFER_DEPTH-1) ? 0 : weight_buffer_ptr + 1;
            end
            
            // Handle different dataflow modes - moved to match pipe_activation_in timing
            if (current_state == IDLE && next_state == COMPUTE) begin
                case (dataflow_mode)
                    2'b00: begin  // Weight stationary
                        pipe_weight <= weight_buffer[0]; // Always use first weight
                        $display("Time %0t: Using weight %0d from position 0 (weight stationary)", 
                                $time, $signed(weight_buffer[0]));
                    end
                    
                    2'b01: begin  // Output stationary
                        // Use weight based on position
                        pipe_weight <= weight_buffer[weight_buffer_ptr % WEIGHT_BUFFER_DEPTH]; 
                        $display("Time %0t: Using weight %0d from position %0d (output stationary)", 
                                $time, $signed(weight_buffer[weight_buffer_ptr % WEIGHT_BUFFER_DEPTH]), 
                                weight_buffer_ptr % WEIGHT_BUFFER_DEPTH);
                    end
                    
                    2'b10: begin  // Input stationary
                        // Use direct input weight
                        pipe_weight <= weight_in;
                        $display("Time %0t: Using direct weight %0d (input stationary)", 
                                $time, $signed(weight_in));
                    end
                    
                    default: begin
                        pipe_weight <= weight_buffer[0];
                        $display("Time %0t: Using default weight %0d from position 0", 
                                $time, $signed(weight_buffer[0]));
                    end
                endcase
            end
        end
    end
    
    // MAC operation (pipelined) with explicit sign handling and detailed debug
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            accumulator <= 0;
            mac_result <= 0;
            tmp_mult <= 0;
        end else begin
            // Clear accumulator when requested
            if (clear_acc) begin
                accumulator <= 0;
                mac_result <= 0;
                $display("Time %0t: Accumulator cleared", $time);
            end
            
            // COMPUTE state - perform multiplication and accumulation
            if (current_state == COMPUTE) begin
                // Store multiplication in temp register for debugging
                tmp_mult <= pipe_activation_in * pipe_weight;
                
                // Debug the multiplication with sign information
                $display("Time %0t: Multiply %0d * %0d = %0d", 
                        $time, $signed(pipe_activation_in), $signed(pipe_weight), 
                        $signed(pipe_activation_in * pipe_weight));
                
                // Ensure sign extension by using signed addition
                accumulator <= $signed(accumulator) + $signed(pipe_activation_in * pipe_weight);
                mac_result <= $signed(accumulator) + $signed(pipe_activation_in * pipe_weight);
                
                // Debug accumulation result
                $display("Time %0t: Accumulator value: %0d -> %0d", 
                        $time, $signed(accumulator), 
                        $signed(accumulator + (pipe_activation_in * pipe_weight)));
                
                // Pre-compute sigmoid index for next cycle if using sigmoid
                if (act_func_sel == ACT_SIGMOID) begin
                    if ($signed(accumulator) + $signed(pipe_activation_in * pipe_weight) <= (-8 << FRAC_BITS))
                        sigmoid_idx_reg <= 0;
                    else if ($signed(accumulator) + $signed(pipe_activation_in * pipe_weight) >= (7 << FRAC_BITS))
                        sigmoid_idx_reg <= 15;
                    else
                        sigmoid_idx_reg <= ((($signed(accumulator) + $signed(pipe_activation_in * pipe_weight)) >>> FRAC_BITS) + 8) & 4'hF;
                end
            end
        end
    end
    
    // ACTIVATE state - apply activation function
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            activation_result <= 0;
        end else if (current_state == ACTIVATE) begin
            // Debug raw MAC result before activation
            $display("Time %0t: Pre-activation MAC result = %0d", $time, $signed(mac_result));
            
            case (act_func_sel)
                ACT_NONE: begin
                    // Linear (no activation) - scale back to DATA_WIDTH
                    if ($signed(mac_result) > 0) begin
                        // Positive case - clamp to max value if needed
                        if ($signed(mac_result) > ((1 << (DATA_WIDTH-1)) - 1)) 
                            activation_result <= (1 << (DATA_WIDTH-1)) - 1;
                        else
                            activation_result <= mac_result[DATA_WIDTH-1:0];
                    end else begin
                        // Negative case - clamp to min value if needed
                        if ($signed(mac_result) < -(1 << (DATA_WIDTH-1)))
                            activation_result <= -(1 << (DATA_WIDTH-1));
                        else
                            activation_result <= mac_result[DATA_WIDTH-1:0];
                    end
                    
                    // Debug message with actual value
                    $display("Time %0t: Linear activation result: %0d", $time, 
                            ($signed(mac_result) > 0) ? 
                              (($signed(mac_result) > ((1 << (DATA_WIDTH-1)) - 1)) ? 
                                (1 << (DATA_WIDTH-1)) - 1 : mac_result[DATA_WIDTH-1:0]) :
                              (($signed(mac_result) < -(1 << (DATA_WIDTH-1))) ? 
                                -(1 << (DATA_WIDTH-1)) : mac_result[DATA_WIDTH-1:0]));
                end
                
                ACT_RELU: begin
                    // ReLU activation: max(0, x)
                    // For debugging, output raw data
                    $display("Time %0t: ReLU input = %0d, sign bit = %0b", 
                            $time, $signed(mac_result), mac_result[ACCUM_WIDTH-1]);
                    
                    // Direct check for negative values by testing sign bit
                    if (mac_result[ACCUM_WIDTH-1]) begin
                        // Negative input, ReLU = 0
                        activation_result <= 0;
                        $display("Time %0t: ReLU activation result: 0 (negative input: %0d)", 
                                $time, $signed(mac_result));
                    end else begin
                        // Positive input (or zero)
                        if ($signed(mac_result) > ((1 << (DATA_WIDTH-1)) - 1)) begin
                            // Clamp to max positive value
                            activation_result <= (1 << (DATA_WIDTH-1)) - 1;
                            $display("Time %0t: ReLU activation result: %0d (clamped from %0d)", 
                                    $time, (1 << (DATA_WIDTH-1)) - 1, $signed(mac_result));
                        end else begin
                            // Within range, use value directly
                            activation_result <= mac_result[DATA_WIDTH-1:0];
                            $display("Time %0t: ReLU activation result: %0d", 
                                    $time, $signed(mac_result[DATA_WIDTH-1:0]));
                        end
                    end
                end
                
                ACT_SIGMOID: begin
                    // Use pre-computed sigmoid index from COMPUTE state
                    activation_result <= sigmoid_lut[sigmoid_idx_reg];
                    $display("Time %0t: Sigmoid activation result: %0d (index %0d)", $time, 
                            sigmoid_lut[sigmoid_idx_reg], sigmoid_idx_reg);
                end
                
                default: begin
                    // Default to linear activation with proper scaling
                    if ($signed(mac_result) > 0) begin
                        if ($signed(mac_result) > ((1 << (DATA_WIDTH-1)) - 1)) 
                            activation_result <= (1 << (DATA_WIDTH-1)) - 1;
                        else
                            activation_result <= mac_result[DATA_WIDTH-1:0];
                    end else begin
                        if ($signed(mac_result) < -(1 << (DATA_WIDTH-1)))
                            activation_result <= -(1 << (DATA_WIDTH-1));
                        else
                            activation_result <= mac_result[DATA_WIDTH-1:0];
                    end
                end
            endcase
            
            // Store the previous cycle's activation result for FIFO writing
            if (prev_state == ACTIVATE) begin
                fifo_write_data <= activation_result;
            end
        end
    end
    
    // MODIFIED: Fixed output FIFO management with delayed activation result
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_fifo_rd_ptr <= 0;
            out_fifo_wr_ptr <= 0;
            out_fifo_count <= 0;
            result_out <= 0;
            fifo_write_data <= 0;
        end else begin
            // Special handling for the first activation result
            if (prev_state == COMPUTE && current_state == ACTIVATE) begin
                fifo_write_data <= 0; // Initialize for first cycle
            end
            
            // Update fifo_write_data at the end of ACTIVATE state
            if (current_state == ACTIVATE && next_state == OUTPUT_DATA) begin
                fifo_write_data <= activation_result;
            end
            
            // Write to output FIFO in OUTPUT_DATA state
            if (current_state == OUTPUT_DATA && !output_fifo_full) begin
                out_fifo[out_fifo_wr_ptr] <= fifo_write_data;
                out_fifo_wr_ptr <= (out_fifo_wr_ptr == FIFO_DEPTH-1) ? 0 : out_fifo_wr_ptr + 1;
                out_fifo_count <= out_fifo_count + 1;
                
                $display("Time %0t: Result %0d added to output FIFO", $time, $signed(fifo_write_data));
            end
            
            // Read from output FIFO when downstream is ready
            if (downstream_ready && !output_fifo_empty) begin
                result_out <= out_fifo[out_fifo_rd_ptr];
                out_fifo_rd_ptr <= (out_fifo_rd_ptr == FIFO_DEPTH-1) ? 0 : out_fifo_rd_ptr + 1;
                out_fifo_count <= out_fifo_count - 1;
                
                $display("Time %0t: Sending output data %0d", $time, $signed(out_fifo[out_fifo_rd_ptr]));
            end
        end
    end
    
    // Forward activation to next PE (for systolic array)
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            activation_out <= 0;
        end else if (forward_output && current_state == OUTPUT_DATA) begin
            // In a systolic array, forward the current activation to the next PE
            activation_out <= pipe_activation_in;
        end
    end
    
    // Debug state transitions
    always @(posedge clk) begin
        if (enable) begin
            case (current_state)
                IDLE: $display("Time %0t: State = IDLE", $time);
                COMPUTE: $display("Time %0t: State = COMPUTE", $time);
                ACTIVATE: $display("Time %0t: State = ACTIVATE", $time);
                OUTPUT_DATA: $display("Time %0t: State = OUTPUT_DATA", $time);
            endcase
        end
    end

endmodule