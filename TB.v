/**
 * Simplified Testbench for Neural Network PE
 * 
 * This testbench provides basic testing of the Neural Network PE
 * with minimal syntax complexity.
 */

`timescale 1ns/1ps

module neural_network_pe_tb();

    // Test parameters
    parameter DATA_WIDTH = 8;
    parameter WEIGHT_WIDTH = 8;
    parameter ACCUM_WIDTH = 24;
    parameter FRAC_BITS = 4;
    parameter FIFO_DEPTH = 8;
    parameter WEIGHT_BUFFER_DEPTH = 16;
    
    parameter CLK_PERIOD = 10; // 10ns clock period (100MHz)
    
    // Clock and reset signals
    reg clk;
    reg rst_n;
    
    // Control signals
    reg enable;
    reg [1:0] act_func_sel;
    reg load_weight;
    reg clear_acc;
    reg forward_output;
    
    // Data inputs
    reg signed [DATA_WIDTH-1:0] activation_in;
    reg signed [WEIGHT_WIDTH-1:0] weight_in;
    
    // Handshaking signals
    reg upstream_valid;
    wire upstream_ready;
    reg downstream_ready;
    wire downstream_valid;
    
    // Configuration signals
    reg [1:0] dataflow_mode;
    
    // Data outputs
    wire signed [DATA_WIDTH-1:0] activation_out;
    wire signed [DATA_WIDTH-1:0] result_out;
    
    // Test control variables
    integer i;
    integer test_cycle_count;
    
    // Instantiate the Unit Under Test (UUT)
    neural_network_pe #(
        .DATA_WIDTH(DATA_WIDTH),
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .ACCUM_WIDTH(ACCUM_WIDTH),
        .FRAC_BITS(FRAC_BITS),
        .FIFO_DEPTH(FIFO_DEPTH),
        .WEIGHT_BUFFER_DEPTH(WEIGHT_BUFFER_DEPTH)
    ) uut (
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable),
        .act_func_sel(act_func_sel),
        .load_weight(load_weight),
        .clear_acc(clear_acc),
        .forward_output(forward_output),
        .activation_in(activation_in),
        .weight_in(weight_in),
        .upstream_valid(upstream_valid),
        .upstream_ready(upstream_ready),
        .downstream_ready(downstream_ready),
        .downstream_valid(downstream_valid),
        .dataflow_mode(dataflow_mode),
        .activation_out(activation_out),
        .result_out(result_out)
    );
    
    // Clock generation
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end
    
    // Main test procedure
    initial begin
        // Initialize waveform dump for analysis
        $dumpfile("neural_network_pe_tb.vcd");
        $dumpvars(0, neural_network_pe_tb);
        
        // Initialize all signals
        rst_n = 0;
        enable = 0;
        act_func_sel = 0;
        load_weight = 0;
        clear_acc = 0;
        forward_output = 0;
        activation_in = 0;
        weight_in = 0;
        upstream_valid = 0;
        downstream_ready = 1;
        dataflow_mode = 0;
        test_cycle_count = 0;
        
        $display("Starting Neural Network PE Test");
        
        // Apply reset
        #(CLK_PERIOD*5);
        rst_n = 1;
        #(CLK_PERIOD*2);
        
        //-----------------------------------------------
        // Test 1: Basic MAC operation
        //-----------------------------------------------
        $display("Test 1: Basic MAC operation");
        
        // Set configuration
        act_func_sel = 0; // Linear activation
        dataflow_mode = 0; // Weight stationary
        
        // Load weight = 4
        @(posedge clk);
        weight_in = 4;
        load_weight = 1;
        @(posedge clk);
        load_weight = 0;
        
        // Enable processing and clear accumulator
        @(posedge clk);
        enable = 1;
        clear_acc = 1;
        @(posedge clk);
        clear_acc = 0;
        
        // Test with inputs 1, 2, 3
        for (i = 1; i <= 3; i = i + 1) begin
            // Send activation input
            @(posedge clk);
            activation_in = i;
            upstream_valid = 1;
            @(posedge clk);
            upstream_valid = 0;
            
            // Wait for processing
            #(CLK_PERIOD*5);
            
            // Check output
            if (downstream_valid) begin
                $display("Input: %0d, Result: %0d", i, result_out);
            end
        end
        
        //-----------------------------------------------
        // Test 2: ReLU activation
        //-----------------------------------------------
        $display("Test 2: ReLU activation");
        
        // Change to ReLU activation
        @(posedge clk);
        act_func_sel = 1;
        clear_acc = 1;
        @(posedge clk);
        clear_acc = 0;
        
        // Test with positive and negative inputs
        for (i = -2; i <= 2; i = i + 1) begin
            // Send activation input
            @(posedge clk);
            activation_in = i;
            upstream_valid = 1;
            @(posedge clk);
            upstream_valid = 0;
            
            // Wait for processing
            #(CLK_PERIOD*5);
            
            // Check output
            if (downstream_valid) begin
                $display("Input: %0d, ReLU Result: %0d", i, result_out);
            end
        end
        
        //-----------------------------------------------
        // Test 3: Sigmoid activation
        //-----------------------------------------------
        $display("Test 3: Sigmoid activation");
        
        // Change to Sigmoid activation
        @(posedge clk);
        act_func_sel = 2;
        clear_acc = 1;
        @(posedge clk);
        clear_acc = 0;
        
        // Test with values that span the sigmoid range
        for (i = -8; i <= 8; i = i + 4) begin
            // Send activation input
            @(posedge clk);
            activation_in = i;
            upstream_valid = 1;
            @(posedge clk);
            upstream_valid = 0;
            
            // Wait for processing
            #(CLK_PERIOD*5);
            
            // Check output
            if (downstream_valid) begin
                $display("Input: %0d, Sigmoid Result: %0d", i, result_out);
            end
        end
        
        //-----------------------------------------------
        // Test 4: Different dataflow modes
        //-----------------------------------------------
        $display("Test 4: Output stationary dataflow mode");
        
        // Change to output stationary mode
        @(posedge clk);
        act_func_sel = 0; // Back to linear
        dataflow_mode = 1; // Output stationary
        clear_acc = 1;
        @(posedge clk);
        clear_acc = 0;
        
        // Load multiple weights
        for (i = 2; i <= 6; i = i + 2) begin
            @(posedge clk);
            weight_in = i;
            load_weight = 1;
            @(posedge clk);
            load_weight = 0;
        end
        
        // Send same activation multiple times to see different weights used
        for (i = 0; i < 3; i = i + 1) begin
            @(posedge clk);
            activation_in = 5;
            upstream_valid = 1;
            @(posedge clk);
            upstream_valid = 0;
            
            // Wait for processing
            #(CLK_PERIOD*5);
            
            // Check output
            if (downstream_valid) begin
                $display("Using weight #%0d, Result: %0d", i+1, result_out);
            end
        end
        
        // Disable processing and finish
        @(posedge clk);
        enable = 0;
        
        #(CLK_PERIOD*10);
        $display("All tests completed");
        $finish;
    end

endmodule