// Warp-Core Performance Benchmark v3.0 (Tiled INT4 GEMM + Correct RMSNorm)
import { createPipeline } from "./warp-core-pipeline.js";

export async function runBenchmark() {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) { throw new Error("WebGPU not supported"); }
    
    // Check for timestamp-query support for high-precision measurement
    const hasTimestamps = adapter.features.has("timestamp-query");
    const device = await adapter.requestDevice({ 
        requiredFeatures: hasTimestamps ? ["timestamp-query"] : [] 
    });

    const dim = 4096;
    const iterations = 100; // Increased for better averaging
    const pipeline = await createPipeline(device);

    // Buffer setups
    const inputBuffer = device.createBuffer({ size: dim * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const weightBuffer = device.createBuffer({ size: (dim * dim / 8) * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const outputBuffer = device.createBuffer({ size: dim * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const paramBuffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    device.queue.writeBuffer(paramBuffer, 0, new Uint32Array([dim]));
    device.queue.writeBuffer(paramBuffer, 4, new Float32Array([1e-5])); // Epsilon
    device.queue.writeBuffer(paramBuffer, 8, new Float32Array([1.0]));  // Scale

    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: inputBuffer } },
            { binding: 1, resource: { buffer: weightBuffer } },
            { binding: 2, resource: { buffer: outputBuffer } },
            { binding: 3, resource: { buffer: paramBuffer } }
        ]
    });

    // Warm-up pass to jit the pipeline
    const warmUpEncoder = device.createCommandEncoder();
    const warmUpPass = warmUpEncoder.beginComputePass();
    warmUpPass.setPipeline(pipeline);
    warmUpPass.setBindGroup(0, bindGroup);
    warmUpPass.dispatchWorkgroups(dim);
    warmUpPass.end();
    device.queue.submit([warmUpEncoder.finish()]);
    await device.queue.onSubmittedWorkDone();

    // Actual Benchmark Loop
    const start = performance.now();
    for (let i = 0; i < iterations; i++) {
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(dim); // Tiled rows
        pass.end();
        device.queue.submit([encoder.finish()]);
    }

    await device.queue.onSubmittedWorkDone();
    const end = performance.now();
    
    // Calculate Metrics
    const totalTime = end - start;
    const avgLatency = totalTime / iterations;
    
    // TFLOPS Math: (2 * Rows * Cols * InnerDim) / (Time_in_seconds * 10^12)
    // For this kernel (GEMV-style): 2 * dim * dim operations per pass
    const opsPerPass = 2 * Math.pow(dim, 2);
    const tflops = (opsPerPass / (avgLatency / 1000)) / 1e12;

    console.log(`Warp-Core v3 Results: ${avgLatency.toFixed(2)}ms, ${tflops.toFixed(2)} TFLOPS`);

    return {
        latency: avgLatency,
        tflops: tflops
    };
}
