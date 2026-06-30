struct Uniforms {
    width: u32, height: u32, depth: u32, tau: f32,
    // ... matching your SoupUniforms ...
};

@group(0) @binding(0) var<uniform> params: Uniforms;
@group(0) @binding(1) var<storage, read> f_3d: array<f32>;
// A tiny buffer of 4 integers: [sum_x, sum_y, sum_z, total_weight]
@group(0) @binding(2) var<storage, read_write> metrics: array<atomic<u32>>;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.width || id.y >= params.height || id.z >= params.depth) { return; }

    let base_idx = (id.z * params.width * params.height + id.y * params.width + id.x) * 19u;

    // Calculate macroscopic density (rho)
    var rho = 0.0;
    for (var d = 0u; d < 19u; d++) {
        rho += f_3d[base_idx + d];
    }

    // The knot is a high-pressure/low-pressure vortex.
    // Variance from 1.0 indicates the presence of the knot.
    let variance = abs(rho - 1.0);

    // Only track nodes that are significantly part of the knot structure
    if (variance > 0.01) {
        // WGSL doesn't support atomic floats, so we scale the weight into an integer
        let weight = u32(variance * 1000.0);

        // Atomically add to the global Center of Mass sums
        atomicAdd(&metrics[0], id.x * weight);
        atomicAdd(&metrics[1], id.y * weight);
        atomicAdd(&metrics[2], id.z * weight);
        atomicAdd(&metrics[3], weight);
    }
}