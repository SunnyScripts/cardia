// macro_double_slit.wgsl

// ... (Uniforms and bindings remain the same as previous 3D shader) ...
@group(0) @binding(3) var<storage, read> env_map: array<f32>;

@compute @workgroup_size(8, 8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // ... (Index calculation and 3D Laplacian remain the same) ...
    let i = get_idx_3d(x, y, z, width, height);

    // 1. Read the Eulerian Geometry (Is this node a wall?)
    let is_wall = env_map[i]; // Returns 1.0 or 0.0

    // 2. Generate the ZPE micro-currents (The Stochastic Skin)
    let seed = u32(f32(i) * params.time_seed * 13.0);
    let zpe_kick = pcg_hash(seed) * params.zpe_variance;

    // 3. Calculate the Effective Potential
    // If is_wall is 0.0, this entire line equals 0.0 (Perfect Vacuum)
    // If is_wall is 1.0, it applies massive drag, modulated by the ZPE jitter
    let effective_wall_potential = is_wall * params.wall_strength * (1.0 + zpe_kick);

    // 4. Calculate Particle Cavitation (n=2 Non-Linearity)
    let density = (real * real) + (imag * imag);
    let non_linear_drag = params.g * density;

    // 5. Total potential acting on this node
    let total_v = effective_wall_potential + non_linear_drag;

    // 6. Evolve the Wave Function
    let d_real = -0.5 * laplacian_imag + total_v * imag;
    let d_imag =  0.5 * laplacian_real - total_v * real;

    psi_real_out[i] = real + d_real * params.dt;
    psi_imag_out[i] = imag + d_imag * params.dt;
}