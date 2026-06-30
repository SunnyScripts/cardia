// goethe_atom.wgsl

struct Uniforms {
    dx: f32,             // Scaled node spacing
    dt: f32,             // Time step
    g: f32,              // n=2 non-linear fluid drag
    trap_strength: f32,  // The strength of the molecular bond holding the atom
    zpe_variance: f32,   // The Casimir noise amplitude (scaled by S)
    time_seed: f32,      // An advancing time value to animate the random noise
};

@group(0) @binding(0) var<uniform> params: Uniforms;
@group(0) @binding(1) var<storage, read> psi_real_in: array<f32>;
@group(0) @binding(2) var<storage, read> psi_imag_in: array<f32>;
@group(0) @binding(3) var<storage, read_write> psi_real_out: array<f32>;
@group(0) @binding(4) var<storage, read_write> psi_imag_out: array<f32>;

// 1. GPU Pseudo-Random Number Generator (The ZPE Engine)
// GPUs don't have Math.random(). We use a high-speed bitwise hash to generate
// a deterministic but chaotic noise field based on the node's position and time.
fn pcg_hash(seed: u32) -> f32 {
    var state = seed * 747796405u + 2891336453u;
    var word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    let result = (word >> 22u) ^ word;
    // Return a float between -1.0 and 1.0
    return (f32(result) / f32(0xffffffffu)) * 2.0 - 1.0;
}

fn get_idx(x: u32, y: u32, width: u32) -> u32 {
    return y * width + x;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let width = 512u;
    let height = 512u;
    let x = global_id.x;
    let y = global_id.y;

    if (x == 0u || x >= width - 1u || y == 0u || y >= height - 1u) { return; }

    let i = get_idx(x, y, width);
    let real = psi_real_in[i];
    let imag = psi_imag_in[i];

    // 2. The Laplacian (Fluid connectivity)
    let laplacian_real = (psi_real_in[get_idx(x+1u, y, width)] +
                          psi_real_in[get_idx(x-1u, y, width)] +
                          psi_real_in[get_idx(x, y+1u, width)] +
                          psi_real_in[get_idx(x, y-1u, width)] -
                          4.0 * real) / (params.dx * params.dx);

    let laplacian_imag = (psi_imag_in[get_idx(x+1u, y, width)] +
                          psi_imag_in[get_idx(x-1u, y, width)] +
                          psi_imag_in[get_idx(x, y+1u, width)] +
                          psi_imag_in[get_idx(x, y-1u, width)] -
                          4.0 * imag) / (params.dx * params.dx);

    // 3. The Harmonic Trap (The Molecular Bond)
    // Calculate distance from the center of the grid
    let center_x = f32(width) / 2.0;
    let center_y = f32(height) / 2.0;
    let dist_x = (f32(x) - center_x) * params.dx;
    let dist_y = (f32(y) - center_y) * params.dx;
    let r_squared = dist_x * dist_x + dist_y * dist_y;

    // V = 1/2 * k * r^2
    let molecular_trap = 0.5 * params.trap_strength * r_squared;

    // 4. The Micro-Currents (ZPE Noise)
    // Generate a unique random seed for this exact node at this exact millisecond
    let seed = u32(f32(i) * params.time_seed * 13.0);
    let zpe_kick = pcg_hash(seed) * params.zpe_variance;

    // 5. The Cavitation Limit (n=2 Drag)
    let density = (real * real) + (imag * imag);
    let non_linear_drag = params.g * density;

    // 6. Total Hydrodynamic Pressure
    // The node's state is dictated by the Trap, the Noise, and its own Density
    let total_potential = molecular_trap + zpe_kick + non_linear_drag;

    // 7. Gross-Pitaevskii Time Evolution
    let d_real = -0.5 * laplacian_imag + total_potential * imag;
    let d_imag =  0.5 * laplacian_real - total_potential * real;

    psi_real_out[i] = real + d_real * params.dt;
    psi_imag_out[i] = imag + d_imag * params.dt;
}