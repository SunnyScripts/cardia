// compute.wgsl

// The physical constants of your universe
struct Uniforms {
    dx: f32,       // Grid spacing: 1.3e-27
    dt: f32,       // Time step
    g: f32,        // Non-linear coupling constant (The n=2 drag multiplier)
    mass: f32,     // Effective mass of the vacuum acoustic phonon
};

@group(0) @binding(0) var<uniform> params: Uniforms;
@group(0) @binding(1) var<storage, read> psi_real_in: array<f32>;
@group(0) @binding(2) var<storage, read> psi_imag_in: array<f32>;
@group(0) @binding(3) var<storage, read_write> psi_real_out: array<f32>;
@group(0) @binding(4) var<storage, read_write> psi_imag_out: array<f32>;

// Helper function to get 1D index from 2D grid
fn get_idx(x: u32, y: u32, width: u32) -> u32 {
    return y * width + x;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let width = 1024u; // Define your grid resolution
    let x = global_id.x;
    let y = global_id.y;

    if (x == 0u || x >= width - 1u || y == 0u || y >= width - 1u) {
        return; // Handle boundaries later (Absorbing boundary conditions needed)
    }

    let i = get_idx(x, y, width);

    // 1. Read current wave state
    let real = psi_real_in[i];
    let imag = psi_imag_in[i];

    // 2. Calculate the Laplacian (Spatial curvature of the wave)
    // This is how the wave interacts with the grid nodes around it
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

    // 3. The Knockout Punch: The Non-Linear Fluid Drag (Cavitation Trigger)
    // The density of the wave is |psi|^2.
    let density = (real * real) + (imag * imag);
    let non_linear_potential = params.g * density;

    // 4. Evolve the wave using the Gross-Pitaevskii Equation (Euler integration)
    // Real updates based on Imaginary, Imaginary updates based on Real
    let d_real = -0.5 * laplacian_imag + non_linear_potential * imag;
    let d_imag =  0.5 * laplacian_real - non_linear_potential * real;

    psi_real_out[i] = real + d_real * params.dt;
    psi_imag_out[i] = imag + d_imag * params.dt;
}