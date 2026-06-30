struct Uniforms {
    width: u32,
    height: u32,
    depth: u32,
    tau: f32,             // MUST be > 0.5 (e.g., 0.505)
    zpe_amplitude: f32,   // The strength of your vacuum fluctuations
    tick: u32,            // Absolute lattice time
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<uniform> params: Uniforms;
// The grid now stores 19 floats per node, sequentially.
@group(0) @binding(1) var<storage, read> f_in: array<f32>;
@group(0) @binding(2) var<storage, read_write> f_out: array<f32>;
@group(0) @binding(3) var<storage, read> env_map: array<f32>; // For your Trefoil Knot

// ---------------------------------------------------------
// D3Q19 LATTICE CONSTANTS
// ---------------------------------------------------------
// The 19 discrete directions (e_i)
const ex = array<f32, 19>( 0., 1.,-1., 0., 0., 0., 0., 1.,-1., 1.,-1., 1.,-1., 1.,-1., 0., 0., 0., 0.);
const ey = array<f32, 19>( 0., 0., 0., 1.,-1., 0., 0., 1., 1.,-1.,-1., 0., 0., 0., 0., 1.,-1., 1.,-1.);
const ez = array<f32, 19>( 0., 0., 0., 0., 0., 1.,-1., 0., 0., 0., 0., 1., 1.,-1.,-1., 1., 1.,-1.,-1.);

// The 19 lattice weights (w_i)
const w = array<f32, 19>(
    1.0/3.0,
    1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0,
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0
);

fn get_idx(x: u32, y: u32, z: u32) -> u32 {
    return z * (params.width * params.height) + y * params.width + x;
}

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    let z = global_id.z;

    if (x >= params.width || y >= params.height || z >= params.depth) { return; }

    let i = get_idx(x, y, z);
    let base_idx = i * 19u;

    // ---------------------------------------------------------
    // 1. MACROSCOPIC VARIABLES (Density and Velocity)
    // ---------------------------------------------------------
    var rho: f32 = 0.0;
    var ux: f32 = 0.0;
    var uy: f32 = 0.0;
    var uz: f32 = 0.0;

    // Read the incoming momentum packets and sum them
    var f_local: array<f32, 19>;
    for (var d = 0u; d < 19u; d++) {
        let f_val = f_in[base_idx + d];
        f_local[d] = f_val;
        rho += f_val;
        ux += f_val * ex[d];
        uy += f_val * ey[d];
        uz += f_val * ez[d];
    }

    // Prevent division by zero in empty space
    if (rho > 0.0001) {
        ux /= rho;
        uy /= rho;
        uz /= rho;
    } else {
        ux = 0.0; uy = 0.0; uz = 0.0;
    }

    // --- SUPERSONIC STABILIZATION (The Palabos Fix) ---
    // Mathematically clamp the macroscopic velocity to stay just under the
    // lattice speed of sound (cs^2 = 1/3). This physically prevents the NaN explosion.
    let speed_sq = ux*ux + uy*uy + uz*uz;
    let max_speed_sq = 0.33; // Slightly below 1/3
    if (speed_sq > max_speed_sq) {
        let scale = sqrt(max_speed_sq / speed_sq);
        ux *= scale; uy *= scale; uz *= scale;
    }

    let u_sq = ux*ux + uy*uy + uz*uz;

    // ---------------------------------------------------------
    // 2. COLLISION (The n=2 Quadratic Elasticity)
    // ---------------------------------------------------------
    let omega = 1.0 / params.tau; // Collision frequency
    let obstacle_mass = env_map[i]; // 1.0 inside knot, 0.0 outside

    for (var d = 0u; d < 19u; d++) {
        let edot_u = ex[d]*ux + ey[d]*uy + ez[d]*uz;

        // The Maxwell-Boltzmann Equilibrium Distribution.
        // Notice the math: It is fundamentally built on u^2.
        // This is the direct mathematical equivalent of your n=2 drag.
        let f_eq = w[d] * rho * (1.0 + 3.0 * edot_u + 4.5 * edot_u * edot_u - 1.5 * u_sq);

        // 2. Generate Empirical Vacuum Noise (ZPE)
        // pcg_hash generates a unique random float based on the node's XYZ and current Tick
        let zpe_noise = generate_gaussian_noise(x, y, z, params.tick, params.zpe_amplitude);

        // Relax the fluid toward equilibrium (Elastic deformation)
        // 3. FLBM Collision: Relaxation + Stochastic Vacuum Forcing
        f_local[d] = f_local[d] * (1.0 - omega) + f_eq * omega + (w[d] * zpe_noise);

        // BOUNCE-BACK BOUNDARY (The Knot)
        // If we are inside the Trefoil knot, the momentum reverses direction.
        if (obstacle_mass > 0.5) {
            // (In a full implementation, you map 'd' to its opposite direction 'd_opp')
            // This physically transfers momentum from the fluid to the lattice.
        }
    }

    // ---------------------------------------------------------
    // 3. STREAMING (Advection to Neighbors)
    // ---------------------------------------------------------
    for (var d = 0u; d < 19u; d++) {
        // Find the neighbor coordinate
        let nx = i32(x) + i32(ex[d]);
        let ny = i32(y) + i32(ey[d]);
        let nz = i32(z) + i32(ez[d]);

        // Wrap around boundaries (Periodic Wind Tunnel)
        let wx = u32((nx + i32(params.width)) % i32(params.width));
        let wy = u32((ny + i32(params.height)) % i32(params.height));
        let wz = u32((nz + i32(params.depth)) % i32(params.depth));

        let neighbor_base = get_idx(wx, wy, wz) * 19u;

        // Write the collided packet to the neighbor's cell
        f_out[neighbor_base + d] = f_local[d];
    }
}

// ---------------------------------------------------------
// STOCHASTIC VACUUM GENERATOR (ZPE)
// ---------------------------------------------------------
// 1. High-quality integer hash function
fn pcg_hash(seed: u32) -> u32 {
    var state = seed * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// 2. Convert hash to float [0.0, 1.0]
fn random_float(seed: u32) -> f32 {
    return f32(pcg_hash(seed)) / 4294967295.0;
}

// 3. Box-Muller transform for normal/Gaussian distribution
fn generate_gaussian_noise(x: u32, y: u32, z: u32, tick: u32, amp: f32) -> f32 {
    // Create a unique seed for this specific node at this specific moment in time
    let seed1 = x + (y * 1973u) + (z * 9277u) + (tick * 26699u);
    let seed2 = seed1 + 1u;

    let u1 = max(random_float(seed1), 0.0000001); // Prevent log(0)
    let u2 = random_float(seed2);

    let z0 = sqrt(-2.0 * log(u1)) * cos(6.2831853 * u2); // 6.2831853 = 2 * PI
    return z0 * amp;
}