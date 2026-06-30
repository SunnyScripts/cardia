const softening_epsilon = 1.0; // Biot-Savart softening parameter. It prevents the velocity from reaching infinity when pos is exactly on the knot line
const softening_cling = 0.1; // prevent the "static cling" force from causing a NaN crash

struct ShaderTelemetry {
    debug_x: f32, debug_y: f32, debug_z: f32, local_rho: f32,
    local_ux: f32, local_uy: f32, local_uz: f32,
    korteweg_fx: f32, phase_lock_fx: f32, lorentz_factor: f32,
    tripwire_triggered: atomic<u32>, _pad: u32,
}

struct PhysicsUniforms {
    width: u32, height: u32, depth: u32, tick: u32,
    tau: f32, zpe_amplitude: f32, base_speed_lu: f32, u_core_lu: f32,
    gamma_lu: f32, coherence_length_lu: f32, shan_chen_g: f32, phase_lock_rate: f32,

    // 🌟 FIXED: 7 vec4s = 28 u32 values, strictly aligned to 16-byte boundaries
    inverse_directions: array<vec4<u32>, 7>,
}

struct BlueprintParams {
    ring_count: u32, blend_factor: f32, gamma_lu: f32, r_nucleon: f32,
    u_core_lu: f32, n_state: f32, l_state: f32, m_state: f32,
    core_pos: vec4<f32>,
    rings: array<vec4<f32>, 256>,
    ring_props: array<vec4<f32>, 256>,
}

@group(0) @binding(0) var<uniform> params: PhysicsUniforms;

// 🌟 THE BAILEY A-A GRID: One single read/write array!
@group(0) @binding(1) var<storage, read_write> f_grid: array<f32>;

@group(0) @binding(2) var<uniform> atom_params: BlueprintParams;
@group(0) @binding(3) var<storage, read_write> macro_rho: array<f32>;
@group(0) @binding(4) var<storage, read_write> telemetry: ShaderTelemetry;

// ---------------------------------------------------------
// D3Q27 LATTICE CONSTANTS
// ---------------------------------------------------------
const ex = array<f32, 27>(
    0.,
    1., -1., 0.,  0., 0.,  0.,
    1., -1., 1., -1., 1., -1., 1., -1., 0.,  0., 0.,  0.,
    1., -1., 1., -1., 1., -1., 1., -1.
);
const ey = array<f32, 27>(
    0.,
    0.,  0., 1., -1., 0.,  0.,
    1., -1.,-1.,  1., 0.,  0., 0.,  0., 1., -1., 1., -1.,
    1., -1., 1., -1.,-1.,  1.,-1.,  1.
);
const ez = array<f32, 27>(
    0.,
    0.,  0., 0.,  0., 1., -1.,
    0.,  0., 0.,  0., 1., -1.,-1.,  1., 1., -1.,-1.,  1.,
    1., -1.,-1.,  1., 1., -1.,-1.,  1.
);

const w = array<f32, 27>(
    8.0/27.0,
    2.0/27.0, 2.0/27.0, 2.0/27.0, 2.0/27.0, 2.0/27.0, 2.0/27.0,
    1.0/54.0, 1.0/54.0, 1.0/54.0, 1.0/54.0, 1.0/54.0, 1.0/54.0, 1.0/54.0, 1.0/54.0, 1.0/54.0, 1.0/54.0, 1.0/54.0, 1.0/54.0,
    1.0/216.0, 1.0/216.0, 1.0/216.0, 1.0/216.0, 1.0/216.0, 1.0/216.0, 1.0/216.0, 1.0/216.0
);

const rev_d = array<u32, 27>(
    0,
    2, 1, 4, 3, 6, 5,
    8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17,
    20, 19, 22, 21, 24, 23, 26, 25
);

fn get_idx(x: u32, y: u32, z: u32) -> u32 {
    return z * (params.width * params.height) + y * params.width + x;
}

fn safe_isnan(val: f32) -> bool {
    let bits: u32 = bitcast<u32>(val);
    let exponent: u32 = (bits >> 23u) & 0xFFu;
    let fraction: u32 = bits & 0x7FFFFFu;

    // An IEEE-754 float is NaN if the exponent is all 1s (0xFF)
    // and the trailing fraction/mantissa is non-zero.
    return (exponent == 0xFFu) && (fraction != 0u);
}

// Helper function to read from the packed vec4 array
fn get_inverse_direction(d: u32) -> u32 {
    let vec_idx = d / 4u;   // Which vec4 it lives in (0 to 6)
    let comp_idx = d % 4u;  // Which component (.x, .y, .z, or .w)

    let v = params.inverse_directions[vec_idx];

    if (comp_idx == 0u) { return v.x; }
    if (comp_idx == 1u) { return v.y; }
    if (comp_idx == 2u) { return v.z; }
    return v.w; // comp_idx == 3u
}

// ---------------------------------------------------------
// STOCHASTIC VACUUM GENERATOR
// ---------------------------------------------------------


fn random_float(seed: u32) -> f32 {
    let mantissa = seed >> 8u;
    return f32(mantissa) / 16777215.0;
}

fn get_local_macro(nx: u32, ny: u32, nz: u32) -> vec4<f32> {
    let base = (nz * params.width * params.height + ny * params.width + nx) * 27u;
    var r = 0.0;
    var u = vec3<f32>(0.0);

    for (var d = 0u; d < 27u; d++) {
        let f_val = f_grid[base + d];
        r += f_val;
        u += vec3<f32>(ex[d], ey[d], ez[d]) * f_val;
    }
    return vec4<f32>(u.x, u.y, u.z, r);
}

fn get_vorticity_at(x: u32, y: u32, z: u32) -> vec3<f32> {
    // 1. Boundary-safe neighbor indices
    let px = (x + 1u) % params.width; let mx = (x + params.width - 1u) % params.width;
    let py = (y + 1u) % params.height; let my = (y + params.height - 1u) % params.height;
    let pz = (z + 1u) % params.depth; let mz = (z + params.depth - 1u) % params.depth;

    // 2. Fetch macroscopic velocity at neighbors (u = J/rho)
    let u_px = get_local_u(px, y, z, get_local_rho(px, y, z));
    let u_mx = get_local_u(mx, y, z, get_local_rho(mx, y, z));
    let u_py = get_local_u(x, py, z, get_local_rho(x, py, z));
    let u_my = get_local_u(x, my, z, get_local_rho(x, my, z));
    let u_pz = get_local_u(x, y, pz, get_local_rho(x, y, pz));
    let u_mz = get_local_u(x, y, mz, get_local_rho(x, y, mz));

    // 3. Finite Difference Curl Calculation
    // w = curl(u) = (duz/dy - duy/dz, dux/dz - duz/dx, duy/dx - dux/dy)
    let w_x = (u_py.z - u_my.z) * 0.5 - (u_pz.y - u_mz.y) * 0.5;
    let w_y = (u_pz.x - u_mz.x) * 0.5 - (u_px.z - u_mx.z) * 0.5;
    let w_z = (u_px.y - u_mx.y) * 0.5 - (u_py.x - u_my.x) * 0.5;

    return vec3<f32>(w_x, w_y, w_z);
}

// 1. The Vector Potential (Your old get_xi)
// This generates the underlying stochastic grid field
fn get_vector_potential(x: u32, y: u32, z: u32, tick: u32) -> vec3<f32> {
    // ONE hash instead of three
    let base_seed = hash31(x, y, z, tick, 13u);

    // Fast bit-mixing for the X, Y, Z channels
    let h_x = base_seed * 747796405u + 2891336453u;
    let h_y = h_x * 747796405u + 2891336453u;
    let h_z = h_y * 747796405u + 2891336453u;

    return vec3<f32>(
        (random_float(h_x) - 0.5) * 2.0,
        (random_float(h_y) - 0.5) * 2.0,
        (random_float(h_z) - 0.5) * 2.0
    );
}



fn get_local_rho(nx: u32, ny: u32, nz: u32) -> f32 {
    let base = (nz * params.width * params.height + ny * params.width + nx) * 27u;
    var r = 0.0;
    for (var d = 0u; d < 27u; d++) { r += f_grid[base + d]; }
    return r;
}

fn get_local_u(nx: u32, ny: u32, nz: u32, local_rho: f32) -> vec3<f32> {
    let base = (nz * params.width * params.height + ny * params.width + nx) * 27u;
    var u = vec3<f32>(0.0);
    for (var d = 0u; d < 27u; d++) {
        let f_val = f_grid[base + d];
        u += vec3<f32>(ex[d], ey[d], ez[d]) * f_val;
    }
    return u / max(local_rho, 0.0001);
}

fn re_equilibrate(rho: f32, u: vec3<f32>, f_arr: ptr<function, array<f32, 27>>) {
    let usq = dot(u, u);
    for (var d = 0u; d < 27u; d++) {
        let cu = dot(vec3<f32>(ex[d], ey[d], ez[d]), u);
        // This calculates the equilibrium state for the new velocity
        (*f_arr)[d] = w[d] * rho * (1.0 + 3.0*cu + 4.5*cu*cu - 1.5*usq);
    }
}

fn hash31(x: u32, y: u32, z: u32, tick: u32, offset: u32) -> u32 {
    var h = x * 374761393u + y * 668265263u + z * 3266489917u + tick * 2654435761u + offset;
    h = (h ^ (h >> 13u)) * 3266489917u;
    h = (h ^ (h >> 16u)) * 668265263u;
    return h ^ (h >> 13u);
}

// The Divergence-Free ZPE Micro-Current
// Calculates the cross-derivatives to yield pure rotational flow
fn get_xi(x: u32, y: u32, z: u32, tick: u32) -> vec3<f32> {
    // 1. Neighbors indices
    let px = (x + 1u) % params.width; let mx = (x + params.width - 1u) % params.width;
    let py = (y + 1u) % params.height; let my = (y + params.height - 1u) % params.height;
    let pz = (z + 1u) % params.depth; let mz = (z + params.depth - 1u) % params.depth;

    // 2. Sample 7-point stencil (Center + 6 neighbors) for Laplacian smoothing
    let A_c  = get_vector_potential(x, y, z, tick);
    let A_px = get_vector_potential(px, y, z, tick); let A_mx = get_vector_potential(mx, y, z, tick);
    let A_py = get_vector_potential(x, py, z, tick); let A_my = get_vector_potential(x, my, z, tick);
    let A_pz = get_vector_potential(x, y, pz, tick); let A_mz = get_vector_potential(x, y, mz, tick);

    // 3. APPLY LOW-PASS FILTER (The "Acoustic Opacity Limit")
    // This stencil averages the neighbors to kill frequencies near the Nyquist limit.
    // The alpha factor (0.15) defines the "sharpness" of the energy cutoff.
    let alpha = 0.15;
    let smoothed_px = mix(A_px, (A_px + A_c + A_py + A_my + A_pz + A_mz) / 6.0, alpha);
    let smoothed_mx = mix(A_mx, (A_mx + A_c + A_py + A_my + A_pz + A_mz) / 6.0, alpha);
    let smoothed_py = mix(A_py, (A_py + A_c + A_px + A_mx + A_pz + A_mz) / 6.0, alpha);
    let smoothed_my = mix(A_my, (A_my + A_c + A_px + A_mx + A_pz + A_mz) / 6.0, alpha);
    let smoothed_pz = mix(A_pz, (A_pz + A_c + A_px + A_mx + A_py + A_my) / 6.0, alpha);
    let smoothed_mz = mix(A_mz, (A_mz + A_c + A_px + A_mx + A_py + A_my) / 6.0, alpha);

    // 4. Compute discrete curl on the filtered (ballistic) potential
    let curl_x = (smoothed_py.z - smoothed_my.z) * 0.5 - (smoothed_pz.y - smoothed_mz.y) * 0.5;
    let curl_y = (smoothed_pz.x - smoothed_mz.x) * 0.5 - (smoothed_px.z - smoothed_mx.z) * 0.5;
    let curl_z = (smoothed_px.y - smoothed_mx.y) * 0.5 - (smoothed_py.x - smoothed_mx.x) * 0.5;

    return vec3<f32>(curl_x, curl_y, curl_z);
}

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>)
{
    let x = global_id.x; let y = global_id.y; let z = global_id.z;
    if (x >= params.width || y >= params.height || z >= params.depth) { return; }

    let i = get_idx(x, y, z);
    let base_idx = i * 27u;
    let is_even_tick = (params.tick % 2u) == 0u;

    // ====================================================================
    // 2. FETCH STATE (Bailey A-A Alternating Read)
    // ====================================================================
    var rho: f32 = 0.0;
    var ux: f32 = 0.0; var uy: f32 = 0.0; var uz: f32 = 0.0;
    var f_local: array<f32, 27>;

    for (var d = 0u; d < 27u; d++) {
        var read_idx = base_idx + d;

        if (!is_even_tick) {
            // ODD TICK: Pull from neighbors along the standard lattice vectors
            let mx = u32((i32(x) - i32(ex[d]) + i32(params.width)) % i32(params.width));
            let my = u32((i32(y) - i32(ey[d]) + i32(params.height)) % i32(params.height));
            let mz = u32((i32(z) - i32(ez[d]) + i32(params.depth)) % i32(params.depth));
            read_idx = get_idx(mx, my, mz) * 27u + d;
        }

        let f_val = f_grid[read_idx];
        f_local[d] = f_val;
        rho += f_val;
        ux += f_val * ex[d]; uy += f_val * ey[d]; uz += f_val * ez[d];
    }

    // 🌟 RELATIVISTIC CAVITATION FIX
    // When the core spins at c, density evacuates. We enforce a firm
    // ground-state probability floor to prevent infinite velocity division.
    let safe_rho = max(abs(rho), 0.01);

    ux /= safe_rho; uy /= safe_rho; uz /= safe_rho;

    macro_rho[i] = rho;

    // Calculate distance from the exact center of the grid
    let dist_x = min(x, params.width - 1u - x);
    let dist_y = min(y, params.height - 1u - y);
    let dist_z = min(z, params.depth - 1u - z);
    let min_dist = f32(min(min(dist_x, dist_y), dist_z));

    // ====================================================================
    // 3. SPHERICAL ZPE HORIZON (Geometric Wave Generator)
    // ====================================================================
    let cx_grid = f32(params.width) * 0.5;
    let cy_grid = f32(params.height) * 0.5;
    let cz_grid = f32(params.depth) * 0.5;
    let dx_grid = f32(x) - cx_grid; let dy_grid = f32(y) - cy_grid; let dz_grid = f32(z) - cz_grid;
    let r_dist = sqrt(dx_grid*dx_grid + dy_grid*dy_grid + dz_grid*dz_grid);

    let max_radius = cx_grid - 2.0;
    let sponge_thickness = 15.0;

    // ====================================================================
    // 3.5 VOLUMETRIC ZPE (Symmetric Langevin Thermostat)
    // ====================================================================
    // 1. Generate the pure, divergence-free rotational target field
    let xi = get_xi(x, y, z, params.tick);

    // 2. Scale the field so its maximum physical velocity is your exact ZPE amplitude
    let target_zpe_u = xi * params.zpe_amplitude;

    // 3. The Symmetric Coupling Rate (e.g., 5% thermalization per tick)
    // This acts as BOTH the generation rate and the exit drag rate.
    let zpe_coupling_rate = 0.05;

    // 4. The Force Balance
    // Fluid relaxes toward the ZPE target. Energy in perfectly equals Energy out.
    // The fluid's background energy can never mathematically exceed the ZPE amplitude.
    var ext_fx = (target_zpe_u.x - ux) * zpe_coupling_rate * safe_rho;
    var ext_fy = (target_zpe_u.y - uy) * zpe_coupling_rate * safe_rho;
    var ext_fz = (target_zpe_u.z - uz) * zpe_coupling_rate * safe_rho;

    // ====================================================================
    // 4. TOPOLOGICAL SOLITON & KORTEWEG-AUGMENTED COHESION
    // ====================================================================
    let pos = vec3<f32>(f32(x), f32(y), f32(z));
    var phase_force = vec3<f32>(0.0, 0.0, 0.0);

    let px = (x + 1u) % params.width;  let mx = (x + params.width - 1u) % params.width;
    let py = (y + 1u) % params.height; let my = (y + params.height - 1u) % params.height;
    let pz = (z + 1u) % params.depth;  let mz = (z + params.depth - 1u) % params.depth;

    // --------------------------------------------------------------------
    // 🌟 STAGE 1: THE MOTOR STARTER (Phase Lock Injection)
    // --------------------------------------------------------------------
    if (atom_params.blend_factor > 0.000001) // FIX 4: Immediate engagement
    {
        var knot_u = vec3<f32>(0.0);
        var min_r_sq: f32 = 1000000.0;

        for (var r = 0u; r < atom_params.ring_count; r++)
        {
            let center = atom_params.rings[r].xyz;
            let radius = atom_params.rings[r].w;
            let p = atom_params.ring_props[r].x;
            let q = atom_params.ring_props[r].y;

            var outward_normal = center - atom_params.core_pos.xyz;
            if (length(outward_normal) < 0.001) {
                outward_normal = vec3<f32>(0.0, 0.0, 1.0);
            } else {
                outward_normal = normalize(outward_normal);
            }

            let m_tilt = atom_params.m_state * (3.14159265 / 4.0);
            outward_normal = normalize(vec3<f32>(
                outward_normal.x + sin(m_tilt),
                outward_normal.y,
                outward_normal.z + cos(m_tilt)
            ));

            var basis_u = cross(vec3<f32>(0.0, 1.0, 0.0), outward_normal);
            if (length(basis_u) < 0.01) { basis_u = cross(vec3<f32>(1.0, 0.0, 0.0), outward_normal); }
            basis_u = normalize(basis_u);
            let basis_v = cross(outward_normal, basis_u);

//            let num_segments = 1024u;
//            let ring_dt = 6.2831853 / f32(num_segments);
            let f_macro = atom_params.ring_props[r].z;
            let f_minor = atom_params.ring_props[r].w;

            // 🌟 CALIBRATION FIX: Discretized Biot-Savart Integration
            // The number of segments scales directly with the macroscopic radius
            // ensuring we don't stack infinite sub-grid currents into a single node.
            let R_total = radius;
            let R_macro = R_total * f_macro;
            let R_minor = R_total * f_minor;

            // Calculate circumference and enforce 1 segment per Lattice Unit
            let circumference = 6.2831853 * R_macro;
            let num_segments = max(u32(circumference), 12u); // Minimum 12 segments
            let ring_dt = 6.2831853 / f32(num_segments);

            let core_radius = R_minor;

            for (var seg = 0u; seg < num_segments; seg++) {
                let t = f32(seg) * ring_dt;
                let r_t = R_macro + R_minor * cos(q * t);

                let k_x = r_t * cos(p * t);
                let k_y = r_t * sin(p * t);
                let k_z = R_minor * sin(q * t);

                let dr_dt = -q * R_minor * sin(q * t);
                let d_x = dr_dt * cos(p * t) - p * r_t * sin(p * t);
                let d_y = dr_dt * sin(p * t) + p * r_t * cos(p * t);
                let d_z = q * R_minor * cos(q * t);

                let k_vec = basis_u * k_x + basis_v * k_y + outward_normal * k_z;
                let d_vec = basis_u * d_x + basis_v * d_y + outward_normal * d_z;

                let r_knot = center + k_vec;
                let dl = d_vec * ring_dt;
                let diff = pos - r_knot;
                let r_mag_sq = dot(diff, diff);

                if (r_mag_sq < min_r_sq) { min_r_sq = r_mag_sq; }

                let denom = pow(r_mag_sq + softening_epsilon, 1.5);
                let gaussian_softening = 1.0 - exp(-(r_mag_sq) / (core_radius * core_radius));

                knot_u += (cross(dl, diff) / denom) * gaussian_softening * atom_params.gamma_lu;
            }
        }

        let speed = length(knot_u);
        if (speed > 0.0001)
        {
            let target_u = normalize(knot_u) * atom_params.u_core_lu * tanh(speed / atom_params.u_core_lu);
            let active_blend = atom_params.blend_factor * params.phase_lock_rate;

            phase_force = (target_u - vec3<f32>(ux, uy, uz)) * active_blend * safe_rho;
            ext_fx += phase_force.x;
            ext_fy += phase_force.y;
            ext_fz += phase_force.z;
        }
    }

    // --------------------------------------------------------------------
    // 🌟 STAGE 2: PURE KORTEWEG COHESION (Routed Safely)
    // --------------------------------------------------------------------
    let rho_c = macro_rho[get_idx(x, y, z)];
    let safe_rho_val = max(rho_c, 0.01);

    let psi_c  = 1.0 - exp(-safe_rho_val);
    let psi_px = 1.0 - exp(-max(macro_rho[get_idx(px, y, z)], 0.01));
    let psi_mx = 1.0 - exp(-max(macro_rho[get_idx(mx, y, z)], 0.01));
    let psi_py = 1.0 - exp(-max(macro_rho[get_idx(x, py, z)], 0.01));
    let psi_my = 1.0 - exp(-max(macro_rho[get_idx(x, my, z)], 0.01));
    let psi_pz = 1.0 - exp(-max(macro_rho[get_idx(x, y, pz)], 0.01));
    let psi_mz = 1.0 - exp(-max(macro_rho[get_idx(x, y, mz)], 0.01));

    let w_axial = 2.0 / 27.0;
    var korteweg_force = vec3<f32>(
        w_axial * (psi_px - psi_mx),
        w_axial * (psi_py - psi_my),
        w_axial * (psi_pz - psi_mz)
    );

    // FIX 1: Restore the physical attraction vector
    korteweg_force *= -params.shan_chen_g * psi_c;

    let vel = vec3<f32>(ux, uy, uz);
    let vel_mag = length(vel);
    if (vel_mag > 1e-4) {
        let vel_n = vel / vel_mag;
        korteweg_force -= dot(korteweg_force, vel_n) * vel_n;
    }

    // FIX 2: Route the force safely into the Relativistic clamping pool
    ext_fx += korteweg_force.x;
    ext_fy += korteweg_force.y;
    ext_fz += korteweg_force.z;

    // ====================================================================
    // 5. RLBM RELATIVISTIC HYDRODYNAMICS (Maxwell-Jüttner Manifold)
    // ====================================================================
    const c_light = 0.577350269;
    let c_light_sq = 1.0 / 3.0;

    let raw_speed = sqrt(ux*ux + uy*uy + uz*uz);
    let alpha_speed = raw_speed / c_light;

    // The Lorentz Factor (Gamma)
    let lorentz_factor = cosh(alpha_speed);
    let gamma_sq = lorentz_factor * lorentz_factor;
    let gamma_cubed = gamma_sq * lorentz_factor;

    var speed_scale = 1.0;
    if (raw_speed > 1e-9) { speed_scale = (c_light * tanh(alpha_speed)) / raw_speed; }

    let curved_ux = ux * speed_scale;
    let curved_uy = uy * speed_scale;
    let curved_uz = uz * speed_scale;

    var px_mom = (curved_ux * lorentz_factor) + (ext_fx / safe_rho);
    var py_mom = (curved_uy * lorentz_factor) + (ext_fy / safe_rho);
    var pz_mom = (curved_uz * lorentz_factor) + (ext_fz / safe_rho);

    // 🛡️ ALU OVERFLOW BYPASS: Cap momentum to keep squares safely under 3.4e38
    let p_raw = sqrt(px_mom*px_mom + py_mom*py_mom + pz_mom*pz_mom);
    let p_clamped = min(p_raw, 1e18);

    // Algebraic substitution to bypass cosh/asinh exponential overflow
    let inv_new_lorentz = 1.0 / sqrt(1.0 + (p_clamped * p_clamped) / c_light_sq);

    var t_ux_e = px_mom * inv_new_lorentz;
    var t_uy_e = py_mom * inv_new_lorentz;
    var t_uz_e = pz_mom * inv_new_lorentz;

    let thermo_new_speed = sqrt(t_ux_e*t_ux_e + t_uy_e*t_uy_e + t_uz_e*t_uz_e);
    let t_usq_e = thermo_new_speed * thermo_new_speed;
    let c_usq = curved_ux*curved_ux + curved_uy*curved_uy + curved_uz*curved_uz;

    // 🌟 3. RLBM MAXWELL-JÜTTNER EQUILIBRIUM UPDATE
    for (var d = 0u; d < 27u; d++) {
        let c_eu = ex[d]*curved_ux + ey[d]*curved_uy + ez[d]*curved_uz;

        // Maxwell-Jüttner approximation (O(u^2))
        let c_eq = w[d] * safe_rho * (lorentz_factor + 3.0*gamma_sq*c_eu + 4.5*gamma_cubed*(c_eu*c_eu) - 1.5*lorentz_factor*c_usq);

        let t_eu = ex[d]*t_ux_e + ey[d]*t_uy_e + ez[d]*t_uz_e;
        let t_eq = w[d] * safe_rho * (lorentz_factor + 3.0*gamma_sq*t_eu + 4.5*gamma_cubed*(t_eu*t_eu) - 1.5*lorentz_factor*t_usq_e);

        f_local[d] += (t_eq - c_eq);
    }

    ux = t_ux_e; uy = t_uy_e; uz = t_uz_e;

    let speed_check = sqrt(t_ux_e*t_ux_e + t_uy_e*t_uy_e + t_uz_e*t_uz_e);
    if (speed_check > 0.578 || safe_isnan(speed_check)) {
        var expected: u32 = 0u;
        if (atomicCompareExchangeWeak(&telemetry.tripwire_triggered, expected, 1u).exchanged) {
            telemetry.debug_x = f32(x); telemetry.debug_y = f32(y); telemetry.debug_z = f32(z);
            telemetry.local_rho = safe_rho; telemetry.local_ux = t_ux_e; telemetry.local_uy = t_uy_e; telemetry.local_uz = t_uz_e;
            telemetry.korteweg_fx = korteweg_force.x; telemetry.phase_lock_fx = phase_force.x; telemetry.lorentz_factor = lorentz_factor;
        }
    }

    // ====================================================================
    // 6. RLBM KBC ENTROPIC PROJECTION
    // ====================================================================
    var Pi_xx: f32 = 0.0; var Pi_yy: f32 = 0.0; var Pi_zz: f32 = 0.0;
    var Pi_xy: f32 = 0.0; var Pi_xz: f32 = 0.0; var Pi_yz: f32 = 0.0;
    var delta_f = array<f32, 27>();
    var s_array = array<f32, 27>();
    var h_array = array<f32, 27>();

    let usq_kbc = ux*ux + uy*uy + uz*uz;

    for (var d = 0u; d < 27u; d++) {
        let cx = ex[d]; let cy = ey[d]; let cz = ez[d];
        let cu = cx*ux + cy*uy + cz*uz;

        // 🌟 RLBM EQUILIBRIUM FOR KBC
        let f_eq = w[d] * safe_rho * (lorentz_factor + 3.0*gamma_sq*cu + 4.5*gamma_cubed*(cu*cu) - 1.5*lorentz_factor*usq_kbc);
        let df = f_local[d] - f_eq;

        delta_f[d] = df;
        Pi_xx += df * cx * cx; Pi_yy += df * cy * cy; Pi_zz += df * cz * cz;
        Pi_xy += df * cx * cy; Pi_xz += df * cx * cz; Pi_yz += df * cy * cz;
    }

    for (var d = 0u; d < 27u; d++) {
        let cx = ex[d]; let cy = ey[d]; let cz = ez[d];
        let Q_xx = cx*cx - 0.33333333; let Q_yy = cy*cy - 0.33333333; let Q_zz = cz*cz - 0.33333333;

        let s_i = 4.5 * w[d] * (Pi_xx*Q_xx + Pi_yy*Q_yy + Pi_zz*Q_zz +
                                2.0*Pi_xy*(cx*cy) + 2.0*Pi_xz*(cx*cz) + 2.0*Pi_yz*(cy*cz));
        s_array[d] = s_i;
        h_array[d] = delta_f[d] - s_i;
    }

    let beta = 1.0 / params.tau;
    var numerator_sum: f32 = 0.0;
    var denominator_sum: f32 = 0.0;

    for (var d = 0u; d < 27u; d++) {
        if (w[d] > 0.0) {
            numerator_sum += (s_array[d] * h_array[d]) / w[d];
            denominator_sum += (h_array[d] * h_array[d]) / w[d];
        }
    }

    var gamma_kbc = 2.0;
    // 🌟 FIX 4: 1e-25 threshold so KBC can "see" the quantum noise
    if (denominator_sum > 1e-25) {
        gamma_kbc = beta - (2.0 - beta) * (numerator_sum / denominator_sum);
    }
    gamma_kbc = clamp(gamma_kbc, 1.0, 2.4);

    for (var d = 0u; d < 27u; d++)
    {
        let cu_kbc = ex[d]*ux + ey[d]*uy + ez[d]*uz;

        // 🌟 RLBM EQUILIBRIUM WRITEBACK
        let f_eq = w[d] * safe_rho * (lorentz_factor + 3.0*gamma_sq*cu_kbc + 4.5*gamma_cubed*(cu_kbc * cu_kbc) - 1.5*lorentz_factor*usq_kbc);
        f_local[d] = f_eq + (1.0 - beta) * s_array[d] + (1.0 - gamma_kbc) * h_array[d];
    }

    // ====================================================================
    // 6.5 NON-REFLECTING BOUNDARY CONDITION (Transparent Vacuum)
    // ====================================================================
    if (r_dist > max_radius - sponge_thickness)
    {
        let sponge_factor = clamp((r_dist - (max_radius - sponge_thickness)) / sponge_thickness, 0.0, 1.0);
        for (var d = 0u; d < 27u; d++) {
            let dot_dir = dot(vec3<f32>(ex[d], ey[d], ez[d]), normalize(vec3<f32>(dx_grid, dy_grid, dz_grid)));
            if (dot_dir > 0.0) {
                // FIX 3: Dampen pressure anomalies by forcing equilibrium to 1.0
                f_local[d] = mix(f_local[d], w[d] * 1.0, sponge_factor * 0.05);
            }
        }
    }

    // ====================================================================
    // 7. IN-PLACE STREAMING AND WRITEBACK (Bailey A-A Alternating Write)
    // ====================================================================
    for (var d = 0u; d < 27u; d++) {
        var write_idx = base_idx + d;

        if (is_even_tick) {
            // EVEN TICK: Push out to target neighbors using the inverted lattice directions
            let inv_d = get_inverse_direction(d); // Fetched from uniform parameters
            let px = u32((i32(x) + i32(ex[inv_d]) + i32(params.width)) % i32(params.width));
            let py = u32((i32(y) + i32(ey[inv_d]) + i32(params.height)) % i32(params.height));
            let pz = u32((i32(z) + i32(ez[inv_d]) + i32(params.depth)) % i32(params.depth));
            write_idx = get_idx(px, py, pz) * 27u + inv_d;
        }

        // Write directly back into the single allocated fluid grid
        f_grid[write_idx] = f_local[d];
    }
}