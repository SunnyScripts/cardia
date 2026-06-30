// ======================================================================
// 1. WHERE GAMMA_LU IS DEFINED (Must perfectly match Rust's BlueprintUniforms)
// ======================================================================

/*
//struct BlueprintParams {
//    width: u32,
//    height: u32,
//    depth: u32,
//    ring_count: u32,
//    blend_factor: f32,
//    gamma_lu: f32,    // <--- DEFINED HERE: Scaled by dx/dt in Rust
//    p_wind: f32,
//    q_wind: f32,
//    core_pos: vec4<f32>, // <--- The center of the whole atom, used for orientation
//    rings: array<vec4<f32>, 256>,
//};
*/

struct BlueprintParams {
    width: u32, height: u32, depth: u32, ring_count: u32,
    blend_factor: f32, gamma_lu: f32, _pad1: f32, _pad2: f32,
    core_pos: vec4<f32>,
    rings: array<vec4<f32>, 256>,
    ring_props: array<vec4<f32>, 256>, // NEW: Per-nucleon chirality
}

@group(0) @binding(0) var<uniform> params: BlueprintParams;
@group(0) @binding(1) var<storage, read_write> f_3d: array<f32>;

// D3Q27 LATTICE CONSTANTS
const ex = array<f32, 27>(0., 1.,-1., 0., 0., 0., 0., 1.,-1., 1.,-1., 1.,-1., 1.,-1., 0., 0., 0., 0., 1.,-1., 1.,-1., 1.,-1., 1.,-1.);
const ey = array<f32, 27>(0., 0., 0., 1.,-1., 0., 0., 1.,-1.,-1., 1., 0., 0., 0., 0., 1.,-1., 1.,-1., 1.,-1., 1.,-1.,-1., 1.,-1., 1.);
const ez = array<f32, 27>(0., 0., 0., 0., 0., 1.,-1., 0., 0., 0., 0., 1.,-1.,-1., 1., 1.,-1.,-1., 1., 1.,-1.,-1., 1., 1.,-1.,-1., 1.);
const w = array<f32, 27>(
    8.0/27.0,
    2.0/27.0, 2.0/27.0, 2.0/27.0, 2.0/27.0, 2.0/27.0, 2.0/27.0,
    1.0/54.0, 1.0/54.0, 1.0/54.0, 1.0/54.0, 1.0/54.0, 1.0/54.0, 1.0/54.0, 1.0/54.0, 1.0/54.0, 1.0/54.0, 1.0/54.0, 1.0/54.0,
    1.0/216.0, 1.0/216.0, 1.0/216.0, 1.0/216.0, 1.0/216.0, 1.0/216.0, 1.0/216.0, 1.0/216.0
);

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>)
{
    if (id.x >= params.width || id.y >= params.height || id.z >= params.depth) { return; }

    let pos = vec3<f32>(f32(id.x), f32(id.y), f32(id.z));
    var u = vec3<f32>(0.0);
    var min_r_sq: f32 = 1000000.0;

    // Loop through however many alpha clusters/nucleons Rust tells us to draw
    for (var r = 0u; r < params.ring_count; r++)
    {
        let center = params.rings[r].xyz;
        let radius = params.rings[r].w;

        // Fetch the unique winding for THIS specific nucleon
        let p = params.ring_props[r].x;
        let q = params.ring_props[r].y;

        // 3D Orientation: Build a static orthonormal basis
        var outward_normal = center - params.core_pos.xyz;
        if (length(outward_normal) < 0.001) {
            // If this is the core proton, align it perfectly to the Z-axis
            outward_normal = vec3<f32>(0.0, 0.0, 1.0);
        } else {
            outward_normal = normalize(outward_normal);
        }

        var basis_u = cross(vec3<f32>(0.0, 1.0, 0.0), outward_normal);
        if (length(basis_u) < 0.01) {
            basis_u = cross(vec3<f32>(1.0, 0.0, 0.0), outward_normal);
        }
        basis_u = normalize(basis_u);
        let basis_v = cross(outward_normal, basis_u);

        let num_segments = 512u; // Sufficient for localized sub-knots
        let ring_dt = 6.2831853 / f32(num_segments);

        let R_total = radius * 0.46; // 18.0 * 0.46 ≈ 8.28 nodes

        let R_macro = R_total * 0.4;
        let R_minor = R_total * 0.6;

        // Widen the softening gradient so velocity blends smoothly
        let core_radius = R_total * 0.8;

        for (var i = 0u; i < num_segments; i++) {
            let t = f32(i) * ring_dt;

            // 1. Raw Parametric Position (Standard T_pq geometry)
            let r_t = R_macro + R_minor * cos(q * t);
            let k_x = r_t * cos(p * t);
            let k_y = r_t * sin(p * t);
            let k_z = R_minor * sin(q * t);

            // 2. Raw Analytical Directional Derivatives
            let dr_dt = -q * R_minor * sin(q * t);
            let d_x = dr_dt * cos(p * t) - p * r_t * sin(p * t);
            let d_y = dr_dt * sin(p * t) + p * r_t * cos(p * t);
            let d_z = q * R_minor * cos(q * t);

            // 3. Map BOTH position and direction onto the static 3D Nucleus Frame
            let k_vec = basis_u * k_x + basis_v * k_y + outward_normal * k_z;
            let d_vec = basis_u * d_x + basis_v * d_y + outward_normal * d_z;

            let r_knot = center + k_vec;
            let dl = d_vec * ring_dt;

            // 4. Distance and Gaussian Softening
            let diff = pos - r_knot;
            let r_mag_sq = dot(diff, diff);
            min_r_sq = min(min_r_sq, r_mag_sq);

            let denom = pow(r_mag_sq + 0.1, 1.5);
            let gaussian_softening = 1.0 - exp(-(r_mag_sq) / (core_radius * core_radius));

            // Apply Biot-Savart Law for fluid momentum
            u += (cross(dl, diff) / denom) * gaussian_softening * params.gamma_lu;
        }
    } // End of ring_count loop

    // ======================================================================
    // 3. THE "ALPHAFOLD" PRE-CARVED DENSITY HOLE
    // ======================================================================
    let dist_to_knot = sqrt(min_r_sq);
    let base_radius = params.rings[0].w;
    let R_total = base_radius * 0.46;
    let effective_core = R_total * 1.1;

    // PURE target density. No blend factor here.
    let t_rho = mix(0.85, 1.0, smoothstep(0.0, effective_core, dist_to_knot));

    // ======================================================================
    // 4. VELOCITY CLAMPING & SMOOTHING
    // ======================================================================
    let max_u = 0.25;
    let speed = length(u);
    let velocity_mask = 1.0 - smoothstep(0.0, effective_core * 1.5, dist_to_knot);

    var raw_u = vec3<f32>(0.0);
    if (speed > 0.0001) {
        // PURE target velocity. No blend factor here.
        raw_u = normalize(u) * min(speed, max_u) * velocity_mask;
    }

    let base_idx = (id.z * params.width * params.height + id.y * params.width + id.x) * 27u;

    // ======================================================================
    // 5. NON-DESTRUCTIVE MOMENTUM INJECTION (THE CLEAN BLEND)
    // ======================================================================

    // 1. Read the exact current state of the living fluid
    var c_rho: f32 = 0.0;
    var c_ux: f32 = 0.0; var c_uy: f32 = 0.0; var c_uz: f32 = 0.0;

    for (var d = 0u; d < 27u; d++) {
        let f_val = f_3d[base_idx + d];
        c_rho += f_val;
        c_ux += f_val * ex[d];
        c_uy += f_val * ey[d];
        c_uz += f_val * ez[d];
    }

    c_rho = max(0.001, c_rho);
    c_ux /= c_rho; c_uy /= c_rho; c_uz /= c_rho;
    let c_usq = c_ux*c_ux + c_uy*c_uy + c_uz*c_uz;

    // 2. Define the Target State (Current autonomous flow + Injected spin)
    let t_ux = c_ux + raw_u.x;
    let t_uy = c_uy + raw_u.y;
    let t_uz = c_uz + raw_u.z;
    let t_usq = t_ux*t_ux + t_uy*t_uy + t_uz*t_uz;

    // 3. Shift the populations, blending ONLY the delta ONCE
    for (var d = 0u; d < 27u; d++) {
        let current_f = f_3d[base_idx + d];

        let c_eu = ex[d]*c_ux + ey[d]*c_uy + ez[d]*c_uz;
        let c_eq = w[d] * c_rho * (1.0 + 3.0*c_eu + 4.5*c_eu*c_eu - 1.5*c_usq);

        let t_eu = ex[d]*t_ux + ey[d]*t_uy + ez[d]*t_uz;
        let t_eq = w[d] * t_rho * (1.0 + 3.0*t_eu + 4.5*t_eu*t_eu - 1.5*t_usq);

        let delta_f = t_eq - c_eq;

        // Multiply by blend factor exactly ONCE
        f_3d[base_idx + d] = current_f + (delta_f * params.blend_factor);
    }
}