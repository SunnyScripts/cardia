enable f16;
const softening_epsilon = 0.1;
const softening_cling = 0.1;
const sponge_thickness = 15.0;


struct ShaderTelemetry {
    debug_x: f32, debug_y: f32, debug_z: f32, local_rho: f32,
    local_ux: f32, local_uy: f32, local_uz: f32,
    korteweg_fx: f32, phase_lock_fx: f32, lorentz_factor: f32,
    tripwire_triggered: atomic<u32>,

    max_pi_magnitude: f32,      // Tracks peak localized vacuum shear
    peak_polarized_tau: f32,    // Tracks maximum local tau deviation
    negative_population_count: atomic<u32> // Monitors how close voxels are to mass inversion
}

struct PhysicsUniforms {
    width: u32, height: u32, depth: u32, tick: u32,
    tau: f32, zpe_amplitude: f32, c_polarization: f32, u_core_lu: f32,

    u_axial_coeff: f32, coherence_length_lu: f32, shan_chen_g: f32, phase_lock_rate: f32,

    inverse_directions: array<vec4<u32>, 10>,
    r_minor: f32, r_macro: f32,
    zpe_coupling_rate: f32, trap_blend: f32
}

struct BlueprintParams {
    ring_count: u32, blend_factor: f32, gamma_lu: f32, r_nucleon: f32,
    u_core_lu: f32, n_state: f32, l_state: f32, m_state: f32,
    core_pos: vec4<f32>,
    rings: array<vec4<f32>, 256>,
    ring_props: array<vec4<f32>, 256>,
}

// ---------------------------------------------------------
// 🌟 D3Q39 6TH-ORDER LATTICE CONSTANTS
// ---------------------------------------------------------
const ex = array<f32, 39>(
    0.,
    1., -1., 0., 0., 0., 0.,
    1., -1., 1., -1., 1., -1., 1., -1.,
    2., -2., 0., 0., 0., 0.,
    2., -2., 2., -2., 2., -2., 2., -2., 0., 0., 0., 0.,
    3., -3., 0., 0., 0., 0.
);
const ey = array<f32, 39>(
    0.,
    0., 0., 1., -1., 0., 0.,
    1., 1., -1., -1., 1., 1., -1., -1.,
    0., 0., 2., -2., 0., 0.,
    2., 2., -2., -2., 0., 0., 0., 0., 2., -2., 2., -2.,
    0., 0., 3., -3., 0., 0.
);
const ez = array<f32, 39>(
    0.,
    0., 0., 0., 0., 1., -1.,
    1., 1., 1., 1., -1., -1., -1., -1.,
    0., 0., 0., 0., 2., -2.,
    0., 0., 0., 0., 2., 2., -2., -2., 2., 2., -2., -2.,
    0., 0., 0., 0., 3., -3.
);
const w = array<f32, 39>(
    1./12.,
    1./12., 1./12., 1./12., 1./12., 1./12., 1./12.,
    1./27., 1./27., 1./27., 1./27., 1./27., 1./27., 1./27., 1./27.,
    2./135., 2./135., 2./135., 2./135., 2./135., 2./135.,
    1./432., 1./432., 1./432., 1./432., 1./432., 1./432., 1./432., 1./432., 1./432., 1./432., 1./432., 1./432.,
    1./1620., 1./1620., 1./1620., 1./1620., 1./1620., 1./1620.
);

fn get_idx(x: u32, y: u32, z: u32) -> u32 {
    return z * (params.width * params.height) + y * params.width + x;
}

fn safe_isnan(val: f32) -> bool {
    let bits: u32 = bitcast<u32>(val);
    let exponent: u32 = (bits >> 23u) & 0xFFu;
    let fraction: u32 = bits & 0x7FFFFFu;
    return (exponent == 0xFFu) && (fraction != 0u);
}

fn get_inverse_direction(d: u32) -> u32 {
    let vec_idx = d / 4u;
    let comp_idx = d % 4u;
    let v = params.inverse_directions[vec_idx];

    if (comp_idx == 0u) { return v.x; }
    if (comp_idx == 1u) { return v.y; }
    if (comp_idx == 2u) { return v.z; }
    return v.w;
}

fn random_float(seed: u32) -> f32 {
    let mantissa = seed >> 8u;
    return f32(mantissa) / 16777215.0;
}

//fn get_local_rho(nx: u32, ny: u32, nz: u32) -> f32 {
//    // 🌟 STRIDE CALIBRATION: Scale by 40u to preserve hardware cache alignment
//    let base = (nz * params.width * params.height + ny * params.width + nx) * 40u;
//    var r = 0.0;
//    for (var d = 0u; d < 39u; d++) { r += f_grid[base + d]; }
//    return r;
//}

//fn get_rho(idx: u32) -> f32 {
//    let b = idx * 40u; // 🌟 Padding scaled
//    var r = 0.0;
//    for (var d = 0u; d < 39u; d++) { r += f_grid[b + d]; }
//    return max(0.0001, r);
//}

//fn get_local_u(nx: u32, ny: u32, nz: u32, local_rho: f32) -> vec3<f32> {
//    let base = (nz * params.width * params.height + ny * params.width + nx) * 40u; // 🌟 Padding scaled
//    var u = vec3<f32>(0.0);
//    for (var d = 0u; d < 39u; d++) {
//        let f_val = f_grid[base + d];
//        u += vec3<f32>(ex[d], ey[d], ez[d]) * f_val;
//    }
//    return u / max(local_rho, 0.0001);
//}

//fn get_rho_safe(nx: i32, ny: i32, nz: i32, tick: u32) -> f32 {
//    if (nx < 0 || nx >= i32(params.width) ||
//        ny < 0 || ny >= i32(params.height) ||
//        nz < 0 || nz >= i32(params.depth)) {
//        return 1.0;
//    }
//
//    let is_even = (tick % 2u) == 0u;
//    let target_idx = u32(nz) * params.width * params.height + u32(ny) * params.width + u32(nx);
//    let target_base = target_idx * 40u;
//
//    var r = 0.0;
//    for (var d = 0u; d < 39u; d++) {
//        let mx = nx - i32(ex[d]);
//        let my = ny - i32(ey[d]);
//        let mz = nz - i32(ez[d]);
//
//        if (mx >= 0 && mx < i32(params.width) &&
//            my >= 0 && my < i32(params.height) &&
//            mz >= 0 && mz < i32(params.depth)) {
//
//            if (is_even) {
//                r += f_grid[target_base + d];
//            } else {
//                let inv_d = get_inverse_direction(d);
//                r += f_grid[target_base + inv_d];
//            }
//        } else {
//            r += w[d] * 1.0;
//        }
//    }
//    return max(0.01, r);
//}

// 🌟 FIXED: Restored Race-Free Stencil Fetcher for the Split Architecture
fn get_rho_safe(nx: i32, ny: i32, nz: i32, tick: u32) -> f32 {
    if (nx < 0 || nx >= i32(params.width) || ny < 0 || ny >= i32(params.height) || nz < 0 || nz >= i32(params.depth)) { return 1.0; }

    let is_even = (tick % 2u) == 0u;
    let target_base = (u32(nz) * params.width * params.height + u32(ny) * params.width + u32(nx)) * 20u;

    var r = 0.0;
    for (var d = 0u; d < 39u; d++) {
        let mx = nx - i32(ex[d]); let my = ny - i32(ey[d]); let mz = nz - i32(ez[d]);

        if (mx >= 0 && mx < i32(params.width) && my >= 0 && my < i32(params.height) && mz >= 0 && mz < i32(params.depth)) {
            var read_dir = d;
            if (!is_even) { read_dir = get_inverse_direction(d); }

            if (read_dir < 20u) { r += f32(f_grid_low[target_base + read_dir]); }
            else { r += f32(f_grid_high[target_base + (read_dir - 20u)]); }
        } else {
            r += w[d] * 1.0;
        }
    }
    return max(0.01, r);
}

// 🌟 MURMUR3 AVALANCHE HASH
// Provides cryptographically uniform distribution, eliminating the Marsaglia Effect
fn hash_u32(seed: u32) -> u32 {
    var h = seed;
    h ^= h >> 16u;
    h *= 0x85ebca6bu;
    h ^= h >> 13u;
    h *= 0xc2b2ae35u;
    h ^= h >> 16u;
    return h;
}

fn get_vector_potential(x: u32, y: u32, z: u32, tick: u32) -> vec3<f32> {
    // Generate a unique spatial baseline seed
    let h_base = x * 374761393u + y * 668265263u + z * 3266489917u + tick * 2654435761u;

    // 🌟 INDEPENDENT AXIAL DECORRELATION
    // Generate X, Y, and Z entirely independently using prime offsets
    // to prevent hyperplane alignment.
    let h_x = hash_u32(h_base + 13u);
    let h_y = hash_u32(h_base + 17u);
    let h_z = hash_u32(h_base + 19u);

    // Shift mantissa down and normalize to [-1.0, 1.0]
    let f_x = f32(h_x >> 8u) / 16777215.0;
    let f_y = f32(h_y >> 8u) / 16777215.0;
    let f_z = f32(h_z >> 8u) / 16777215.0;

    return vec3<f32>(
        (f_x - 0.5) * 2.0,
        (f_y - 0.5) * 2.0,
        (f_z - 0.5) * 2.0
    );
}

fn get_xi(x: u32, y: u32, z: u32, tick: u32) -> vec3<f32> {
    // 🌟 INFINITE NOISE FIELD
    // By casting to i32 and letting the coordinates push negative or
    // past params.width, the noise hash will natively generate a seamless,
    // infinite vector potential field extending infinitely beyond the grid.
    let px = u32(i32(x) + 1); let mx = u32(i32(x) - 1);
    let py = u32(i32(y) + 1); let my = u32(i32(y) - 1);
    let pz = u32(i32(z) + 1); let mz = u32(i32(z) - 1);

    let A_c  = get_vector_potential(x, y, z, tick);
    let A_px = get_vector_potential(px, y, z, tick); let A_mx = get_vector_potential(mx, y, z, tick);
    let A_py = get_vector_potential(x, py, z, tick); let A_my = get_vector_potential(x, my, z, tick);
    let A_pz = get_vector_potential(x, y, pz, tick); let A_mz = get_vector_potential(x, y, mz, tick);

    let alpha = 0.15;
    let smoothed_px = mix(A_px, (A_px + A_c + A_py + A_my + A_pz + A_mz) / 6.0, alpha);
    let smoothed_mx = mix(A_mx, (A_mx + A_c + A_py + A_my + A_pz + A_mz) / 6.0, alpha);
    let smoothed_py = mix(A_py, (A_py + A_c + A_px + A_mx + A_pz + A_mz) / 6.0, alpha);
    let smoothed_my = mix(A_my, (A_my + A_c + A_px + A_mx + A_pz + A_mz) / 6.0, alpha);
    let smoothed_pz = mix(A_pz, (A_pz + A_c + A_px + A_mx + A_py + A_my) / 6.0, alpha);
    let smoothed_mz = mix(A_mz, (A_mz + A_c + A_px + A_mx + A_py + A_my) / 6.0, alpha);

    let curl_x = (smoothed_py.z - smoothed_my.z) * 0.5 - (smoothed_pz.y - smoothed_mz.y) * 0.5;
    let curl_y = (smoothed_pz.x - smoothed_mz.x) * 0.5 - (smoothed_px.z - smoothed_mx.z) * 0.5;
    let curl_z = (smoothed_px.y - smoothed_mx.y) * 0.5 - (smoothed_py.x - smoothed_mx.x) * 0.5;

    return vec3<f32>(curl_x, curl_y, curl_z);
}

@group(0) @binding(0) var<uniform> params: PhysicsUniforms;
@group(0) @binding(1) var<storage, read_write> f_grid_low: array<f16>;
@group(0) @binding(2) var<storage, read_write> f_grid_high: array<f16>;
@group(0) @binding(3) var<uniform> atom_params: BlueprintParams;
@group(0) @binding(4) var<storage, read_write> macro_state: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> telemetry: ShaderTelemetry;
@group(0) @binding(6) var<storage, read_write> zpe_field: array<vec4<f16>>;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>)
{
    let x = global_id.x; let y = global_id.y; let z = global_id.z;
    if (x >= params.width || y >= params.height || z >= params.depth) { return; }

    let i = get_idx(x, y, z);
    let base_idx = i * 20u;
    let is_even_tick = (params.tick % 2u) == 0u;

    // ====================================================================
    // 2. FETCH STATE (True Bailey A-A Open Void - Boiling Vacuum)
    // ====================================================================
    var rho: f32 = 0.0;
    var ux: f32 = 0.0; var uy: f32 = 0.0; var uz: f32 = 0.0;
    var f_local: array<f32, 39>;
    let prev_rho = max(macro_state[i].x, 0.01);

    for (var d = 0u; d < 39u; d++) {
        var f_val = 0.0;

        let mx = i32(x) - i32(ex[d]);
        let my = i32(y) - i32(ey[d]);
        let mz = i32(z) - i32(ez[d]);

        // Check if the source neighbor exists
        if (mx >= 0 && mx < i32(params.width) &&
            my >= 0 && my < i32(params.height) &&
            mz >= 0 && mz < i32(params.depth)) {

            var read_dir = d;
            if (!is_even_tick) { read_dir = get_inverse_direction(d); }

            // 🌟 Split Read & Cast (f16 -> f32)
            if (read_dir < 20u) {
                f_val = f32(f_grid_low[base_idx + read_dir]);
            } else {
                f_val = f32(f_grid_high[base_idx + (read_dir - 20u)]);
            }
        } else
        {
            // 🌟 INFINITE BOILING VOID INJECTION
//            let void_xi = get_xi(u32(mx), u32(my), u32(mz), params.tick);
//            let void_u = void_xi * params.zpe_amplitude;
//            let c = vec3<f32>(ex[d], ey[d], ez[d]);

            // 🌟 REVERTED: Back to strict 1.0 vacuum pressure sink
            f_val = w[d] * 1.0;
        }

        f_local[d] = f_val;
        rho += f_val;
        ux += f_val * ex[d]; uy += f_val * ey[d]; uz += f_val * ez[d];
    }

    var safe_rho = max(abs(rho), 0.01);
    let inv_rho = 1.0 / safe_rho;
    ux *= inv_rho; uy *= inv_rho; uz *= inv_rho;
    macro_state[i] = vec4<f32>(rho, ux, uy, uz);

    let cx_grid = f32(params.width) * 0.5;
    let cy_grid = f32(params.height) * 0.5;
    let cz_grid = f32(params.depth) * 0.5;
    let dx_grid = f32(x) - cx_grid; let dy_grid = f32(y) - cy_grid; let dz_grid = f32(z) - cz_grid;
    let r_dist = sqrt(dx_grid*dx_grid + dy_grid*dy_grid + dz_grid*dz_grid);

//    let max_radius = cx_grid - 2.0;

//    let xi = get_xi(x, y, z, params.tick);
//    let target_zpe_u = xi * params.zpe_amplitude;
//    let zpe_coupling_rate = params.zpe_coupling_rate;
//
//    var ext_fx = (target_zpe_u.x - ux) * zpe_coupling_rate * safe_rho;
//    var ext_fy = (target_zpe_u.y - uy) * zpe_coupling_rate * safe_rho;
//    var ext_fz = (target_zpe_u.z - uz) * zpe_coupling_rate * safe_rho;

    // 🌟 TRUE GLOBAL PROPAGATION
    var ext_fx = 0.0;
    var ext_fy = 0.0;
    var ext_fz = 0.0;

    let dist_x = min(f32(x), f32(params.width) - 1.0 - f32(x));
    let dist_y = min(f32(y), f32(params.height) - 1.0 - f32(y));
    let dist_z = min(f32(z), f32(params.depth) - 1.0 - f32(z));
    let min_dist_to_edge = min(min(dist_x, dist_y), dist_z);

    // 🌟 Evaluate everywhere EXCEPT inside the deadening sponge
    if (min_dist_to_edge >= sponge_thickness)
    {
        let px_idx = get_idx((x + 1u) % params.width, y, z);
        let mx_idx = get_idx((x + params.width - 1u) % params.width, y, z);
        let py_idx = get_idx(x, (y + 1u) % params.height, z);
        let my_idx = get_idx(x, (y + params.height - 1u) % params.height, z);
        let pz_idx = get_idx(x, y, (z + 1u) % params.depth);
        let mz_idx = get_idx(x, y, (z + params.depth - 1u) % params.depth);

        let A_px = vec3<f32>(zpe_field[px_idx].xyz); let A_mx = vec3<f32>(zpe_field[mx_idx].xyz);
        let A_py = vec3<f32>(zpe_field[py_idx].xyz); let A_my = vec3<f32>(zpe_field[my_idx].xyz);
        let A_pz = vec3<f32>(zpe_field[pz_idx].xyz); let A_mz = vec3<f32>(zpe_field[mz_idx].xyz);

        let curl_x = (A_py.z - A_my.z) * 0.5 - (A_pz.y - A_mz.y) * 0.5;
        let curl_y = (A_pz.x - A_mz.x) * 0.5 - (A_px.z - A_mx.z) * 0.5;
        let curl_z = (A_px.y - A_mx.y) * 0.5 - (A_py.x - A_my.x) * 0.5;

        // 1. Stochastic Vacuum Fluctuation (The active push)
        let vacuum_fluctuation = vec3<f32>(curl_x, curl_y, curl_z) * params.zpe_amplitude;

        // 2. Higgs-Coupled Drag (The passive friction against extreme velocity)
        let vacuum_drag = -vec3<f32>(ux, uy, uz) * params.zpe_coupling_rate;

        // 🌟 Add them completely independently!
        ext_fx += (vacuum_fluctuation.x + vacuum_drag.x) * safe_rho;
        ext_fy += (vacuum_fluctuation.y + vacuum_drag.y) * safe_rho;
        ext_fz += (vacuum_fluctuation.z + vacuum_drag.z) * safe_rho;
    }

    let pos = vec3<f32>(f32(x), f32(y), f32(z));
    var phase_force = vec3<f32>(0.0, 0.0, 0.0);

    let px = u32(clamp(i32(x) + 1, 0, i32(params.width) - 1));
    let mx = u32(clamp(i32(x) - 1, 0, i32(params.width) - 1));
    let py = u32(clamp(i32(y) + 1, 0, i32(params.height) - 1));
    let my = u32(clamp(i32(y) - 1, 0, i32(params.height) - 1));
    let pz = u32(clamp(i32(z) + 1, 0, i32(params.depth) - 1));
    let mz = u32(clamp(i32(z) - 1, 0, i32(params.depth) - 1));

    // ====================================================================
    // STAGE 4: SOLITON INJECTION AND TRAP
    // ====================================================================

//    let st_c = macro_state[i];
//    let rho_c = macro_state[i].x;
    let safe_rho_val = max(macro_state[i].x, 0.01);
    var dist_to_core = bitcast<f32>(0x7f7fffffu);

    if (params.trap_blend != 0.0)
    {
        dist_to_core = distance(pos, atom_params.core_pos.xyz);

        if (dist_to_core < params.r_macro * 4.0)
        {
            var min_r_sq: f32 = 1000000.0;
            var closest_knot_pos = vec3<f32>(0.0);
            var closest_tangent = vec3<f32>(0.0);

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

                // 🌟 GOLDEN RATIO TILT
                let tilt_theta = 0.61803;
                let tilt_phi = 1.0;

                let cos_t = cos(tilt_theta); let sin_t = sin(tilt_theta);
                let cos_p = cos(tilt_phi);   let sin_p = sin(tilt_phi);

                let y_prime = outward_normal.y * cos_t - outward_normal.z * sin_t;
                let z_prime = outward_normal.y * sin_t + outward_normal.z * cos_t;
                let x_prime = outward_normal.x * cos_p + z_prime * sin_p;
                let z_final = -outward_normal.x * sin_p + z_prime * cos_p;

                outward_normal = normalize(vec3<f32>(x_prime, y_prime, z_final));

                var basis_u = cross(vec3<f32>(0.0, 1.0, 0.0), outward_normal);
                if (length(basis_u) < 0.01) { basis_u = cross(vec3<f32>(1.0, 0.0, 0.0), outward_normal); }
                basis_u = normalize(basis_u);
                let basis_v = cross(outward_normal, basis_u);

                let f_macro = atom_params.ring_props[r].z;
                let f_minor = atom_params.ring_props[r].w;

                let R_total = radius;
                let R_macro = R_total * f_macro;
                let R_minor = R_total * f_minor;

                let num_segments = 180u;
                let ring_dt = 6.2831853 / f32(num_segments);

                // 🌟 1. FIND THE NEAREST POINT ON THE KNOT SKELETON
                for (var seg = 0u; seg < num_segments; seg++)
                {
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
                    let diff = pos - r_knot;
                    let r_mag_sq = dot(diff, diff);

                    if (r_mag_sq < min_r_sq) {
                        min_r_sq = r_mag_sq;
                        closest_knot_pos = r_knot;
                        closest_tangent = d_vec; // The flow path of the tube
                    }
                }
            }

            let r_local = sqrt(min_r_sq);
            var target_u = vec3<f32>(0.0);

            // =================================================================
            // COMPACT SUPPORT LIMIT (Calculate Target Fields)
            // =================================================================
            if (r_local > 0.0001 && r_local < params.r_macro * 3.0)
            {
                // Establish the local Frenet-Serret Frame
                let T = normalize(closest_tangent);               // Tangent
                let N = normalize(pos - closest_knot_pos);        // Normal (Radial out)
                let B = cross(T, N);                              // Binormal (Azimuthal Swirl)

                let r_core = params.r_minor;

                // 🌟 3. TRUE LAMB-OSEEN VORTICITY PROFILE
                let gaussian_drop = 1.0 - exp(-(r_local * r_local) / (r_core * r_core));
                let u_theta = params.u_core_lu * (r_core / r_local) * gaussian_drop;

                target_u = u_theta * B;

                // The analytical axial push velocity
                let gaussian_core_jet = exp(-(r_local * r_local) / (r_core * r_core));
                target_u += params.u_core_lu * params.u_axial_coeff * gaussian_core_jet * T;
            }

            // =================================================================
            // THE COUPLED SOWING PROTOCOL & STATOR (Apply Structural Overwrite)
            // =================================================================
            let cx = f32(params.width) * 0.5;
            let cy = f32(params.height) * 0.5;
            let cz = f32(params.depth) * 0.5;
            let dist_from_center = distance(pos, vec3<f32>(cx, cy, cz));

            // Fades smoothly to 0 starting 20 voxels from the grid edge
            let safe_radius = min(cx, min(cy, cz)) - 20.0;
            let far_field_fade = clamp(1.0 - tanh((dist_from_center - safe_radius) / 8.0), 0.0, 1.0);

            // Master blend coefficient
            let active_blend = params.phase_lock_rate * atom_params.blend_factor;
            let final_blend  = active_blend * far_field_fade;

            // Only modify the lattice if the motor is actively transmitting energy
            if (final_blend > 0.001 && length(target_u) > 0.0001)
            {
                let r_core = params.r_minor;

                // 1. Overwrite Macroscopic Density (Carve the Trench)
                let target_rho = mix(0.15, 1.0, clamp(r_local / (r_core * 1.5), 0.0, 1.0));
                let rho_diff = target_rho - safe_rho;

                rho += rho_diff * final_blend * 0.5;
                safe_rho = max(abs(rho), 0.01);

                // 2. Overwrite Macroscopic Velocity (The Hard-Lock Stator)
                // Safely bound the target to the speed limit
                let limited_target_u = normalize(target_u) * atom_params.u_core_lu * tanh(length(target_u) / atom_params.u_core_lu);
                ux = mix(ux, limited_target_u.x, final_blend);
                uy = mix(uy, limited_target_u.y, final_blend);
                uz = mix(uz, limited_target_u.z, final_blend);

                // Commit directly to macro state for the diagnostics and renderer
                macro_state[i] = vec4<f32>(rho, ux, uy, uz);

                // 3. Re-align Microscopic Populations (The Stability Anchor)
                // Overwrite the underlying populations to match the new equilibrium.
                // This stops the collision operator from interpreting the injection as a shockwave.
                let U_sq = ux*ux + uy*uy + uz*uz;
                for (var d = 0u; d < 39u; d++) {
                    let c = vec3<f32>(ex[d], ey[d], ez[d]);
                    let cU = dot(c, vec3<f32>(ux, uy, uz));
                    let eq_2nd = 1.5 * cU + 1.125 * (cU * cU) - 0.75 * U_sq;

                    let f_target = w[d] * safe_rho * (1.0 + eq_2nd);

                    // Seamlessly blend the underlying vectors into the hard-lock state
                    f_local[d] = mix(f_local[d], f_target, final_blend * 0.5);
                }
            }
        }

        // ====================================================================
        // 🌟 SPHERICAL TITANIUM TRAP (The Collider Anchor)
        // ====================================================================
        let density_variance = abs(safe_rho_val - 1.0);

        // 🌟 THE FIX: Anchor the trap strictly to the walls of the knot
        if (density_variance > 0.05)
        {
            let cx = f32(params.width) * 0.5;
            let cy = f32(params.height) * 0.5;
            let cz = f32(params.depth) * 0.5;

            // 🌟 PURE SPHERICAL SCALING
            let true_disp = vec3<f32>(cx - f32(x), cy - f32(y), cz - f32(z));
            let dist = length(true_disp);

            // 8-voxel deadzone allows it to tumble freely in the center
            let deadzone = 45.0;

            if (dist > deadzone)
            {
                let trap_stiffness = 0.005;
                let active_disp = true_disp * ((dist - deadzone) / dist);

                // Calculate the raw un-clamped force
                var trap_force = active_disp * trap_stiffness * density_variance * params.trap_blend;

                // 🌟 THE FIX: Force Magnitude Safety Clamp
                // Prevents extreme boundary voxels from accelerating past the speed of sound.
                let max_trap_force = 0.002;
                let force_mag = length(trap_force);

                if (force_mag > max_trap_force) {
                    trap_force = (trap_force / force_mag) * max_trap_force;
                }

                ext_fx += trap_force.x;
                ext_fy += trap_force.y;
                ext_fz += trap_force.z;
            }
        }
    }

    // ====================================================================
    // 🌟 STAGE 4: HIGH-ORDER ISOTROPIC KORTEWEG COHESION
    // ====================================================================
    let psi_c = 1.0 - exp(-safe_rho_val);
    var korteweg_force = vec3<f32>(0.0, 0.0, 0.0);

    for (var d = 0u; d < 39u; d++)
    {
        let nx = i32(x) + i32(ex[d]);
        let ny = i32(y) + i32(ey[d]);
        let nz = i32(z) + i32(ez[d]);

        // 🌟 FIXED: Read securely via the A-A streaming pattern
        let n_rho = get_rho_safe(nx, ny, nz, params.tick);

        let psi_neighbor = 1.0 - exp(-n_rho);
        let c = vec3<f32>(ex[d], ey[d], ez[d]);
        korteweg_force += w[d] * psi_neighbor * c;
    }

    // Apply full, un-projected cohesive surface tension
    korteweg_force *= -params.shan_chen_g * psi_c * 1.5;

    // 🌟 THE FIX: The velocity-projection block is removed.
    // Surface tension remains perfectly isotropic, balancing the internal
    // thermodynamic pressure equally across all axes and neutralizing the artificial Z-pull.

    ext_fx += korteweg_force.x;
    ext_fy += korteweg_force.y;
    ext_fz += korteweg_force.z;

    // ====================================================================
    // 5. PURE MINKOWSKI HYDRODYNAMICS (No Clamps)
    // ====================================================================
    const c_light = 0.81649658; // D3Q39 Speed of Light (sqrt(2/3))
    const c_light_sq = 0.66666667;

    let raw_speed = sqrt(ux*ux + uy*uy + uz*uz);

    // Evaluate current frame's Lorentz factor to scale forces
    let alpha_speed = raw_speed / c_light;
    let current_lorentz = cosh(alpha_speed);

    // Apply external forces to the relativistic momentum
    var px_mom = (ux * current_lorentz) + (ext_fx / safe_rho);
    var py_mom = (uy * current_lorentz) + (ext_fy / safe_rho);
    var pz_mom = (uz * current_lorentz) + (ext_fz / safe_rho);

    let p_raw_sq = px_mom*px_mom + py_mom*py_mom + pz_mom*pz_mom;

    // 🌟 NATURAL ASYMPTOTE: Velocity natively bounds itself to c
    let inv_new_lorentz = inverseSqrt(1.0 + p_raw_sq / c_light_sq);
    let final_lorentz = 1.0 / inv_new_lorentz; // Only needed if telemetry uses it

    ux = px_mom * inv_new_lorentz;
    uy = py_mom * inv_new_lorentz;
    uz = pz_mom * inv_new_lorentz;

    // ====================================================================
    // 6. RLBM RECURSIVE REGULARIZATION (3rd-Order Hermite Manifold)
    // ====================================================================
    var Pi: mat3x3<f32>;

    let U_x = ux * final_lorentz; let U_y = uy * final_lorentz; let U_z = uz * final_lorentz;
    let U_sq = U_x*U_x + U_y*U_y + U_z*U_z;

    // 1. Compute the raw 2nd-order non-equilibrium stress tensor (Pi)
    for (var d = 1u; d < 39u; d++) {
        let c = vec3<f32>(ex[d], ey[d], ez[d]);
        let cU = dot(c, vec3<f32>(U_x, U_y, U_z));
        let cU2 = cU * cU;

        // 🌟 THE ISOTROPIC FIX: Exact 2nd-Order Polynomial
        // Prevents anisotropic high-order Taylor leakage from breaking Pi
        let eq_2nd = 1.5 * cU + 1.125 * cU2 - 0.75 * U_sq;
        let f_eq_raw = w[d] * safe_rho * (1.0 + eq_2nd);

        let f_neq = f_local[d] - f_eq_raw;

        Pi[0][0] += f_neq * c.x * c.x; Pi[1][1] += f_neq * c.y * c.y; Pi[2][2] += f_neq * c.z * c.z;
        Pi[0][1] += f_neq * c.x * c.y; Pi[0][2] += f_neq * c.x * c.z; Pi[1][2] += f_neq * c.y * c.z;
    }
    Pi[1][0] = Pi[0][1]; Pi[2][0] = Pi[0][2]; Pi[2][1] = Pi[1][2];

    let trace_Pi_sq = Pi[0][0]*Pi[0][0] + Pi[1][1]*Pi[1][1] + Pi[2][2]*Pi[2][2] +
                      2.0 * (Pi[0][1]*Pi[0][1] + Pi[0][2]*Pi[0][2] + Pi[1][2]*Pi[1][2]);
    let pi_magnitude = sqrt(max(0.0, trace_Pi_sq));

    //POLARIZATION
    var local_tau = f32(params.tau) + (params.c_polarization * pi_magnitude);

    if (safe_rho < 0.2)
    {
        let target_tau = mix(local_tau, 2.0, clamp((0.2 - safe_rho) / 0.15, 0.0, 1.0));
        local_tau = max(local_tau, target_tau);
    }

    let beta = 1.0 / local_tau;
    let cs_sq = 0.66666667;
    let tr_Pi = Pi[0][0] + Pi[1][1] + Pi[2][2];

    // Handle the rest state (d = 0) fast-path
    // 🌟 FIXED: Complete the rest state (d = 0) fast-path collision
    let eq_0 = -0.75 * U_sq;
    let f_eq_raw_0 = w[0] * safe_rho * (1.0 + eq_0);

    // For c=(0,0,0), c_Pi_c and c_Pi_u are 0.0.
    // proj_3 is 0.0. proj_2 simplifies to exactly this:
    let proj_2_0 = 1.125 * (-cs_sq * tr_Pi);

    // Apply the relaxation writeback to the 0th population!
    f_local[0] = f_eq_raw_0 + (1.0 - beta) * w[0] * proj_2_0;
    // Pi tensor additions for c=(0,0,0) are 0.0, so we skip them!

    // Loop through the active velocity vectors
    // 2. Recursive Reconstruction of Ghost Modes
    for (var d = 1u; d < 39u; d++) {
        let c = vec3<f32>(ex[d], ey[d], ez[d]);
        let cu = dot(c, vec3<f32>(ux, uy, uz));

        let c_Pi_c = dot(c, Pi * c);
        let c_Pi_u = dot(c, Pi * vec3<f32>(ux, uy, uz));

        let proj_2 = 1.125 * (c_Pi_c - cs_sq * tr_Pi);
        let proj_3 = 0.5625 * (3.0 * cu * c_Pi_c - 3.0 * cs_sq * (cu * tr_Pi + 2.0 * c_Pi_u));

        let cU = dot(c, vec3<f32>(U_x, U_y, U_z));
        let cU2 = cU * cU;
        let cU3 = cU2 * cU;

        // 🌟 THE ISOTROPIC FIX: Exact 3rd-Order Polynomial Writeback
        let eq_2nd = 1.5 * cU + 1.125 * cU2 - 0.75 * U_sq;
        let eq_3rd = 0.5625 * cU3 - 1.125 * U_sq * cU;
        let f_eq = w[d] * safe_rho * (1.0 + eq_2nd + eq_3rd);

        f_local[d] = f_eq + (1.0 - beta) * w[d] * (proj_2 + proj_3);
    }

    // ====================================================================
    // 6.5 NON-REFLECTING BOUNDARY CONDITION (True Planar Prism Sponge)
    // ====================================================================

//    let dist_x = min(f32(x), f32(params.width) - 1.0 - f32(x));
//    let dist_y = min(f32(y), f32(params.height) - 1.0 - f32(y));
//    let dist_z = min(f32(z), f32(params.depth) - 1.0 - f32(z));

//    let min_dist_to_edge = min(min(dist_x, dist_y), dist_z);

    if (min_dist_to_edge < sponge_thickness)
    {
        let sponge_factor = 1.0 - (min_dist_to_edge / sponge_thickness);

        // 🌟 THE FIX: Construct a perfectly orthogonal planar normal framework
        var norm_grid = vec3<f32>(0.0, 0.0, 0.0);

        // X-Walls
        if (f32(x) < sponge_thickness) { norm_grid.x = -1.0; }
        else if (f32(x) > f32(params.width) - 1.0 - sponge_thickness) { norm_grid.x = 1.0; }

        // Y-Walls
        if (f32(y) < sponge_thickness) { norm_grid.y = -1.0; }
        else if (f32(y) > f32(params.height) - 1.0 - sponge_thickness) { norm_grid.y = 1.0; }

        // Z-Walls
        if (f32(z) < sponge_thickness) { norm_grid.z = -1.0; }
        else if (f32(z) > f32(params.depth) - 1.0 - sponge_thickness) { norm_grid.z = 1.0; }

        // Normalize to handle corner/edge intersections cleanly
        norm_grid = normalize(norm_grid);

        for (var d = 0u; d < 39u; d++) {
            // 🌟 FIXED: Isotropic relaxation preserves zero net-momentum
            f_local[d] = mix(f_local[d], w[d] * 1.0, sponge_factor * 0.25);
        }
    }

    // ====================================================================
    // X_X Kill Switch for Telemetry (Updated for Dynamic Polarization)
    // ====================================================================
//    let speed_check = sqrt(ux*ux + uy*uy + uz*uz);
//
//    // Check if any discrete population vector has dropped below zero in this node
//    var has_negative_pop: u32 = 0u;
//    for (var d = 0u; d < 39u; d++) {
//        if (f_local[d] < 0.0) {
//            has_negative_pop = 1u;
//        }
//    }
//
//    // Increment global negative population counter if mass inversion occurs
//    if (has_negative_pop == 1u) {
//        atomicAdd(&telemetry.negative_population_count, 1u);
//    }
//
//    // Capture peak values frame-wide using atomic maximums so you can monitor
//    // the polarization layer even when the speed tripwire isn't triggered.
//    // Note: Since bitcast-based atomicMax for f32 requires extensions, we can
//    // conditionally record or allow the map-reduce pipeline to handle global maximums.
//
//    if (speed_check > 0.817 || safe_isnan(speed_check)) {
//        var expected: u32 = 0u;
//        if (atomicCompareExchangeWeak(&telemetry.tripwire_triggered, expected, 1u).exchanged) {
//            // Coordinate and Primitive State
//            telemetry.debug_x = f32(x);
//            telemetry.debug_y = f32(y);
//            telemetry.debug_z = f32(z);
//            telemetry.local_rho = safe_rho;
//            telemetry.local_ux = ux;
//            telemetry.local_uy = uy;
//            telemetry.local_uz = uz;
//
//            // Macroscopic Forces & Relativistic Manifold
//            telemetry.korteweg_fx = korteweg_force.x;
//            telemetry.phase_lock_fx = phase_force.x;
//            telemetry.lorentz_factor = final_lorentz;
//
//            // 🌟 NEW QUANTUM VACUUM DIAGNOSTICS
//            // Capture the exact polarization state that caused or accompanied the failure
//            telemetry.max_pi_magnitude = pi_magnitude;
//            telemetry.peak_polarized_tau = local_tau;
//        }
//    }

    // ====================================================================
    // 7. IN-PLACE STREAMING AND WRITEBACK (True Bailey A-A Race-Free)
    // ====================================================================
    for (var d = 0u; d < 39u; d++) {
        let px = i32(x) + i32(ex[d]);
        let py = i32(y) + i32(ey[d]);
        let pz = i32(z) + i32(ez[d]);

        if (px >= 0 && px < i32(params.width) &&
            py >= 0 && py < i32(params.height) &&
            pz >= 0 && pz < i32(params.depth)) {

            let write_idx = get_idx(u32(px), u32(py), u32(pz)) * 20u; // 🌟 20u Stride!

            var write_dir = d;
            if (is_even_tick) { write_dir = get_inverse_direction(d); }

            // 🌟 Split Write & Cast (f32 -> f16)
            if (write_dir < 20u) {
                f_grid_low[write_idx + write_dir] = f16(f_local[d]);
            } else {
                f_grid_high[write_idx + (write_dir - 20u)] = f16(f_local[d]);
            }
        }
        // If out-of-bounds, the data radiates cleanly out into the void.
    }
}