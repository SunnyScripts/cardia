struct PhysicsUniforms {
    width: u32, height: u32, depth: u32, tick: u32,
    tau: f32, zpe_amplitude: f32, c_polarization: f32, u_core_lu: f32,

    u_axial_coeff: f32, coherence_length_lu: f32, shan_chen_g: f32, phase_lock_rate: f32,

    inverse_directions: array<vec4<u32>, 10>,
    r_minor: f32, r_macro: f32,
    zpe_coupling_rate: f32, trap_blend: f32
}

struct DiagUniforms { prev_cx: f32, prev_cy: f32, prev_cz: f32, prev_max_variance: f32, }

struct GpuMetrics {
    nan_count: f32, inf_count: f32,
    sum_ux: f32, sum_uy: f32, sum_uz: f32,
    sum_ux_sq: f32, sum_uy_sq: f32, sum_uz_sq: f32,
    total_mass: f32, peak_density: f32,
    net_mom_x: f32, net_mom_y: f32, net_mom_z: f32,
    total_kinetic_e: f32, max_speed_sq: f32,
    ambient_center_noise: f32,
    sum_x: f32, sum_y: f32, sum_z: f32,
    sum_x_sq: f32, sum_y_sq: f32, sum_z_sq: f32,
    knot_weight: f32, knot_volume: f32,
    knot_mom_z: f32, vacuum_mom_z: f32,
    gl_energy_integral: f32, knot_kinetic_e: f32,
    knot_peak_rho: f32, knot_min_rho: f32,
    inner_rho: f32, inner_count: f32,
    outer_rho: f32, outer_count: f32,
    boundary_echo: f32,
    orbital_wave_amplitude: f32, orbital_node_count: f32,
    max_vorticity_mag: f32, total_enstrophy: f32, total_helicity: f32,
    ang_mom_x: f32, ang_mom_y: f32, ang_mom_z: f32,
    local_core_weight: f32,
    net_gl_force_mag: f32, net_zpe_force_mag: f32,
    max_gl_force: f32,
    quantum_circulation: f32,
    radial_bins: array<f32, 120>,
    gravitational_flux: f32,
    _padding: array<f32, 3>,
}

@group(0) @binding(0) var<uniform> params: PhysicsUniforms;
@group(0) @binding(1) var<storage, read> macro_state: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> out_intermediate: array<GpuMetrics>;
@group(0) @binding(3) var<storage, read_write> out_final: array<GpuMetrics>;
@group(0) @binding(4) var<uniform> diag_params: DiagUniforms;

const ex = array<f32, 39>(0., 1., -1., 0., 0., 0., 0., 1., -1., 1., -1., 1., -1., 1., -1., 2., -2., 0., 0., 0., 0., 2., -2., 2., -2., 2., -2., 2., -2., 0., 0., 0., 0., 3., -3., 0., 0., 0., 0.);
const ey = array<f32, 39>(0., 0., 0., 1., -1., 0., 0., 1., 1., -1., -1., 1., 1., -1., -1., 0., 0., 2., -2., 0., 0., 2., 2., -2., -2., 0., 0., 0., 0., 2., -2., 2., -2., 0., 0., 3., -3., 0., 0.);
const ez = array<f32, 39>(0., 0., 0., 0., 0., 1., -1., 1., 1., 1., 1., -1., -1., -1., -1., 0., 0., 0., 0., 2., -2., 0., 0., 0., 0., 2., 2., -2., -2., 2., 2., -2., -2., 0., 0., 0., 0., 3., -3.);

fn combine(a: GpuMetrics, b: GpuMetrics) -> GpuMetrics {
    var res = a;
    res.nan_count += b.nan_count; res.inf_count += b.inf_count;
    res.sum_ux += b.sum_ux; res.sum_uy += b.sum_uy; res.sum_uz += b.sum_uz;
    res.sum_ux_sq += b.sum_ux_sq; res.sum_uy_sq += b.sum_uy_sq; res.sum_uz_sq += b.sum_uz_sq;
    res.total_mass += b.total_mass; res.peak_density = max(a.peak_density, b.peak_density);
    res.net_mom_x += b.net_mom_x; res.net_mom_y += b.net_mom_y; res.net_mom_z += b.net_mom_z;
    res.total_kinetic_e += b.total_kinetic_e; res.max_speed_sq = max(a.max_speed_sq, b.max_speed_sq);
    res.ambient_center_noise += b.ambient_center_noise;
    res.sum_x += b.sum_x; res.sum_y += b.sum_y; res.sum_z += b.sum_z;
    res.sum_x_sq += b.sum_x_sq; res.sum_y_sq += b.sum_y_sq; res.sum_z_sq += b.sum_z_sq;
    res.knot_weight += b.knot_weight; res.knot_volume += b.knot_volume;
    res.knot_mom_z += b.knot_mom_z; res.vacuum_mom_z += b.vacuum_mom_z;
    res.gl_energy_integral += b.gl_energy_integral; res.knot_kinetic_e += b.knot_kinetic_e;
    res.knot_peak_rho = max(a.knot_peak_rho, b.knot_peak_rho);
    res.knot_min_rho = min(a.knot_min_rho, b.knot_min_rho);
    res.inner_rho += b.inner_rho; res.inner_count += b.inner_count;
    res.outer_rho += b.outer_rho; res.outer_count += b.outer_count;
    res.boundary_echo += b.boundary_echo;
    res.orbital_wave_amplitude += b.orbital_wave_amplitude; res.orbital_node_count += b.orbital_node_count;
    res.max_vorticity_mag = max(a.max_vorticity_mag, b.max_vorticity_mag);
    res.total_enstrophy += b.total_enstrophy; res.total_helicity += b.total_helicity;
    res.ang_mom_x += b.ang_mom_x; res.ang_mom_y += b.ang_mom_y; res.ang_mom_z += b.ang_mom_z;
    res.local_core_weight += b.local_core_weight;
    res.net_gl_force_mag += b.net_gl_force_mag; res.net_zpe_force_mag += b.net_zpe_force_mag;
    res.max_gl_force = max(a.max_gl_force, b.max_gl_force);
    res.quantum_circulation += b.quantum_circulation;
    res.gravitational_flux += b.gravitational_flux;
    for (var i = 0u; i < 120u; i++) { res.radial_bins[i] += b.radial_bins[i]; }
    return res;
}

//fn get_u(idx: u32, rho: f32) -> vec3<f32> {
//    let b = idx * 40u; // 🌟 ALIGNMENT FIX
//    var u = vec3<f32>(0.0);
//    for(var d=0u; d<39u; d++) { u += vec3<f32>(ex[d], ey[d], ez[d]) * f_grid[b + d]; }
//    return u / max(0.0001, rho);
//}

//fn get_rho(idx: u32) -> f32 {
//    let b = idx * 40u; // 🌟 ALIGNMENT FIX
//    var r = 0.0;
//    for(var d=0u; d<39u; d++) { r += f_grid[b + d]; }
//    return max(0.0001, r);
//}

const w = array<f32, 39>(
    1./12.,
    1./12., 1./12., 1./12., 1./12., 1./12., 1./12.,
    1./27., 1./27., 1./27., 1./27., 1./27., 1./27., 1./27., 1./27.,
    2./135., 2./135., 2./135., 2./135., 2./135., 2./135.,
    1./432., 1./432., 1./432., 1./432., 1./432., 1./432., 1./432., 1./432., 1./432., 1./432., 1./432., 1./432.,
    1./1620., 1./1620., 1./1620., 1./1620., 1./1620., 1./1620.
);

fn get_inverse_direction(d: u32) -> u32 {
    let vec_idx = d / 4u;
    let comp_idx = d % 4u;
    let v = params.inverse_directions[vec_idx];
    if (comp_idx == 0u) { return v.x; }
    if (comp_idx == 1u) { return v.y; }
    if (comp_idx == 2u) { return v.z; }
    return v.w;
}

// 🌟 TRUE A-A SAFE FETCHER FOR DIAGNOSTICS
//fn fetch_node_state(x: i32, y: i32, z: i32, tick: u32) -> vec4<f32> {
//    // 1. Hard box guard: if the queried node itself is out of bounds, return pure vacuum
//    if (x < 0 || x >= i32(params.width) || y < 0 || y >= i32(params.height) || z < 0 || z >= i32(params.depth)) {
//        return vec4<f32>(1.0, 0.0, 0.0, 0.0);
//    }
//
//    let is_even = (tick % 2u) == 0u;
//    let b = (u32(z) * (params.width * params.height) + u32(y) * params.width + u32(x)) * 40u;
//
//    var rho = 0.0; var ux = 0.0; var uy = 0.0; var uz = 0.0;
//
//    for (var d = 0u; d < 39u; d++) {
//        var f_val = 0.0;
//
//        // Calculate the spatial source coordinates for this velocity vector
//        let src_x = x - i32(ex[d]);
//        let src_y = y - i32(ey[d]);
//        let src_z = z - i32(ez[d]);
//
//        // 🌟 ISOTROPIC BOUNDARY COUPLING
//        // Check if the streaming source voxel exists within the grid container
//        if (src_x >= 0 && src_x < i32(params.width) &&
//            src_y >= 0 && src_y < i32(params.height) &&
//            src_z >= 0 && src_z < i32(params.depth)) {
//
//            if (is_even) {
//                // EVEN TICK: Read local distribution vector
//                f_val = f_grid[b + d];
//            } else {
//                // ODD TICK: Read local inverse distribution vector (where neighbor pushed it)
//                let inv_d = get_inverse_direction(d);
//                f_val = f_grid[b + inv_d];
//            }
//        } else {
//            // 🌟 TRUE VOID telemetry: If the source is out-of-bounds,
//            // read the exact baseline background vacuum state.
//            f_val = w[d] * 1.0;
//        }
//
//        rho += f_val;
//        ux += f_val * ex[d]; uy += f_val * ey[d]; uz += f_val * ez[d];
//    }
//    return vec4<f32>(rho, ux, uy, uz);
//}

fn fetch_node_state(x: i32, y: i32, z: i32) -> vec4<f32> {
    if (x < 0 || x >= i32(params.width) || y < 0 || y >= i32(params.height) || z < 0 || z >= i32(params.depth)) {
        return vec4<f32>(1.0, 0.0, 0.0, 0.0);
    }
    let idx = u32(z) * (params.width * params.height) + u32(y) * params.width + u32(x);
    return macro_state[idx];
}

@compute @workgroup_size(1, 1, 1)
fn map_blocks(@builtin(global_invocation_id) gid: vec3<u32>) {
    var acc: GpuMetrics;
    acc.knot_min_rho = 1000.0;

    let w = params.width; let h = params.height; let d = params.depth;
    let cx_prev = diag_params.prev_cx; let cy_prev = diag_params.prev_cy; let cz_prev = diag_params.prev_cz;

    // 🌟 DYNAMIC GEOMETRIC BOUNDS
    // Derived strictly from the physical footprint of the knot
    let R_outer = params.r_macro + params.r_minor;
    let R_inner = max(1.0, params.r_macro - params.r_minor);

    let center_noise_bound = (R_inner * 0.5) * (R_inner * 0.5);
    let inner_hole_bound = R_inner * R_inner;
    let outer_shell_bound = R_outer * R_outer;

    let far_field = R_outer * 1.5;
    let far_field_sq = far_field * far_field;

    let orbital_inner = R_outer * 1.5;
    let orbital_outer = R_outer * 1.7;

    // Safety buffer for diagnostic culling (ensures we catch the whole gradient)
    let diag_bound = R_outer + 15.0;
    let diagnostic_bound_sq = diag_bound * diag_bound;

    let base_x = gid.x * 8u; let base_y = gid.y * 8u; let base_z = gid.z * 4u;

    for (var oz = 0u; oz < 4u; oz++)
    {
        for (var oy = 0u; oy < 8u; oy++) {
            for (var ox = 0u; ox < 8u; ox++) {
                let x = base_x + ox; let y = base_y + oy; let z = base_z + oz;
                if (x >= w || y >= h || z >= d) { continue; }

                let i = z * (w * h) + y * w + x;
//                let b_idx = i * 40u;

                var rho = 0.0; var raw_ux = 0.0; var raw_uy = 0.0; var raw_uz = 0.0;
                // 🌟 FETCH CENTER NODE STATE
                let st_c = fetch_node_state(i32(x), i32(y), i32(z));
                rho = max(0.0001, st_c.x);
                raw_ux = st_c.y; raw_uy = st_c.z; raw_uz = st_c.w;

                let c_light = 0.81649658;
                let raw_speed = sqrt(raw_ux*raw_ux + raw_uy*raw_uy + raw_uz*raw_uz);

                var ux = raw_ux; var uy = raw_uy; var uz = raw_uz;
                if (raw_speed > 1e-9) {
                    let speed_scale = (c_light * tanh(raw_speed / c_light)) / raw_speed;
                    ux *= speed_scale; uy *= speed_scale; uz *= speed_scale;
                }
                let speed_sq = ux*ux + uy*uy + uz*uz;

                let px = (x + 1u)%w; let mx = (x + w - 1u)%w;
                let py = (y + 1u)%h; let my = (y + h - 1u)%h;
                let pz = (z + 1u)%d; let mz = (z + d - 1u)%d;

                // 🌟 FETCH NEIGHBOR STATES SAFELY
                let st_px = fetch_node_state(i32(px), i32(y), i32(z));
                let st_mx = fetch_node_state(i32(mx), i32(y), i32(z));
                let st_py = fetch_node_state(i32(x), i32(py), i32(z));
                let st_my = fetch_node_state(i32(x), i32(my), i32(z));
                let st_pz = fetch_node_state(i32(x), i32(y), i32(pz));
                let st_mz = fetch_node_state(i32(x), i32(y), i32(mz));

                let rho_px = st_px.x; let u_px = vec3<f32>(st_px.y, st_px.z, st_px.w);
                let rho_mx = st_mx.x; let u_mx = vec3<f32>(st_mx.y, st_mx.z, st_mx.w);
                let rho_py = st_py.x; let u_py = vec3<f32>(st_py.y, st_py.z, st_py.w);
                let rho_my = st_my.x; let u_my = vec3<f32>(st_my.y, st_my.z, st_my.w);
                let rho_pz = st_pz.x; let u_pz = vec3<f32>(st_pz.y, st_pz.z, st_pz.w);
                let rho_mz = st_mz.x; let u_mz = vec3<f32>(st_mz.y, st_mz.z, st_mz.w);

                let w_x = (u_py.z - u_my.z)*0.5 - (u_pz.y - u_mz.y)*0.5;
                let w_y = (u_pz.x - u_mz.x)*0.5 - (u_px.z - u_mx.z)*0.5;
                let w_z = (u_px.y - u_mx.y)*0.5 - (u_py.x - u_my.x)*0.5;

                let vort_mag = sqrt(w_x*w_x + w_y*w_y + w_z*w_z);

                acc.sum_ux += ux; acc.sum_uy += uy; acc.sum_uz += uz;
                acc.sum_ux_sq += ux*ux; acc.sum_uy_sq += uy*uy; acc.sum_uz_sq += uz*uz;
                acc.total_mass += rho;
                acc.peak_density = max(acc.peak_density, rho);
                acc.net_mom_x += rho * ux; acc.net_mom_y += rho * uy; acc.net_mom_z += rho * uz;
                acc.total_kinetic_e += 0.5 * rho * speed_sq;
                acc.max_speed_sq = max(acc.max_speed_sq, speed_sq);

                let x_c = f32(x); let y_c = f32(y); let z_c = f32(z);

                let knot_var = abs(rho - 1.0);
                let psi_sq = knot_var * knot_var;

                let dist_cx = x_c - f32(w)*0.5; let dist_cy = y_c - f32(h)*0.5; let dist_cz = z_c - f32(d)*0.5;
                let dist_grid_sq = dist_cx*dist_cx + dist_cy*dist_cy + dist_cz*dist_cz;

                // 🌟 DYNAMIC: Ambient noise measured exactly in the hole
                if (dist_grid_sq <= center_noise_bound && knot_var < params.zpe_amplitude * 5.0) {
                    acc.ambient_center_noise += 0.5 * rho * speed_sq;
                }

                let sig = (0.5 * rho * speed_sq) + psi_sq;
                acc.sum_x += x_c * sig; acc.sum_y += y_c * sig; acc.sum_z += z_c * sig;
                acc.sum_x_sq += x_c*x_c * sig; acc.sum_y_sq += y_c*y_c * sig; acc.sum_z_sq += z_c*z_c * sig;
                acc.knot_weight += sig;

                let d_px = x_c - cx_prev; let d_py = y_c - cy_prev; let d_pz = z_c - cz_prev;
                let r_dist_sq = d_px*d_px + d_py*d_py + d_pz*d_pz;
                let r_dist = sqrt(r_dist_sq);

                let speed_lu = sqrt(speed_sq);

                // 🌟 RELATIVE TOPOLOGICAL CULLING
                // We define the core strictly as the region where the variance is greater than
                // 50% of the maximum measured variance of the soliton, ensuring we only count the
                // dense "wall" of the knot, regardless of the injection ramp stage.
                let dynamic_core_threshold = max(0.0005, diag_params.prev_max_variance * 0.50);
                let is_core = knot_var > dynamic_core_threshold;

                // 🌟 RELATIVE SHEATH CULLING (The Mass Fix)
                // By requiring the variance to be at least 10% of the peak core variance,
                // we cut off the acoustic wake completely and only integrate the Ginzburg-Landau
                // energy that is physically binding the topological structure.
                let dynamic_sheath_threshold = max(0.0001, diag_params.prev_max_variance * 0.10);

                if (r_dist_sq < diagnostic_bound_sq) {

                    // Integrate binding energy ONLY across the true geometric boundary
                    if (knot_var > dynamic_sheath_threshold) {
                        acc.gl_energy_integral += psi_sq;
                    }

                    if (is_core) {
                        acc.knot_volume += 1.0;
                        acc.knot_mom_z += rho * uz;
                        acc.knot_kinetic_e += 0.5 * rho * speed_sq;
                    } else {
                        acc.vacuum_mom_z += rho * uz;
                    }
                } else {
                    acc.vacuum_mom_z += rho * uz;
                }

                let r_idx = u32(r_dist);
                if (r_idx < 120u) { acc.radial_bins[r_idx] += psi_sq; }

                acc.knot_peak_rho = max(acc.knot_peak_rho, rho);
                acc.knot_min_rho = min(acc.knot_min_rho, rho);

                // 🌟 DYNAMIC REGIONS
                if (r_dist_sq < inner_hole_bound) {
                    acc.inner_rho += 0.5 * rho * speed_sq; acc.inner_count += 1.0;
                }
                else if (r_dist_sq > outer_shell_bound && r_dist_sq < far_field_sq) {
                    acc.outer_rho += 0.5 * rho * speed_sq; acc.outer_count += 1.0;
                }

                if (z_c <= 1.0 || z_c >= f32(d - 2u)) { acc.boundary_echo += speed_sq; }

                if (r_dist > orbital_inner && r_dist < orbital_outer) {
                    acc.orbital_wave_amplitude += knot_var; acc.orbital_node_count += 1.0;
                }

                acc.max_vorticity_mag = max(acc.max_vorticity_mag, vort_mag);
                acc.total_enstrophy += rho * (vort_mag*vort_mag);
                acc.total_helicity += ux*w_x + uy*w_y + uz*w_z;

                // 🌟 DYNAMIC CIRCULATION INTEGRAL
                if (r_dist_sq < outer_shell_bound) { acc.quantum_circulation += w_z; }

                let force_weight = (0.5 * rho * speed_sq) + psi_sq;
                if (force_weight > 1e-9) {
                    let p_x = rho * ux; let p_y = rho * uy; let p_z = rho * uz;
                    acc.ang_mom_x += d_py * p_z - d_pz * p_y;
                    acc.ang_mom_y += d_pz * p_x - d_px * p_z;
                    acc.ang_mom_z += d_px * p_y - d_py * p_x;
                    acc.local_core_weight += force_weight;

                    let cs_sq = 0.66666667;
                    let fx_gl = -cs_sq * (rho_px - rho_mx) * 0.5;
                    let fy_gl = -cs_sq * (rho_py - rho_my) * 0.5;
                    let fz_gl = -cs_sq * (rho_pz - rho_mz) * 0.5;

                    let gl_mag = sqrt(fx_gl*fx_gl + fy_gl*fy_gl + fz_gl*fz_gl);

                    acc.net_gl_force_mag += gl_mag * force_weight;
                    acc.max_gl_force = max(acc.max_gl_force, gl_mag);

                    // 🌟 DYNAMIC FLUX BOUNDARY
                    if (r_dist_sq > outer_shell_bound && r_dist_sq < far_field_sq) {
                        let r_norm_x = d_px / r_dist;
                        let r_norm_y = d_py / r_dist;
                        let r_norm_z = d_pz / r_dist;
                        acc.gravitational_flux += (fx_gl * r_norm_x + fy_gl * r_norm_y + fz_gl * r_norm_z);
                    }

                    let zpe_mag = speed_sq * (params.zpe_amplitude*params.zpe_amplitude*1000.0) * rho;
                    acc.net_zpe_force_mag += zpe_mag * force_weight;
                }
            }
        }
    }
    // 🌟 DYNAMIC LINEAR INDEXING: Prevents data race overlaps
    let wgs_x = (params.width + 7u) / 8u;
    let wgs_y = (params.height + 7u) / 8u;
    let linear_id = gid.z * (wgs_x * wgs_y) + gid.y * wgs_x + gid.x;

    out_intermediate[linear_id] = acc;
}

@compute @workgroup_size(1, 1, 1)
fn reduce_global()
{
    var final_m: GpuMetrics;
    final_m.knot_min_rho = 1000.0;

    // 🌟 DYNAMIC REDUCTION: Sum the entire 376^3 grid
    let wgs_x = (params.width + 7u) / 8u;
    let wgs_y = (params.height + 7u) / 8u;
    let wgs_z = (params.depth + 3u) / 4u;
    let total_groups = wgs_x * wgs_y * wgs_z;

    for (var i = 0u; i < total_groups; i++) {
        final_m = combine(final_m, out_intermediate[i]);
    }
    out_final[0] = final_m;
}