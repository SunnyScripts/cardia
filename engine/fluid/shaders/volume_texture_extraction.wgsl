struct PhysicsUniforms {
    width: u32, height: u32, depth: u32, tick: u32,
    tau: f32, zpe_amplitude: f32, c_polarization: f32, u_core_lu: f32,

    u_axial_coeff: f32, coherence_length_lu: f32, shan_chen_g: f32, phase_lock_rate: f32,

    inverse_directions: array<vec4<u32>, 10>,
    r_minor: f32, r_macro: f32,
    zpe_coupling_rate: f32, trap_blend: f32
}

// 🌟 THE AUTO-EXPOSURE LINK
struct DiagUniforms { prev_cx: f32, prev_cy: f32, prev_cz: f32, prev_max_variance: f32 }

@group(0) @binding(0) var<uniform> params: PhysicsUniforms;
@group(0) @binding(1) var<storage, read> env_map: array<f32>;
@group(0) @binding(2) var<storage, read> macro_state: array<vec4<f32>>;
@group(0) @binding(3) var out_volume: texture_storage_3d<rgba8unorm, write>;
@group(0) @binding(4) var<storage, read_write> accum_buffer: array<f32>;
@group(0) @binding(5) var<uniform> diag_params: DiagUniforms;

fn get_idx(x: u32, y: u32, z: u32) -> u32 {
    return z * (params.width * params.height) + y * params.width + x;
}

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x; let y = global_id.y; let z = global_id.z;
    if (x >= params.width || y >= params.height || z >= params.depth) { return; }

    let i = get_idx(x, y, z);

    if (env_map[i] > 0.5) {
        textureStore(out_volume, vec3<i32>(i32(x), i32(y), i32(z)), vec4<f32>(0.2, 0.2, 0.25, 0.95));
        return;
    }

    let dist_x = min(x, params.width - 1u - x); let dist_y = min(y, params.height - 1u - y); let dist_z = min(z, params.depth - 1u - z);
    if (f32(min(min(dist_x, dist_y), dist_z)) < 12.0) {
        textureStore(out_volume, vec3<i32>(i32(x), i32(y), i32(z)), vec4<f32>(0.0));
        return;
    }

    // 🌟 THE UPDATE: Read density directly from the .x component
    let rho = macro_state[i].x;
    let variance = abs(rho - 1.0);
    let current_accum = accum_buffer[i];

    // Temporal smoothing
    let new_accum = mix(current_accum, variance, 0.05);
    accum_buffer[i] = new_accum;

    // ======================================================================
    // 🌟 AUTO-EXPOSURE THRESHOLDING (Safeguarded)
    // ======================================================================
    let max_var = max(params.zpe_amplitude * 10.0, diag_params.prev_max_variance);

    let noise_floor = params.zpe_amplitude * 3.0;

    // STRICT CLAMPING: Ensures edge1 is always > edge0 for smoothstep
    let sheath_threshold = max(noise_floor * 2.0, max_var * 0.25);
    let core_threshold = max(sheath_threshold * 1.5, max_var * 0.75);

    if (new_accum < noise_floor) {
        textureStore(out_volume, vec3<i32>(i32(x), i32(y), i32(z)), vec4<f32>(0.0));
        return;
    }

    // ======================================================================
    // STABILIZED 3D NORMALS
    // ======================================================================
    let px = (x + 1u) % params.width; let mx = (x + params.width - 1u) % params.width;
    let py = (y + 1u) % params.height; let my = (y + params.height - 1u) % params.height;
    let pz = (z + 1u) % params.depth; let mz = (z + params.depth - 1u) % params.depth;

    let acc_px = accum_buffer[get_idx(px, y, z)]; let acc_mx = accum_buffer[get_idx(mx, y, z)];
    let acc_py = accum_buffer[get_idx(x, py, z)]; let acc_my = accum_buffer[get_idx(x, my, z)];
    let acc_pz = accum_buffer[get_idx(x, y, pz)]; let acc_mz = accum_buffer[get_idx(x, y, mz)];

    let grad_x = (acc_px - acc_mx) * 0.5;
    let grad_y = (acc_py - acc_my) * 0.5;
    let grad_z = (acc_pz - acc_mz) * 0.5;

    var normal = -vec3<f32>(grad_x, grad_y, grad_z);
    let grad_mag = length(normal);
    if (grad_mag > 1e-8) {
        normal = normal / grad_mag;
    } else {
        normal = vec3<f32>(0.0, 1.0, 0.0);
    }

    // Lambertian lighting + Specular Highlight
    let light_dir = normalize(vec3<f32>(1.0, 1.5, 0.5));
    let view_dir = normalize(vec3<f32>(0.0, 0.0, 1.0));
    let half_vector = normalize(light_dir + view_dir);

    let ambient = 0.40; //was 0.2
    let diffuse = max(dot(normal, light_dir), 0.0) * 0.8;
    let specular = pow(max(dot(normal, half_vector), 0.0), 32.0) * 0.4;
    let lighting = ambient + diffuse + specular;

    // ======================================================================
    // SIGNAL-TO-NOISE TRANSFER
    // ======================================================================
    let orbital_signal = smoothstep(noise_floor, sheath_threshold, new_accum);
    let core_signal = smoothstep(sheath_threshold, core_threshold, new_accum);

    // 🌟 UNLOCKED OPTIONAL RENDERING UPGRADE: Kinetic Emission Mapping
    let local_u = macro_state[i].yzw;
    let local_speed = length(local_u);

    // Create a dynamic kinetic hyper-signal based on localized velocity
    let kinetic_signal = smoothstep(0.01, params.u_core_lu, local_speed);

    let fluid_base = vec3<f32>(0.20, 0.02, 0.40); // Deep Purple/Blue for vacuum wake
    let fluid_mid  = vec3<f32>(0.90, 0.20, 0.05); // Crimson Red for the boundary
    let fluid_core = vec3<f32>(1.00, 0.80, 0.15); // Radiant Gold for the topology

    // 🌟 Burn Effect: High-speed regions glow radiant gold regardless of density
    var blended_color = mix(fluid_base, fluid_mid, orbital_signal);
    blended_color = mix(blended_color, fluid_core, max(core_signal, kinetic_signal));
    blended_color *= lighting;

    // ======================================================================
    // OPACITY MAPPING
    // ======================================================================
    // 🌟 BOOSTED: Start at 0.25 so the raymarcher cannot discard the outer sheath
    let base_alpha = mix(0.25, 1.0, core_signal);
    let dynamic_alpha_scale = 1.0;

    var final_opacity = base_alpha * dynamic_alpha_scale;
    final_opacity = clamp(final_opacity, 0.0, 1.0);

    let color = vec4<f32>(blended_color, final_opacity);
    textureStore(out_volume, vec3<i32>(i32(x), i32(y), i32(z)), color);
}