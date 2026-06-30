enable f16;

struct PhysicsUniforms {
    width: u32, height: u32, depth: u32, tick: u32,
    tau: f32, zpe_amplitude: f32, c_polarization: f32, u_core_lu: f32,

    u_axial_coeff: f32, coherence_length_lu: f32, shan_chen_g: f32, phase_lock_rate: f32,

    inverse_directions: array<vec4<u32>, 10>,
    r_minor: f32, r_macro: f32,
    zpe_coupling_rate: f32, trap_blend: f32
}

fn hash_u32(seed: u32) -> u32 {
    var h = seed;
    h ^= h >> 16u; h *= 0x85ebca6bu; h ^= h >> 13u; h *= 0xc2b2ae35u; h ^= h >> 16u;
    return h;
}

@group(0) @binding(0) var<uniform> params: PhysicsUniforms;
@group(0) @binding(1) var<storage, read_write> zpe_field: array<vec4<f16>>;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height || gid.z >= params.depth) { return; }

    let h_base = gid.x * 374761393u + gid.y * 668265263u + gid.z * 3266489917u + params.tick * 2654435761u;

    let f_x = f32(hash_u32(h_base + 13u) >> 8u) / 16777215.0;
    let f_y = f32(hash_u32(h_base + 17u) >> 8u) / 16777215.0;
    let f_z = f32(hash_u32(h_base + 19u) >> 8u) / 16777215.0;

    let idx = gid.z * (params.width * params.height) + gid.y * params.width + gid.x;

    // Save the raw vector potential
    zpe_field[idx] = vec4<f16>(f16((f_x - 0.5) * 2.0), f16((f_y - 0.5) * 2.0), f16((f_z - 0.5) * 2.0), f16(0.0));
}