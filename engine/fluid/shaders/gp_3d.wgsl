struct Uniforms
{
    width: u32,
    height: u32,
    depth: u32,
    dx: f32,
    dt: f32,
    g: f32,
    wall_v: f32,
    flow_velocity: f32,
    time: f32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32
};

@group(0) @binding(0) var<uniform> params: Uniforms;
@group(0) @binding(1) var<storage, read> env_map: array<f32>;
@group(0) @binding(2) var<storage, read> psi_in: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> psi_out: array<vec2<f32>>;

fn get_idx(x: u32, y: u32, z: u32, w: u32, h: u32) -> u32 {
    return z * (w * h) + y * w + x;
}

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    let z = global_id.z;
    let w = params.width;
    let h = params.height;
    let d = params.depth;

    let i = get_idx(x, y, z, w, h);

    // 0. Frictionless Walls
    if (x == 0u || x >= w - 1u || y == 0u || y >= h - 1u || z == 0u || z >= d - 1u) {
        let clamp_x = clamp(x, 1u, w - 2u);
        let clamp_y = clamp(y, 1u, h - 2u);
        let clamp_z = clamp(z, 1u, d - 2u);
        psi_out[i] = psi_in[get_idx(clamp_x, clamp_y, clamp_z, w, h)];
        return;
    }

    // -------------------------------------------------------------
    // 1. The Spatial Quantum Pump (UNTHROTTLED)
    // -------------------------------------------------------------
    if (x <= 1u) {
        // REMOVED the * 0.01 multiplier!
        // A slider value of 2.0 now means Mach 2.0!
        let k = params.flow_velocity;

        // Galilean shifted dispersion
        let omega = 0.5 * k * k;
        let phase = (k * f32(x)) - (omega * params.time);
        psi_out[i] = vec2<f32>(cos(phase), sin(phase));
        return;
    }

    let psi = psi_in[i];

    // 2. The 3D Laplacian
    let neighbor_sum = psi_in[get_idx(x + 1u, y, z, w, h)] +
                       psi_in[get_idx(x - 1u, y, z, w, h)] +
                       psi_in[get_idx(x, y + 1u, z, w, h)] +
                       psi_in[get_idx(x, y - 1u, z, w, h)] +
                       psi_in[get_idx(x, y, z + 1u, w, h)] +
                       psi_in[get_idx(x, y, z - 1u, w, h)];

    let laplacian = (neighbor_sum - (6.0 * psi)) / (params.dx * params.dx);

    // -------------------------------------------------------------
    // 3. THE COMPLEX ABSORBING POTENTIAL (Adiabatic Fade-In)
    // -------------------------------------------------------------
    let cx = f32(w) / 2.0; let cy = f32(h) / 2.0; let cz = f32(d) / 2.0;
    let dx = f32(x) - cx; let dy = f32(y) - cy; let dz = f32(z) - cz;
    let angle = atan2(dy, dx);
    let lobed_radius = 12.0 + 4.0 * sin(3.0 * angle);
    let dist = sqrt(pow(sqrt(dx*dx + dy*dy) - lobed_radius, 2.0) + dz*dz);

    let obstacle_shape = 1.0 - smoothstep(2.0, 5.0, dist);

    // ADIABATIC RAMP: Smoothly fade the knot in over the first 5 seconds.
    // This gently pushes the fluid out of the way instead of detonating it!
    let ramp = smoothstep(0.0, 5.0, params.time);

    let V_ext = 50.0 * obstacle_shape * ramp;
    let Gamma = 10.0 * obstacle_shape * ramp;

    // -------------------------------------------------------------
        // 4. PURE HYBRID SPLIT-STEP (With Thermal Damping)
        // -------------------------------------------------------------
        let density = (psi.x * psi.x) + (psi.y * psi.y);

        let local_g = 0.1;
        let total_v = local_g * (density - 1.0) + V_ext + (env_map[i] * params.wall_v);

        // QUANTUM FRICTION (Phenomenological Damping)
        // gamma_damp simulates the thermal cloud. It physically dampens
        // microscopic phase boiling without killing the macroscopic wake.
        let gamma_damp = 0.01;

        // The dampened kinetic Laplacian split
        let k_real = 0.5 * params.dt * (gamma_damp * laplacian.x - laplacian.y);
        let k_imag = 0.5 * params.dt * (laplacian.x + gamma_damp * laplacian.y);

        let temp_real = psi.x + k_real;
        let temp_imag = psi.y + k_imag;

        let phase_theta = -total_v * params.dt;
        let cos_v = cos(phase_theta);
        let sin_v = sin(phase_theta);

        var final_psi = vec2<f32>(
            temp_real * cos_v - temp_imag * sin_v,
            temp_real * sin_v + temp_imag * cos_v
        );

    // -------------------------------------------------------------
    // 5. IMAGINARY DECAY & THREE-BODY RECOMBINATION
    // -------------------------------------------------------------
    // 1. The Imaginary Potential (Gamma) absorbs fluid inside the knot
    let decay = exp(-Gamma * params.dt);
    final_psi = final_psi * decay;

    // 2. THE PRESSURE RELIEF VALVE: Three-Body Recombination (Quintic Loss)
    // In real BECs, extreme density spikes cause atoms to form molecules and escape.
    // This physically prevents the Forward Euler numerical explosion!
    let current_density = (final_psi.x * final_psi.x) + (final_psi.y * final_psi.y);

    // K3 is the three-body loss coefficient.
    // At density 1.0, the loss is ~0%. At density 5.0, it aggressively dampens the wave.
    let K3 = 0.05;
    let three_body_decay = exp(-K3 * current_density * current_density * params.dt);
    final_psi = final_psi * three_body_decay;

    // 3. Micro-relaxation to prevent ambient Euler drift
    let current_mag = length(final_psi);
    if (current_mag > 0.001) {
        let target_psi = (final_psi / current_mag) * 1.0;
        final_psi = mix(final_psi, target_psi, 0.001 * (1.0 - obstacle_shape));
    }

    // -------------------------------------------------------------
    // 6. PHASE-RELAXING SPONGE
    // -------------------------------------------------------------
    let sponge_dist = f32(w - 1u) - f32(x);
    if (sponge_dist < 20.0)
    {
        let sponge_factor = 1.0 - (sponge_dist / 20.0);
        if (current_mag > 0.001) {
            let target_psi = (final_psi / current_mag) * 1.0;
            final_psi = mix(final_psi, target_psi, sponge_factor * 0.2);
        }
    }

    // -------------------------------------------------------------
    // 7. PASSIVE NEUMANN DRAIN (Fixes the Backward Wind!)
    // -------------------------------------------------------------
    if (x >= params.width - 2u)
    {
        // The fluid simply steps off the grid based on the node immediately
        // to its left. NO forced phase, NO backward pressure.
        psi_out[i] = psi_in[get_idx(x - 1u, y, z, w, h)];
        return;
    }

    psi_out[i] = final_psi;
}