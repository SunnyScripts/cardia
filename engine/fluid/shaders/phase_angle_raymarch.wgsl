struct RenderParams {
    width: u32,
    height: u32,
    depth: u32,
    padding: u32,
    noise_threshold: f32,
    wave_threshold: f32,
    matter_threshold: f32,
    alpha_scale: f32,
};

@group(0) @binding(0) var<uniform> params: RenderParams;
@group(0) @binding(1) var<storage, read> psi_3d: array<vec2<f32>>;
@group(0) @binding(2) var out_texture: texture_storage_2d<rgba8unorm, write>;

fn get_idx(x: u32, y: u32, z: u32, w: u32, h: u32) -> u32 {
    return z * (w * h) + y * w + x;
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    let w = params.width;
    let h = params.height;
    let d = params.depth;

    if (x == 0u || x >= w - 1u || y == 0u || y >= h - 1u) { return; }

    // Look ONLY at the dead center Z-slice
    let z = d / 2u;
    let i = get_idx(x, y, z, w, h);

    let psi = psi_3d[i];
    let density = (psi.x * psi.x) + (psi.y * psi.y);

    // -------------------------------------------------------------
    // PHASE STRIPE VISUALIZER (See the Invisible Wind!)
    // -------------------------------------------------------------
    // Calculate the raw phase angle (-PI to PI)
    let phase = atan2(psi.y, psi.x);

    // Create alternating bands of light and dark based on the phase
    // By multiplying by 4.0, we get multiple stripes per wavelength
    let flow_lines = (cos(phase * 4.0) + 1.0) * 0.5;

    // Paint the flowing phase lines in a soft blue
    var final_color = vec4<f32>(flow_lines * 0.3, flow_lines * 0.5, flow_lines * 0.9, 1.0);

    // -------------------------------------------------------------
    // DRAW THE OBSTACLE
    // -------------------------------------------------------------
    // If the fluid density is crushed to 0.0 by our mask, paint it solid black
    if (density < 0.05) {
        final_color = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    // Highlight the turbulent core inside the donut hole in faint Red
    else if (density > 0.05 && density < 0.9) {
        final_color.r += 0.5;
    }

    textureStore(out_texture, vec2<i32>(i32(x), i32(y)), final_color);
}