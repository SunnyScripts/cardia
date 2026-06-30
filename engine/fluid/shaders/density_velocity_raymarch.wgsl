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
// The buffer is now a flat array of 19 floats per node
@group(0) @binding(1) var<storage, read> f_3d: array<f32>;
@group(0) @binding(2) var out_texture: texture_storage_2d<rgba8unorm, write>;

// LBM Direction Vectors
const ex = array<f32, 19>( 0., 1.,-1., 0., 0., 0., 0., 1.,-1., 1.,-1., 1.,-1., 1.,-1., 0., 0., 0., 0.);
const ey = array<f32, 19>( 0., 0., 0., 1.,-1., 0., 0., 1., 1.,-1.,-1., 0., 0., 0., 0., 1.,-1., 1.,-1.);

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

    // Look at the dead center Z-slice
    let z = d / 2u;
    let i = get_idx(x, y, z, w, h);
    let base_idx = i * 19u;

    var rho: f32 = 0.0;
    var ux: f32 = 0.0;
    var uy: f32 = 0.0;

    for (var dir = 0u; dir < 19u; dir++) {
        let f_val = f_3d[base_idx + dir];
        rho += f_val;
        ux += f_val * ex[dir];
        uy += f_val * ey[dir];
    }

    if (rho > 0.0001) {
        ux /= rho;
        uy /= rho;
    }

    let speed = sqrt(ux*ux + uy*uy);

    // --- THE MICROSCOPE EXPOSURE FIX ---
    // Multiply the tiny physical velocities by a massive visual factor.
    // Use your Slint slider (params.alpha_scale) to dial this in live!
    let visual_boost = 1000.0 * params.alpha_scale;

        let display_ux = ux * visual_boost;
        let display_uy = uy * visual_boost;
        let display_speed = speed * visual_boost;

        // Ichor Palette: Blood Red to Mythological Gold
        // Base vacuum is a deep crimson. Kinetic energy blooms into bright gold.
        var final_color = vec4<f32>(
            0.15 + (display_speed * 2.5) + abs(display_ux), // R: High baseline, scales aggressively
            0.00 + (display_speed * 1.8) + display_uy,      // G: Scales up behind red to create gold
            0.00 + (display_speed * 0.1),                   // B: Minimal blue to keep the gold pure
            1.0
        );

        // Visualize the atoms/knots (Density anomalies)
        let density_variance = abs(rho - 1.0) * 500.0;
        if (density_variance > 1.0) {
            // Draw topological knots as searing white-gold anomalies
            final_color += vec4<f32>(density_variance, density_variance * 0.9, density_variance * 0.4, 0.0);
        }

        textureStore(out_texture, vec2<i32>(i32(x), i32(y)), final_color);
}