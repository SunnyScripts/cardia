struct Camera {
    camera_pos: vec3<f32>,
    inv_view_proj: mat4x4<f32>,
    resolution: vec2<f32>,
    density_gain: f32,
    quality_step: f32, // 🌟 NEW: Adaptive sampling multiplier
};

@group(0) @binding(0) var<uniform> cam: Camera;
@group(0) @binding(1) var volume_tex: texture_3d<f32>;
@group(0) @binding(2) var vol_sampler: sampler;
@group(0) @binding(3) var out_texture: texture_storage_2d<rgba8unorm, write>;

// Branchless Ray-AABB Intersection (Slab Method)
fn intersect_aabb(ro: vec3<f32>, rd: vec3<f32>, box_min: vec3<f32>, box_max: vec3<f32>) -> vec2<f32> {
    let inv_rd = 1.0 / rd;
    let t0 = (box_min - ro) * inv_rd;
    let t1 = (box_max - ro) * inv_rd;

    let t_min = min(t0, t1);
    let t_max = max(t0, t1);

    let t_near = max(max(t_min.x, t_min.y), t_min.z);
    let t_far = min(min(t_max.x, t_max.y), t_max.z);

    return vec2<f32>(t_near, t_far);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = f32(global_id.x);
    let y = f32(global_id.y);
    if (x >= cam.resolution.x || y >= cam.resolution.y) { return; }

    // 1. Generate Camera Ray
    let uv = vec2<f32>(x / cam.resolution.x, y / cam.resolution.y);
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0); // Normalized Device Coordinates

    // Unproject to world space
    let look_at = cam.inv_view_proj * vec4<f32>(ndc.x, ndc.y, 1.0, 1.0);
    let ray_dir = normalize((look_at.xyz / look_at.w) - cam.camera_pos);
    let ray_origin = cam.camera_pos;

    // 2. Define the Fluid Bounding Box (Now a perfect cube!)
    let box_min = vec3<f32>(-1.0, -1.0, -1.0);
    let box_max = vec3<f32>( 1.0,  1.0,  1.0);

    // 3. Intersect the Ray with the Volume
    let hits = intersect_aabb(ray_origin, ray_dir, box_min, box_max);
    let t_near = max(hits.x, 0.0);
    let t_far = hits.y;

    var final_color = vec3<f32>(0.0);
    var final_alpha = 0.0;

    if (t_near < t_far)
    {
        let noise = fract(sin(dot(ndc, vec2<f32>(12.9898, 78.233))) * 43758.5453);

        // 🌟 FAST MODE: Multiply the base step size by the quality_step uniform.
        // Moving = 3.0 (Fast/Blocky). Still = 1.0 (High Quality).
        let base_step = 0.040;
        let step_size = base_step * max(1.0, cam.quality_step);
        let t_start = t_near + noise * step_size;

        for (var t = t_start; t < t_far; t += step_size) {
            let p = ray_origin + ray_dir * t;

            let uvw = vec3<f32>(
                (p.x - box_min.x) / (box_max.x - box_min.x),
                (p.y - box_min.y) / (box_max.y - box_min.y),
                (p.z - box_min.z) / (box_max.z - box_min.z)
            );

            let voxel = textureSampleLevel(volume_tex, vol_sampler, uvw, 0.0);

            if (voxel.a < 0.01) { continue; }

            // Optical density calculation
            let optical_density = (voxel.a * voxel.a) * cam.density_gain * 5.0;
            let transmittance = exp(-optical_density * step_size);

            let emitted_color = voxel.rgb * optical_density * step_size;
            final_color += emitted_color * (1.0 - final_alpha);
            final_alpha += (1.0 - transmittance) * (1.0 - final_alpha);

            if (final_alpha >= 0.95) { break; }
        }
    }

    let background = vec3<f32>(0.0, 0.0, 0.0);
    let composite = final_color + background * (1.0 - final_alpha);

    textureStore(out_texture, vec2<i32>(i32(x), i32(y)), vec4<f32>(composite, 1.0));
}