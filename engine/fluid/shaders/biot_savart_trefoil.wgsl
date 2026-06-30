struct Uniforms {
    width: u32,
    height: u32,
    depth: u32,
    _pad: u32,
};

@group(0) @binding(0) var<uniform> params: Uniforms;
@group(0) @binding(1) var<storage, read_write> f_out: array<f32>;

const ex = array<f32, 19>( 0., 1.,-1., 0., 0., 0., 0., 1.,-1., 1.,-1., 1.,-1., 1.,-1., 0., 0., 0., 0.);
const ey = array<f32, 19>( 0., 0., 0., 1.,-1., 0., 0., 1., 1.,-1.,-1., 0., 0., 0., 0., 1.,-1., 1.,-1.);
const ez = array<f32, 19>( 0., 0., 0., 0., 0., 1.,-1., 0., 0., 0., 0., 1., 1.,-1.,-1., 1., 1.,-1.,-1.);
const w = array<f32, 19>(
    1.0/3.0, 
    1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0,
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0
);

// Parametric Trefoil Knot Curve
fn trefoil(t: f32, scale: f32) -> vec3<f32> {
    return vec3<f32>(
        (sin(t) + 2.0 * sin(2.0 * t)) * scale,
        (cos(t) - 2.0 * cos(2.0 * t)) * scale,
        (-sin(3.0 * t)) * scale
    );
}

// Derivative (Tangent) of the Trefoil Curve
fn trefoil_prime(t: f32, scale: f32) -> vec3<f32> {
    return vec3<f32>(
        (cos(t) + 4.0 * cos(2.0 * t)) * scale,
        (-sin(t) + 4.0 * sin(2.0 * t)) * scale,
        (-3.0 * cos(3.0 * t)) * scale
    );
}

fn get_idx(x: u32, y: u32, z: u32) -> u32 {
    return z * (params.width * params.height) + y * params.width + x;
}

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x; let y = global_id.y; let z = global_id.z;
    if (x >= params.width || y >= params.height || z >= params.depth) { return; }

    let center = vec3<f32>(f32(params.width) / 2.0, f32(params.height) / 2.0, f32(params.depth) / 2.0);
    let p = vec3<f32>(f32(x), f32(y), f32(z)) - center;

    var u = vec3<f32>(0.0, 0.0, 0.0);
    let steps = 256u; // High-resolution integration
    let dt = 6.2831853 / f32(steps); 
    
    // Vortex Parameters
    let gamma = 0.08;       // Circulation strength (must stay strictly subsonic!)
    let scale = 15.0;       // Radius of the knot in grid nodes
    let core_radius = 2.0;  // Prevents division by zero at the exact center of the curve

    // Discrete Biot-Savart Line Integral
    for (var i = 0u; i < steps; i++) {
        let t = f32(i) * dt;
        let l = trefoil(t, scale);
        let dl = trefoil_prime(t, scale) * dt;
        
        let r_vec = p - l;
        let r_mag = length(r_vec);
        
        // Rosenhead-Moore core regularization
        let denom = pow(r_mag * r_mag + core_radius * core_radius, 1.5);
        u += cross(dl, r_vec) / denom;
    }
    
    u *= (gamma / 12.56637); // Gamma / 4PI

    let speed_sq = dot(u, u);
    let i_idx = get_idx(x, y, z);
    let base_idx = i_idx * 19u;
    
    // Inject the generated velocity field back into the 19 momentum packets
    for (var d = 0u; d < 19u; d++) {
        let edot_u = ex[d]*u.x + ey[d]*u.y + ez[d]*u.z;
        let f_eq = w[d] * 1.0 * (1.0 + 3.0 * edot_u + 4.5 * edot_u * edot_u - 1.5 * speed_sq);
        f_out[base_idx + d] = f_eq;
    }
}