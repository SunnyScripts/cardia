use std::sync::Arc;
use wgpu::util::DeviceExt;
use bytemuck::{Pod, Zeroable};
// use glam::{Vec3, Mat4};
// use wgpu::hal::{DynCommandEncoder, DynDevice, DynQueue};
use rayon::prelude::*;
use half::f16;
use wgpu::hal::DynDevice;
// use rand::Rng;
// use rand::SeedableRng;
// use rand::rngs::SmallRng;

use crate::logger::RunLogger;

// ==========================================
// UNIVERSAL PHYSICAL CONSTANTS
// ==========================================
pub const HBAR: f64 = 1.054571817e-34;
pub const C: f64 = 299792458.0;
pub const M_NUCLEON: f64 = 1.674927498e-27;
pub const PI: f64 = std::f64::consts::PI;

pub const C_EFF: f64 = 1000.0;
pub const LBM_MACH_DIVERGENCE_LIMIT: f32 = 0.50; // Increased headroom due to RLBM

// 🌟 D3Q39 CALIBRATION: The native speed of sound is sqrt(2/3)
pub const CS_LATTICE: f64 = 0.81649658092;
pub const MACH_TARGET: f64 = 0.1876;

// --- CRITICAL: ASTROPHYSICAL SCALING MANIFOLD ---
pub const L_NODE: f64 = 1.0e-32;//1.22e-26;
pub const SCALE_FACTOR: f64 = 1.5e9;
pub const DX: f64 = L_NODE * SCALE_FACTOR;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct PhysicsState {
    pub width: u32, pub height: u32, pub depth: u32, pub tick: u32,
    pub tau: f32, pub zpe_amplitude: f32, pub c_polarization: f32, pub u_core_lu: f32,

    pub u_axial_coeff: f32, pub coherence_length_lu: f32, pub shan_chen_g: f32, pub phase_lock_rate: f32,

    pub inverse_directions: [u32; 40],
    pub r_minor: f32, pub r_macro: f32,
    pub zpe_coupling_rate: f32,
    pub trap_blend: f32,
}

impl PhysicsState {
    pub fn derive(width: u32, height: u32, depth: u32, mach_target: f64) -> Self {
        let dx = DX;
        let r_proton_m = 0.8409e-15;
        let coherence_length = r_proton_m / dx;

        let k_scaling = 0.0265_f64;
        let thickness_factor = k_scaling / mach_target;
        let r_minor = coherence_length * thickness_factor;

        let u_core = CS_LATTICE * mach_target;

        let zpe_amplitude_lu = CS_LATTICE / SCALE_FACTOR.sqrt();
        let nu_kss = (1.0 / 3.0) * CS_LATTICE * (1.0 / SCALE_FACTOR);
        let tau = (3.0 * (nu_kss + zpe_amplitude_lu) + 0.5) as f32;
        let tau_f = tau as f64;

        let target_gamma = std::f64::consts::TAU * r_minor * u_core;
        let stability_ratio = (0.5 + (0.25 + (2.0 * std::f64::consts::PI * tau_f / target_gamma)).sqrt()).exp();
        let r_macro = r_minor * stability_ratio;

        // ===========================================================================
        // 🌟 ANALYTICAL BELTRAMI AXIAL FLOW DERIVATION
        // ===========================================================================
        // Derived directly from the (p=3, q=2) torus winding pitch matrix
        let u_axial_coeff = (2.0 * r_minor) / (3.0 * r_macro);

        // Total kinetic energy scales with the combined magnitude of both 3D velocity paths
        let effective_mach_sq = mach_target.powi(2) * (1.0 + u_axial_coeff.powi(2));

        // ===========================================================================
        // 🌟 D3Q39 CALIBRATED KINETIC DERIVATION
        // ===========================================================================
        let g_crit = 3.25_f64;

        // 1. Dynamic Pressure Penalty accounts for the true effective 3D speed
        let g_dynamic = 0.834_f64 * effective_mach_sq;

        // 2. Topological Curvature Penalty
        let curvature_penalty = r_macro / (r_minor * std::f64::consts::PI);

        // 3. ZPE Thermal Dissipation Penalty
        let thermal_penalty = (zpe_amplitude_lu as f64 / CS_LATTICE) / (tau_f - 0.5);

        let g_continuum = g_crit + g_dynamic + curvature_penalty + thermal_penalty;

        let beta_relaxation = 1.0 / tau_f;
        let lattice_compliance = (beta_relaxation - 1.0) / (beta_relaxation + mach_target);
        let g_analytical = g_continuum * lattice_compliance;
        let shan_chen_g = -(g_analytical) as f32;

        let phase_lock_rate = (u_core / r_minor) as f32;
        let zpe_coupling_rate = (zpe_amplitude_lu / tau_f) as f32;
        let base_polarization = 0.45; //Smagorinsky Constant //$$\nu_{\text{sgs}} = (C_s \Delta x)^2 |\bar{S}|$$
        let c_polarization = (base_polarization + (u_axial_coeff * 7.0)) as f32;

        // 🌟 THE 39 VECTORS (6th-Order Isotropy)
        let ex = [
            0.,
            1., -1., 0., 0., 0., 0.,
            1., -1., 1., -1., 1., -1., 1., -1.,
            2., -2., 0., 0., 0., 0.,
            2., -2., 2., -2., 2., -2., 2., -2., 0., 0., 0., 0.,
            3., -3., 0., 0., 0., 0.
        ];
        let ey = [
            0.,
            0., 0., 1., -1., 0., 0.,
            1., 1., -1., -1., 1., 1., -1., -1.,
            0., 0., 2., -2., 0., 0.,
            2., 2., -2., -2., 0., 0., 0., 0., 2., -2., 2., -2.,
            0., 0., 3., -3., 0., 0.
        ];
        let ez = [
            0.,
            0., 0., 0., 0., 1., -1.,
            1., 1., 1., 1., -1., -1., -1., -1.,
            0., 0., 0., 0., 2., -2.,
            0., 0., 0., 0., 2., 2., -2., -2., 2., 2., -2., -2.,
            0., 0., 0., 0., 3., -3.
        ];

        let mut inverse_directions = [0u32; 40];
        for i in 0..39 {
            for j in 0..39 {
                if ex[i] == -ex[j] && ey[i] == -ey[j] && ez[i] == -ez[j] {
                    inverse_directions[i] = j as u32; break;
                }
            }
        }

        Self {
            width, height, depth, tick: 0,
            tau, zpe_amplitude: zpe_amplitude_lu as f32,
            c_polarization, u_core_lu: u_core as f32,

            // 🌟 Assign the derived coefficient to the updated field slot
            u_axial_coeff: u_axial_coeff as f32,

            coherence_length_lu: coherence_length as f32,
            shan_chen_g,
            phase_lock_rate,
            inverse_directions,
            r_minor: r_minor as f32, r_macro: r_macro as f32,
            zpe_coupling_rate,
            trap_blend: 0.0,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct BlueprintState {
    // Block 1
    pub ring_count: u32, pub blend_factor: f32, pub _pad1: f32, pub r_nucleon: f32,
    // Block 2
    pub u_core_lu: f32, pub n_state: f32, pub l_state: f32, pub m_state: f32,
    // Block 3
    pub core_pos: [f32; 4],
    // Arrays
    pub rings: [[f32; 4]; 256],
    pub ring_props: [[f32; 4]; 256],
}

impl BlueprintState {
    pub fn empty() -> Self {
        Self {
            ring_count: 0, blend_factor: 0.0, _pad1: 0.0, r_nucleon: 0.0,
            u_core_lu: 0.0, n_state: 0.0, l_state: 0.0, m_state: 0.0,
            core_pos: [0.0; 4], rings: [[0.0; 4]; 256], ring_props: [[0.0; 4]; 256],
        }
    }

    pub fn inject_nucleus(&mut self, cx: f32, cy: f32, cz: f32, z: u32, n: u32, phys: &PhysicsState) {
        println!("⚛️ Generating Atom Parameters | Z: {}, N: {}", z, n);

        let bond_fm = 1.2e-15;
        let d_nodes = bond_fm / DX as f32;
        let total_nucleons = z + n;

        let mut placed_protons = 0;
        let mut placed_neutrons = 0;
        let phi = std::f32::consts::PI * (3.0 - (5.0_f32).sqrt());

        // 🌟 FRACTIONAL RESOLUTION: Convert absolute LUs to pure ratios for the shader.
        // Setting f_macro to 1.0 establishes r_macro as the outer boundary scale factor,
        // meaning f_minor becomes the exact mathematical ratio between the two profiles.
        let f_macro = 1.0_f32;
        let f_minor = (phys.r_minor / phys.r_macro) as f32;

        for i in 0..total_nucleons {
            let shell_radius = if total_nucleons == 1 { 0.0 } else { d_nodes * (total_nucleons as f32).powf(0.3333) };
            let y = 1.0 - (i as f32 / (total_nucleons as f32 - 1.0).max(1.0)) * 2.0;
            let radius_at_y = (1.0 - y * y).sqrt();
            let theta = phi * i as f32;

            // 🌟 FIX: The 4th component must be the absolute target macro radius in LU
            self.rings[i as usize] = [
                cx + theta.cos() * radius_at_y * shell_radius,
                cy + y * shell_radius,
                cz + theta.sin() * radius_at_y * shell_radius,
                phys.r_macro
            ];

            let is_proton = if placed_protons < z && (placed_protons <= placed_neutrons || placed_neutrons == n) {
                placed_protons += 1; true
            } else {
                placed_neutrons += 1; false
            };

            // 🌟 FIX: Pass normalized fractions [p, q, f_macro, f_minor] to prevent arithmetic scaling explosions
            self.ring_props[i as usize] = [
                if is_proton { 3.0 } else { -3.0 },
                2.0,
                f_macro,
                f_minor
            ];
        }

        self.ring_count = total_nucleons;
        self.blend_factor = 0.0;
        self._pad1 = 0.0;
        self.r_nucleon = phys.r_macro; // Linked directly to physics macro scale
        self.u_core_lu = phys.u_core_lu;
        self.core_pos = [cx, cy, cz, 0.0];
        self.n_state = 1.0;
        self.l_state = 0.0;
        self.m_state = 0.0;

        println!("✅ Nucleus Blueprint Generated ({} stable nucleons).", self.ring_count);
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct RenderParams {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub target_orbital_amp: f32,
    pub noise_threshold: f32,
    pub wave_threshold: f32,
    pub matter_threshold: f32,
    pub alpha_scale: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct DiagUniforms {
    pub prev_cx: f32, pub prev_cy: f32, pub prev_cz: f32,
    pub prev_max_variance: f32, // Passed from 1.0 - m.knot_min_rho
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuMetrics {
    pub nan_count: f32, pub inf_count: f32,
    pub sum_ux: f32, pub sum_uy: f32, pub sum_uz: f32,
    pub sum_ux_sq: f32, pub sum_uy_sq: f32, pub sum_uz_sq: f32,
    pub total_mass: f32, pub peak_density: f32,
    pub net_mom_x: f32, pub net_mom_y: f32, pub net_mom_z: f32,
    pub total_kinetic_e: f32, pub max_speed_sq: f32,
    pub ambient_center_noise: f32,
    pub sum_x: f32, pub sum_y: f32, pub sum_z: f32,
    pub sum_x_sq: f32, pub sum_y_sq: f32, pub sum_z_sq: f32,
    pub knot_weight: f32, pub knot_volume: f32,
    pub knot_mom_z: f32, pub vacuum_mom_z: f32,
    pub gl_energy_integral: f32, pub knot_kinetic_e: f32,
    pub knot_peak_rho: f32, pub knot_min_rho: f32,
    pub inner_rho: f32, pub inner_count: f32,
    pub outer_rho: f32, pub outer_count: f32,
    pub boundary_echo: f32,
    pub orbital_wave_amplitude: f32, pub orbital_node_count: f32,
    pub max_vorticity_mag: f32, pub total_enstrophy: f32, pub total_helicity: f32,
    pub ang_mom_x: f32, pub ang_mom_y: f32, pub ang_mom_z: f32,
    pub local_core_weight: f32,
    pub net_gl_force_mag: f32, pub net_zpe_force_mag: f32,
    pub max_gl_force: f32,
    pub quantum_circulation: f32,
    pub radial_bins: [f32; 120],
    pub gravitational_flux: f32,
    pub _padding: [f32; 3], // Padded to exactly 172 floats (688 bytes) to satisfy 16-byte alignment
}

impl Default for GpuMetrics {
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct CameraUniform {
    camera_pos: [f32; 4],
    inv_view_proj: [f32; 16],
    resolution: [f32; 2],
    density_gain: f32,
    pub quality_step: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ShaderTelemetry {
    pub debug_x: f32,
    pub debug_y: f32,
    pub debug_z: f32,
    pub local_rho: f32,
    pub local_ux: f32,
    pub local_uy: f32,
    pub local_uz: f32,
    pub korteweg_fx: f32,
    pub phase_lock_fx: f32,
    pub lorentz_factor: f32,
    pub tripwire_triggered: u32,
    pub max_pi_magnitude: f32,      // Tracks peak localized vacuum shear
    pub peak_polarized_tau: f32,    // Tracks maximum local tau deviation
    pub negative_population_count: u32 // Monitors how close voxels are to mass inversion
}

impl Default for ShaderTelemetry {
    fn default() -> Self { unsafe { std::mem::zeroed() } }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct SoupUniforms {
    // Block 1 (16 bytes)
    pub width: u32, pub height: u32, pub depth: u32, pub tau: f32,
    // Block 2 (16 bytes)
    pub zpe_amplitude: f32, pub tick: u32, pub q_max_lu: f32, pub l_sq_lu: f32,
    // Block 3 (16 bytes) - NEW: Derived Fluid Dynamics Scaling
    pub alpha_lu: f32,        // Scaled Madelung Surface Tension
    pub vc_epsilon_lu: f32,   // Scaled Magnuss Phase Lock
    pub eta_p_lu: f32,        // Scaled Oldroyd-B Elasticity
    pub lambda_1_lu: f32,     // Scaled Oldroyd-B Relaxation Time
    base_speed_lu: f32, u_core_lu: f32, gamma_lu: f32, g_target: f32,
}

// Update this struct wherever it is defined
#[derive(Clone)]
pub struct EngineMetrics {
    pub mass: String,
    pub kinetic: String,
    pub momentum: String,
    pub density: String,
    pub mach: String,
    pub stability: String,
    pub correlation: String,
    pub axial: String,

    pub net_speed: f32,
    pub heading_yaw: f32,
    pub heading_pitch: f32,

    pub center_x: f32,
    pub center_y: f32,
    pub center_z: f32,

    pub vacuum_mom_z: f32,
    pub knot_mom_z: f32,
    pub boundary_echo: f32,

    pub event: String,

    pub knot_volume: f64,
    pub knot_peak_rho: f32,
    pub knot_min_rho: f32,

    pub total_enstrophy: f32,
    pub total_helicity: f32,

    pub casimir_delta: f32,
    pub real_time: f64,

    pub shape_anisotropy: f32,
    pub orbital_amplitude: f32,
    pub max_vorticity: f32,

    pub compton_freq: f32,
    pub quantum_circulation: f32,

    // --- PROBABILITY ACCUMULATION LAYER ---
    pub radial_probability: Vec<f32>,
    pub probability_volume: Option<Vec<f64>>,
    pub sample_count: u64,

    pub ang_mom_x: f32,
    pub ang_mom_y: f32,
    pub ang_mom_z: f32,

    pub ambient_center_noise: f64,
    pub total_system_energy: f32,
    pub energy_loss: f32,
    pub g_measured: f32,

    pub target_mass_error: f64,
    pub target_volume_error: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineState {
    Physics,
    WaitingForDiagnosticsMap,
    WaitingForRenderMap,
}

pub struct SoupEngine {
    pub baseline_metrics: Option<EngineMetrics>,
    device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,

    physics_pipeline: wgpu::ComputePipeline,
    zpe_pipeline: wgpu::ComputePipeline,
    extract_pipeline: wgpu::ComputePipeline,
    raymarch_pipeline: wgpu::ComputePipeline,

    // 🌟 PURIFIED: Single bind groups for in-place Bailey A-A pattern
    physics_bind_group: wgpu::BindGroup,
    zpe_bind_group: wgpu::BindGroup,
    extract_bind_group: wgpu::BindGroup,
    raymarch_bind_group: wgpu::BindGroup,
    pub metrics_bind_group: wgpu::BindGroup,

    pub physics: PhysicsState,
    physics_uniform_buffer: wgpu::Buffer,
    // pub constants: FluidConstants,

    pub blueprint_uniform_buffer: wgpu::Buffer,
    pub blueprint: BlueprintState,

    // 🌟 PURIFIED: A single read/write fluid grid buffer
    // f_grid_buffer: wgpu::Buffer,

    pub frame_counter: u32,
    pub render_skip: u32,

    pub camera_yaw: f32,
    pub camera_pitch: f32,
    camera_uniform_buffer: wgpu::Buffer,

    pub scale: f64,
    pub width: u32,
    pub height: u32,
    pub depth: u32,

    pub camera_target: glam::Vec3,
    pub camera_radius: f32,
    // pub render_params: RenderParams,
    // pub render_uniform_buffer: wgpu::Buffer,

    pub injection_ticks: u32,
    pub event_flag: Option<String>,
    pub dt: f64,

    pub latest_metrics: Option<EngineMetrics>,
    pub previous_helicity: f32,

    // 🌟 PURIFIED: Ginzburg-Landau (GBC) Vacuum Constants
    // pub coherence_length_lu: f32,
    // pub gl_alpha: f32,
    // pub phase_lock_rate: f32,

    pub accum_volume: Vec<f64>,
    pub accum_samples: u64,
    pub visual_accum_buffer: wgpu::Buffer,

    pub run_dir: String,
    pub frame_capture_buffer: std::sync::Arc<wgpu::Buffer>,
    pub png_texture: wgpu::Texture,

    pub u_field_cache: Vec<[f32; 3]>,
    pub rho_field_cache: Vec<f32>,

    pub persistent_radial_bins: [f32; 120],
    pub persistent_radial_samples: u64,

    pub diagnostics_rx: Option<std::sync::mpsc::Receiver<EngineMetrics>>,
    pub capture_frequency: u32,

    pub image_map_rx: Option<std::sync::mpsc::Receiver<()>>,
    pub pending_image_frame: u32,

    pub diag_map_rx: Option<std::sync::mpsc::Receiver<()>>,
    pub pending_diag_frame: u32,
    pub pending_render_capture: bool,

    pub engine_state: EngineState,
    pub diag_rx: Option<std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>,
    pub render_rx: Option<std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>,
    pub last_diag_frame: u32,
    pub last_render_frame: u32,

    pub physics_passes: u64,
    pub just_ran_diag: bool,
    pub just_ran_render: bool,

    pub diag_uniform_buffer: wgpu::Buffer,
    pub metrics_intermediate_buffer: wgpu::Buffer,
    pub metrics_final_buffer: wgpu::Buffer,
    pub metrics_staging_buffer: Arc<wgpu::Buffer>,

    pub map_pipeline: wgpu::ComputePipeline,
    pub reduce_pipeline: wgpu::ComputePipeline,

    pub logger: RunLogger,
    pub telemetry_buffer: wgpu::Buffer,
    pub telemetry_staging_buffer: std::sync::Arc<wgpu::Buffer>,

    pub has_injected: bool
}


// Spacetime memory scales with the local fluid resolution
// fn get_target_relaxation(dx: f64) -> f64 {
//     let planck_time = 5.39e-44;
//     // Spacetime memory is a log-scale function of the resolution relative to Planck
//     planck_time * (1.0 / dx).log10()
// }

impl SoupEngine {
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        slint_texture_view: wgpu::TextureView,
        slint_texture: wgpu::Texture,
        width: u32,
        height: u32,
        depth: u32
    ) -> Self
    {
        // ====================================================================
        // 1. FILE SYSTEM & CAPTURE BUFFERS
        // ====================================================================
        let run_dir = format!("frames/run_{}", chrono::Local::now().format("%Y%m%d_%H%M%S"));
        std::fs::create_dir_all(&run_dir).expect("Failed to create frames directory");

        let padded_bytes_per_row = 3328;
        let frame_capture_buffer = std::sync::Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Frame Capture Buffer"),
            size: (padded_bytes_per_row * 750) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        // ====================================================================
        // 2. SHADER COMPILATION
        // ====================================================================
        let physics_shader = device.create_shader_module(wgpu::include_wgsl!("../shaders/rr_lbm.wgsl"));
        let zpe_shader = device.create_shader_module(wgpu::include_wgsl!("../shaders/zpe_sower.wgsl")); // 🌟 NEW
        let extract_shader = device.create_shader_module(wgpu::include_wgsl!("../shaders/volume_texture_extraction.wgsl"));
        let raymarch_shader = device.create_shader_module(wgpu::include_wgsl!("../shaders/volume_raymarch.wgsl"));
        let metrics_shader = device.create_shader_module(wgpu::include_wgsl!("../shaders/metrics.wgsl"));

        let physics_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Physics Pipeline"), layout: None, module: &physics_shader, entry_point: Some("main"), compilation_options: Default::default(), cache: None,
        });
        let zpe_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ZPE Pipeline"), layout: None, module: &zpe_shader, entry_point: Some("main"), compilation_options: Default::default(), cache: None,
        });
        let extract_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Extract Pipeline"), layout: None, module: &extract_shader, entry_point: Some("main"), compilation_options: Default::default(), cache: None,
        });
        let raymarch_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Raymarch Pipeline"), layout: None, module: &raymarch_shader, entry_point: Some("main"), compilation_options: Default::default(), cache: None,
        });

        // ====================================================================
        // 3. PHYSICAL CONSTANTS & UNIFORMS
        // ====================================================================
        // 🌟 MASSIVE CLEANUP: All math and array inversions happen inside derive()
        let physics = PhysicsState::derive(width, height, depth, MACH_TARGET);
        println!("Mach Target: {:.6}", MACH_TARGET);
        println!("Physics: {:?}", physics);
        // println!("[INIT] Vacuum Parameters Derived. U_Core: {}", physics.u_core_lu);

        // let render_params = RenderParams {
        //     width, height, depth, target_orbital_amp: 0.0,
        //     noise_threshold: 0.0001,   // Filter out the low-level ZPE floor
        //     wave_threshold: 0.0005,    // The "transition" layer
        //     matter_threshold: 0.002,   // 🌟 CRITICAL: Raise this to ignore the ZPE fog
        //     alpha_scale: 0.8,          // Slightly dial back the total accumulation
        // };
        let total_nodes = (width * height * depth) as usize;
        let w: [f32; 39] = [
            1./12.,
            1./12., 1./12., 1./12., 1./12., 1./12., 1./12.,
            1./27., 1./27., 1./27., 1./27., 1./27., 1./27., 1./27., 1./27.,
            2./135., 2./135., 2./135., 2./135., 2./135., 2./135.,
            1./432., 1./432., 1./432., 1./432., 1./432., 1./432., 1./432., 1./432., 1./432., 1./432., 1./432., 1./432.,
            1./1620., 1./1620., 1./1620., 1./1620., 1./1620., 1./1620.
        ];

        let mut initial_f_low = vec![f16::from_f32(0.0); total_nodes * 20];
        let mut initial_f_high = vec![f16::from_f32(0.0); total_nodes * 20];

        println!("[INIT] Sowing Quantum Foam into Split f16 LBM Lattice...");

        // Zip the two slices together to initialize them concurrently
        initial_f_low.par_chunks_mut(20).zip(initial_f_high.par_chunks_mut(20))
            .enumerate().for_each(|(_idx, (node_low, node_high))| {
            let foam_rho = 1.0_f32;

            // Fill Buffer A (Vectors 0 - 19)
            for d in 0..20 {
                node_low[d] = f16::from_f32(w[d] * foam_rho);
            }

            // Fill Buffer B (Vectors 20 - 38). Slot 39 remains 0.0 padding!
            for d in 20..39 {
                node_high[d - 20] = f16::from_f32(w[d] * foam_rho);
            }
        });

        // Bind both buffers to the GPU
        let f_grid_low_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Fluid Grid Low Buffer"),
            contents: bytemuck::cast_slice(&initial_f_low),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        });

        let f_grid_high_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Fluid Grid High Buffer"),
            contents: bytemuck::cast_slice(&initial_f_high),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        });

        // ====================================================================
        // 5. GPU BUFFER ALLOCATION
        // ====================================================================
        println!("[INIT] Uploading State to GPU...");

        // let f_grid_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        //     label: Some("Fluid Grid Buffer"),
        //     contents: bytemuck::cast_slice(&initial_f),
        //     usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        // });

        let initial_env = vec![0.0f32; total_nodes];
        let env_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Environment Map Buffer"),
            contents: bytemuck::cast_slice(&initial_env),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let initial_accum = vec![0.0f32; total_nodes];
        let visual_accum_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Visual Accumulator Buffer"),
            contents: bytemuck::cast_slice(&initial_accum),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let telemetry_size = std::mem::size_of::<ShaderTelemetry>() as wgpu::BufferAddress;
        let initial_telemetry = ShaderTelemetry::default();

        let telemetry_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Telemetry Storage Buffer"),
            contents: bytemuck::cast_slice(&[initial_telemetry]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        });

        let telemetry_staging_buffer = std::sync::Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Telemetry Staging Buffer"),
            size: telemetry_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        let physics_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Physics Uniform Buffer"), contents: bytemuck::cast_slice(&[physics]), usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // 🌟 UPDATED: Bound directly to the size of BlueprintState
        let blueprint_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Blueprint Uniform Buffer"), size: std::mem::size_of::<BlueprintState>() as wgpu::BufferAddress, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
        });

        // let render_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        //     label: Some("Render Uniform Buffer"), contents: bytemuck::cast_slice(&[physics_uniform_buffer]), usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        // });

        let camera_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Camera Uniform Buffer"), size: std::mem::size_of::<CameraUniform>() as u64, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
        });

        // ====================================================================
        // 6. RAYMARCHING TEXTURE & METRICS PIPELINES
        // ====================================================================
        let volume_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Volume Texture"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: depth },
            mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        let volume_view = volume_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let volume_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Trilinear Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge, address_mode_v: wgpu::AddressMode::ClampToEdge, address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear, min_filter: wgpu::FilterMode::Linear, mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });

        let zpe_field_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ZPE Field Buffer"),
            size: (total_nodes * std::mem::size_of::<[f16; 4]>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let wgs_x = (width + 7) / 8;
        let wgs_y = (height + 7) / 8;
        let wgs_z = (depth + 3) / 4;
        let total_groups = wgs_x * wgs_y * wgs_z;

        let struct_size = std::mem::size_of::<GpuMetrics>() as u64;

        let diag_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor { label: Some("Diag Uniforms"), size: 16, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let metrics_intermediate_buffer = device.create_buffer(&wgpu::BufferDescriptor { label: Some("Metrics Intermediate"), size: (total_groups as u64) * struct_size, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false });
        let metrics_final_buffer = device.create_buffer(&wgpu::BufferDescriptor { label: Some("Metrics Final"), size: struct_size, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
        let metrics_staging_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor { label: Some("Metrics Staging"), size: struct_size, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false }));

        let metrics_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Metrics Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });

        let metrics_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Metrics Pipeline Layout"), bind_group_layouts: &[Some(&metrics_layout)], immediate_size: 0,
        });

        let map_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Map Pipeline"), layout: Some(&metrics_pipeline_layout), module: &metrics_shader, entry_point: Some("map_blocks"), compilation_options: Default::default(), cache: None,
        });

        let reduce_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Reduce Pipeline"), layout: Some(&metrics_pipeline_layout), module: &metrics_shader, entry_point: Some("reduce_global"), compilation_options: Default::default(), cache: None,
        });

        let macro_state_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Macro State Buffer"),
            size: (total_nodes * std::mem::size_of::<[f32; 4]>()) as u64, // 🌟 FIXED
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ====================================================================
        // 7. BIND GROUPS (Single Instantiation)
        // ====================================================================
        // ====================================================================
        // LIST 1: Master Physics Bind Group (7 Entries)
        // ====================================================================
        let physics_layout = physics_pipeline.get_bind_group_layout(0);
        let physics_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Physics Bind Group"),
            layout: &physics_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: physics_uniform_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: f_grid_low_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: f_grid_high_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: blueprint_uniform_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: macro_state_buffer.as_entire_binding() },
                // wgpu::BindGroupEntry { binding: 5, resource: telemetry_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: zpe_field_buffer.as_entire_binding() },
            ],
        });

        // ====================================================================
        // LIST 2: Lean ZPE Sower Bind Group (Only the 2 entries it actually uses!)
        // ====================================================================
        let zpe_layout = zpe_pipeline.get_bind_group_layout(0);
        let zpe_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ZPE Sower Bind Group"),
            layout: &zpe_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: physics_uniform_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: zpe_field_buffer.as_entire_binding() },
            ],
        });

        let extract_layout = extract_pipeline.get_bind_group_layout(0);
        let extract_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Extract Bind Group"), layout: &extract_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: physics_uniform_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: env_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: macro_state_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&volume_view) },
                wgpu::BindGroupEntry { binding: 4, resource: visual_accum_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: diag_uniform_buffer.as_entire_binding() },
            ],
        });

        let raymarch_layout = raymarch_pipeline.get_bind_group_layout(0);
        let raymarch_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Raymarch Bind Group"), layout: &raymarch_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: camera_uniform_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&volume_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&volume_sampler) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&slint_texture_view) },
            ],
        });

        let metrics_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Metrics Bind Group"), layout: &metrics_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: physics_uniform_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: macro_state_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: metrics_intermediate_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: metrics_final_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: diag_uniform_buffer.as_entire_binding() },
            ],
        });

        // ====================================================================
        // 8. FINAL ENGINE ASSEMBLY
        // ====================================================================
        // 🌟 UPDATED: Generates an empty shell, ready for injection via UI
        let blueprint = BlueprintState::empty();

        let mut engine = Self {
            baseline_metrics: None,
            device, queue, physics_pipeline,
            zpe_pipeline,
            extract_pipeline, raymarch_pipeline,

            physics_bind_group, zpe_bind_group, extract_bind_group, raymarch_bind_group, metrics_bind_group,

            // 🌟 UPDATED: Replaced all the loose params with the single unified structs
            physics, physics_uniform_buffer,
            blueprint, blueprint_uniform_buffer,

            // f_grid_buffer,

            frame_counter: 0, render_skip: 250,
            camera_yaw: -0.78f32, camera_pitch: 0.5f32, camera_uniform_buffer,
            scale: DX, width, height, depth,

            camera_target: glam::Vec3::ZERO, camera_radius: 3.0,
            // render_params, render_uniform_buffer,

            injection_ticks: 0, event_flag: None,

            // Calculate dt locally since we excluded it from the GPU buffer to save bytes
            dt: (DX * C_EFF) / CS_LATTICE,

            latest_metrics: None, previous_helicity: 0.0,

            accum_volume: vec![0.0_f64; total_nodes], accum_samples: 0, visual_accum_buffer,
            run_dir, frame_capture_buffer, png_texture: slint_texture,

            u_field_cache: vec![[0.0_f32; 3]; total_nodes], rho_field_cache: vec![0.0_f32; total_nodes],
            persistent_radial_bins: [0.0; 120], persistent_radial_samples: 0,

            diagnostics_rx: None, capture_frequency: 250,
            image_map_rx: None, pending_image_frame: 0,
            diag_map_rx: None, pending_diag_frame: 0, pending_render_capture: false,

            engine_state: EngineState::Physics,
            diag_rx: None, render_rx: None,
            last_diag_frame: u32::MAX, last_render_frame: u32::MAX,

            physics_passes: 0, just_ran_diag: false, just_ran_render: false,

            diag_uniform_buffer, metrics_intermediate_buffer, metrics_final_buffer, metrics_staging_buffer,
            map_pipeline, reduce_pipeline,
            logger: RunLogger::new(), telemetry_buffer, telemetry_staging_buffer,
            has_injected: false
        };

        engine.update_camera_matrix(1.0);
        engine.logger.start_new_run(engine.width, engine.height, engine.depth, &engine.physics);

        engine
    }

    pub fn pan_camera(&mut self, dx: f32, dy: f32) {
        let forward = glam::Vec3::new(
            -self.camera_pitch.cos() * self.camera_yaw.sin(),
            -self.camera_pitch.sin(),
            -self.camera_pitch.cos() * self.camera_yaw.cos(),
        ).normalize();

        let right = forward.cross(glam::Vec3::Y).normalize();
        let up = right.cross(forward).normalize();

        // Scale pan speed based on distance (radius) to maintain a consistent "feel"
        let pan_speed = 0.001 * self.camera_radius;

        // Slint's Y axis is inverted relative to standard 3D space, so we subtract dy
        self.camera_target -= right * dx * pan_speed;
        self.camera_target += up * dy * pan_speed;

        // Notice: We don't write to the GPU here anymore!
    }

    pub fn update_camera_matrix(&mut self, quality_step: f32) {
        // 1. Calculate where the camera IS in space relative to the target
        let cam_x = self.camera_target.x + self.camera_radius * self.camera_pitch.cos() * self.camera_yaw.sin();
        let cam_y = self.camera_target.y + self.camera_radius * self.camera_pitch.sin();
        let cam_z = self.camera_target.z + self.camera_radius * self.camera_pitch.cos() * self.camera_yaw.cos();
        let camera_pos = glam::Vec3::new(cam_x, cam_y, cam_z);

        // 2. Look AT the target
        let view = glam::Mat4::look_at_rh(camera_pos, self.camera_target, glam::Vec3::Y);
        let proj = glam::Mat4::perspective_rh(45.0_f32.to_radians(), 800.0 / 750.0, 0.1, 100.0);
        let inv_view_proj = (proj * view).inverse();

        // 3. Assemble the uniform with the dynamically requested quality
        let uniform = CameraUniform {
            camera_pos: [camera_pos.x, camera_pos.y, camera_pos.z, 0.0],
            inv_view_proj: inv_view_proj.to_cols_array(),
            resolution: [800.0, 750.0],
            density_gain: 1.0,
            quality_step, // 🌟 Dynamically applied
        };

        self.queue.write_buffer(&self.camera_uniform_buffer, 0, bytemuck::cast_slice(&[uniform]));
    }

    pub fn execute_fast_raymarch_pass(&mut self, quality_multiplier: f32) {
        // 1. Update the GPU's camera matrix with the requested speed/quality
        self.update_camera_matrix(quality_multiplier);

        // 2. Dispatch ONLY the raymarch pipeline (bypass the 16MB texture extraction)
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Fast Raymarch") });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            cpass.set_pipeline(&self.raymarch_pipeline);
            cpass.set_bind_group(0, &self.raymarch_bind_group, &[]);

            // Dispatch across the 800x750 window resolution
            cpass.dispatch_workgroups((800 + 15) / 16, (750 + 15) / 16, 1);
        }

        self.queue.submit(Some(encoder.finish()));
    }

    pub fn step(&mut self) {
        if self.diagnostics_rx.is_some() {
            // Yielding while waiting for UI to drain diagnostics
            return;
        }

        let passes = self.physics_passes;

        // =====================================================================
        // 1. DIAGNOSTICS MILESTONE
        // =====================================================================
        if passes > 0 && passes % 100 == 0 && !self.just_ran_diag {
            println!("[SIM: HEARTBEAT] ⏳ Waiting for GPU to complete Pass {} for Diagnostics...", passes);
            let _ = self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });

            self.execute_diagnostic_pass();
            self.just_ran_diag = true;
            return;
        }

        // =====================================================================
        // 2. RENDER MILESTONE
        // =====================================================================
        if passes > 0 && passes % 250 == 1 && !self.just_ran_render
        {
            println!("[SIM: HEARTBEAT] ⏳ Waiting for GPU to complete Pass {} for Rendering...", passes);
            let _ = self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });

            // 🌟 FIX: Only save to disk every 1000 passes, otherwise just render to UI
            let save_to_disk = passes % 250 == 1;
            self.execute_render_pass(save_to_disk);
            self.just_ran_render = true;
            return;
        }

        // =====================================================================
        // 3. STANDARD PHYSICS DISPATCH
        // =====================================================================
        self.just_ran_diag = false;
        self.just_ran_render = false;

        // =====================================================================
        // 3. ADIABATIC INJECTION PROTOCOL (20,000 TICK TRIPLE-PHASE MANIFOLD)
        // =====================================================================
        if self.injection_ticks > 0
        {
            let total_phase = 1200.0;
            let current = self.injection_ticks as f32; // Counts DOWN from 1200 to 0
            let elapsed = total_phase - current;       // Counts UP from 0 to 1200

            // 1. INJECTION MOTOR TIMELINE (0 to 600 Ticks)
            // 🌟 FIXED: Scaled peak target to 1.0 to ensure a total mathematical hard-lock
            let blend_factor = if elapsed < 150.0 {
                (elapsed / 150.0).powi(2) * 1.00
            } else if elapsed < 450.0 {
                1.00 // 🌟 Full throttle containment matrix
            } else if elapsed < 600.0 {
                (1.0 - ((elapsed - 450.0) / 150.0)) * 1.00
            } else {
                0.00 // Motor completely cold for the remaining free flight
            };

            // 2. PENNING TRAP CONTAINMENT TIMELINE (0 to 1200 Ticks)
            let trap_factor = if elapsed < 1000.0 {
                // Holds at 1.0 target strength throughout injection AND the extra 400 ticks
                1.00
            } else if elapsed < 1200.0 {
                // Smooth 200-tick linear ease-out decay to zero
                (1.0 - ((elapsed - 1000.0) / 200.0)) * 1.00
            } else {
                0.00
            };

            // Assign directly to your distinct uniform buffers
            self.blueprint.blend_factor = blend_factor; // Passed to atom_params
            self.physics.trap_blend = trap_factor as f32; // Passed to global params uniform

            // Update GPU buffers
            self.queue.write_buffer(&self.blueprint_uniform_buffer, 0, bytemuck::cast_slice(&[self.blueprint]));
            self.queue.write_buffer(&self.physics_uniform_buffer, 0, bytemuck::cast_slice(&[self.physics]));

            self.injection_ticks -= 1;

            if self.injection_ticks == 0
            {
                self.event_flag = Some("Penning Trap Released".to_string());
                println!("✅ Extended Containment Hold Expired. Topological vacuum released to complete free flight.");
                // self.physics.trap_blend = 0.0;
                // self.queue.write_buffer(&self.physics_uniform_buffer, 0, bytemuck::cast_slice(&[self.physics]));

            }

            self.blueprint.blend_factor = blend_factor;
            self.blueprint.n_state = 1.0;
            self.blueprint.l_state = 0.0;
            self.blueprint.m_state = 0.0;
            self.blueprint.u_core_lu = self.physics.u_core_lu;

            // self.queue.write_buffer(&self.blueprint_uniform_buffer, 0, bytemuck::cast_slice(&[self.blueprint]));
            // self.injection_ticks -= 1;

            if self.injection_ticks == 600 {
                self.event_flag = Some("Proton Injection Complete".to_string());
                println!("✅ Adiabatic Decoupling Complete. Core successfully shifted to native thermodynamic maintenance mode.");
                // self.blueprint.blend_factor = 0.0;
                // self.queue.write_buffer(&self.blueprint_uniform_buffer, 0, bytemuck::cast_slice(&[self.blueprint]));
            }
        }

        // --- Compute Dispatch ---
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        self.physics.tick += 1;
        self.queue.write_buffer(&self.physics_uniform_buffer, 0, bytemuck::cast_slice(&[self.physics]));

        // 🌟 1. PRE-PASS: Generate the Volumetric ZPE Field
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("ZPE Sower Pass"), timestamp_writes: None });
            cpass.set_pipeline(&self.zpe_pipeline);
            // Reuses the exact same physics bind group
            cpass.set_bind_group(0, &self.zpe_bind_group, &[]);
            cpass.dispatch_workgroups((self.width + 7) / 8, (self.height + 7) / 8, (self.depth + 3) / 4);
        }

        // 🌟 2. MAIN PASS: Execute LBM Fluid Physics
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("LBM Physics Pass"), timestamp_writes: None });
            cpass.set_pipeline(&self.physics_pipeline);
            cpass.set_bind_group(0, &self.physics_bind_group, &[]);
            cpass.dispatch_workgroups((self.width + 7) / 8, (self.height + 7) / 8, (self.depth + 3) / 4);
        }

        self.queue.submit(Some(encoder.finish()));

        // Print a tiny heartbeat marker every pass so you know it's alive
        print!(".");
        use std::io::Write;
        let _ = std::io::stdout().flush();

        self.physics_passes += 1;
        self.frame_counter += 1;
    }

    pub fn execute_render_pass(&mut self, save_to_disk: bool)
    {
        // save_to_disk = true; // Temporary override for testing
        println!("[SIM: MILESTONE] Render Pass at {} physics passes (Save to disk: {})", self.physics_passes, save_to_disk);

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        // 1. EXTRACT PASS: Generate the 3D Texture from the unified f_grid
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            cpass.set_pipeline(&self.extract_pipeline);
            // 🌟 UNIFIED: Only one bind group needed for Bailey A-A pattern
            cpass.set_bind_group(0, &self.extract_bind_group, &[]);
            cpass.dispatch_workgroups((self.width + 7) / 8, (self.height + 7) / 8, (self.depth + 3) / 4);
        }

        // 2. RAYMARCH PASS: Push the 3D Volume to the 2D PNG Texture
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            cpass.set_pipeline(&self.raymarch_pipeline);
            cpass.set_bind_group(0, &self.raymarch_bind_group, &[]);
            // Uses the fixed window resolution for the dispatch
            cpass.dispatch_workgroups((800 + 15) / 16, (750 + 15) / 16, 1);
        }

        if save_to_disk {
            let padded_bytes_per_row = 3328;

            // 3. COPY: VRAM Texture -> RAM Buffer
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo { texture: &self.png_texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                wgpu::TexelCopyBufferInfo { buffer: &self.frame_capture_buffer, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(padded_bytes_per_row), rows_per_image: Some(750) } },
                wgpu::Extent3d { width: 800, height: 750, depth_or_array_layers: 1 },
            );
            self.queue.submit(Some(encoder.finish()));

            // 4. MAP: Wait for GPU to finish copying
            let (tx, rx) = std::sync::mpsc::channel();
            self.frame_capture_buffer.slice(..).map_async(wgpu::MapMode::Read, move |_| { tx.send(()).unwrap(); });
            let _ = self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
            rx.recv().unwrap();

            // 5. UNPAD: Remove wgpu 256-byte row alignment padding
            let mut unpadded_data = Vec::with_capacity((800 * 750 * 4) as usize);
            {
                let padded_data = self.frame_capture_buffer.slice(..).get_mapped_range();
                for row in 0..750 {
                    let start = (row * padded_bytes_per_row) as usize;
                    let end = start + (800 * 4) as usize;
                    unpadded_data.extend_from_slice(&padded_data[start..end]);
                }
            }
            self.frame_capture_buffer.unmap();

            // 6. SAVE: Push the PNG encode to a background OS thread
            let run_dir = self.run_dir.clone();
            let passes = self.physics_passes;
            std::thread::spawn(move || {
                let filename = format!("{}/frame_{:05}.png", run_dir, passes);
                image::save_buffer(&filename, &unpadded_data, 800, 750, image::ColorType::Rgba8).expect("Failed to save PNG");
                println!("[RENDER: SUCCESS] Saved physics state {} to disk.", passes);
            });
        } else {
            // Just submit the render so the Slint UI gets the updated texture instantly!
            self.queue.submit(Some(encoder.finish()));
        }
    }

    fn execute_diagnostic_pass(&mut self) {
        println!("[SIM: MILESTONE] Map-Reducing Diagnostics on GPU...");

        let (prev_cx, prev_cy, prev_cz) = if let Some(ref m) = self.latest_metrics {
            (m.center_x, m.center_y, m.center_z)
        } else {
            (self.width as f32 / 2.0, self.height as f32 / 2.0, self.depth as f32 / 2.0)
        };

        let prev_max_variance = if let Some(ref m) = self.latest_metrics {
            1.0 - (m.knot_min_rho as f32)
        } else {
            0.0001 // Fallback for tick 0
        };
        let diag_params = DiagUniforms { prev_cx, prev_cy, prev_cz, prev_max_variance };
        self.queue.write_buffer(&self.diag_uniform_buffer, 0, bytemuck::cast_slice(&[diag_params]));

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        // 2. Map Pass (Parallel Chunking)
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            cpass.set_pipeline(&self.map_pipeline);
            cpass.set_bind_group(0, &self.metrics_bind_group, &[]);
            cpass.dispatch_workgroups((self.width + 7) / 8, (self.height + 7) / 8, (self.depth + 3) / 4);
        }

        // 3. Reduce Pass (Global Collapse)
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            cpass.set_pipeline(&self.reduce_pipeline);
            cpass.set_bind_group(0, &self.metrics_bind_group, &[]);
            cpass.dispatch_workgroups(1, 1, 1);
        }

        // 4. Copy the result to the CPU staging buffer
        let struct_size = std::mem::size_of::<GpuMetrics>() as u64;
        encoder.copy_buffer_to_buffer(&self.metrics_final_buffer, 0, &self.metrics_staging_buffer, 0, struct_size);
        self.queue.submit(Some(encoder.finish()));

        // 5. Synchronous GPU Wait (100% safe on this background thread)
        let (tx, rx) = std::sync::mpsc::channel();
        self.metrics_staging_buffer.slice(..).map_async(wgpu::MapMode::Read, move |res| { tx.send(res).unwrap(); });
        let _ = self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        let _ = rx.recv().unwrap();

        // 6. Read the GPU reduction
        let m: GpuMetrics = {
            let mapped_data = self.metrics_staging_buffer.slice(..).get_mapped_range();
            *bytemuck::from_bytes(&mapped_data[..])
        };
        self.metrics_staging_buffer.unmap();

        // =================================================================
        // 7. FORMAT METRICS FOR UI & CSV
        // =================================================================
        let n_f32 = (self.width * self.height * self.depth) as f32;
        let mean_ux = m.sum_ux / n_f32; let mean_uy = m.sum_uy / n_f32; let mean_uz = m.sum_uz / n_f32;
        let var_x = (m.sum_ux_sq / n_f32) - (mean_ux * mean_ux);
        let var_y = (m.sum_uy_sq / n_f32) - (mean_uy * mean_uy);
        let var_z = (m.sum_uz_sq / n_f32) - (mean_uz * mean_uz);
        let avg_corr = (var_x + var_y + var_z) / 3.0;

        // 🌟 THE NEW SPHERICAL KINEMATIC METRICS
        let net_vx = m.net_mom_x / m.total_mass;
        let net_vy = m.net_mom_y / m.total_mass;
        let net_vz = m.net_mom_z / m.total_mass;

        // The true global drift speed in Lattice Units
        let net_speed = (net_vx.powi(2) + net_vy.powi(2) + net_vz.powi(2)).sqrt();

        // Spherical Heading (Azimuth and Elevation)
        let heading_yaw = net_vz.atan2(net_vx).to_degrees();
        let heading_pitch = net_vy.atan2((net_vx.powi(2) + net_vz.powi(2)).sqrt()).to_degrees();

        let inner_pressure = if m.inner_count > 0.0 { m.inner_rho / m.inner_count } else { 0.0 };
        let outer_pressure = if m.outer_count > 0.0 { m.outer_rho / m.outer_count } else { 0.0 };
        let casimir_delta = outer_pressure - inner_pressure;

        let mut cx = 0.0; let mut cy = 0.0; let mut cz = 0.0;
        let mut shape_anisotropy = 0.0;

        if m.knot_weight > 0.0 {
            cx = m.sum_x / m.knot_weight; cy = m.sum_y / m.knot_weight; cz = m.sum_z / m.knot_weight;
            let spatial_var_x = (m.sum_x_sq / m.knot_weight) - (cx * cx);
            let spatial_var_y = (m.sum_y_sq / m.knot_weight) - (cy * cy);
            let spatial_var_z = (m.sum_z_sq / m.knot_weight) - (cz * cz);
            let macro_spread = (spatial_var_x + spatial_var_y) / 2.0;
            shape_anisotropy = if macro_spread > 0.0 { spatial_var_z / macro_spread } else { 0.0 };
        }

        let avg_orbital_amplitude = if m.orbital_node_count > 0.0 { m.orbital_wave_amplitude / m.orbital_node_count } else { 0.0 };

        let max_mach = m.max_speed_sq.sqrt() / 0.81649658092; // D3Q39 c_s limit

        // ===========================================================================
        // 🌟 VOLUMETRIC COARSE-GRAINING CALIBRATION
        // ===========================================================================
        // Stored potential energy must be positive
        let madelung_e = 0.5 * (self.physics.shan_chen_g.abs() as f64) * (m.gl_energy_integral as f64);
        let total_energy_lu = (m.knot_kinetic_e as f64) + madelung_e;

        // Calculate the dimensionless volume of your coherence sphere
        let coherence_vol_lu = (4.0 / 3.0) * std::f64::consts::PI * (self.physics.coherence_length_lu as f64).powi(3);

        // Normalize the raw lattice accumulation
        let normalized_energy_lu = total_energy_lu / coherence_vol_lu;

        // 🌟 THE FIX: Map to relativistic energy using the squared lattice speed of light
        let c_lu_sq = CS_LATTICE.powi(2);
        let energy_conversion_factor = (HBAR * C) / (DX * c_lu_sq);

        let total_energy_joules = normalized_energy_lu * energy_conversion_factor;

        // E = mc^2 -> m = E / c^2
        let mass_phys_kg = total_energy_joules / (C * C);

        let target_mass_kg = 1.674927498e-27_f64;
        let mass_error_percent = ((mass_phys_kg - target_mass_kg) / target_mass_kg) * 100.0;
        let compton_freq = (mass_phys_kg * (C * C) / HBAR) as f32;

        // 🌟 EXACT TOPOLOGICAL TREFOIL VOLUME
        // We remove the magic numbers and pull the absolute physical scales from the engine.
        // let r_minor = self.physics.r_minor as f64;
        // let r_macro = self.physics.r_macro as f64;

        // Centerline length of a (p=3, q=2) torus knot
        // let p = 3.0_f64;
        // let q = 2.0_f64;
        // let knot_length = 2.0 * std::f64::consts::PI * (p.powi(2) * r_macro.powi(2) + q.powi(2) * r_minor.powi(2)).sqrt();
        // let ideal_v_vortex_lu = std::f64::consts::PI * r_minor.powi(2) * knot_length;

        // The error now correctly compares the fluid's measured knot volume against the mathematical ideal trefoil volume
        // let volume_error_percent = ((m.knot_volume as f64 - ideal_v_vortex_lu) / ideal_v_vortex_lu) * 100.0;

        // 🌟 EMPIRICAL PROTON VOLUME COMPARISON
        // A standard physical proton is modeled as a sphere with radius = coherence_length
        let r_p_lu = self.physics.coherence_length_lu as f64;
        let empirical_proton_vol_lu = (4.0 / 3.0) * std::f64::consts::PI * r_p_lu.powi(3);
        let empirical_volume_error = ((m.knot_volume as f64 - empirical_proton_vol_lu) / empirical_proton_vol_lu) * 100.0;

        let current_event = self.event_flag.take().unwrap_or_else(|| "None".to_string());

        // Convert the 120 radial bins from the GPU into your rust struct
        let mut radial_f64 = [0.0f64; 120];
        for i in 0..120 { radial_f64[i] = m.radial_bins[i] as f64; }

        let final_metrics = EngineMetrics {
            mass: format!("{:.2}", m.total_mass),
            kinetic: format!("{:.2}", m.total_kinetic_e),
            momentum: format!("Yaw: {:.1}° | Ptch: {:.1}°", heading_yaw, heading_pitch),
            density: format!("{:.4}", m.peak_density),
            mach: format!("{:.4}", max_mach),
            stability: format!("{},{}", m.nan_count, m.inf_count),
            correlation: format!("{:.4}", avg_corr),
            axial: format!("{:.6} LU/t", net_speed),

            // 🌟 ASSIGN THE RAW FIELDS HERE
            net_speed,
            heading_yaw,
            heading_pitch,

            center_x: cx, center_y: cy, center_z: cz,
            event: current_event,

            real_time: (self.physics_passes as f64) * self.dt,
            knot_volume: m.knot_volume as f64,
            vacuum_mom_z: m.vacuum_mom_z,
            knot_mom_z: m.knot_mom_z,
            boundary_echo: m.boundary_echo,
            knot_peak_rho: m.knot_peak_rho,
            total_enstrophy: m.total_enstrophy,
            total_helicity: m.total_helicity,
            casimir_delta: casimir_delta,
            knot_min_rho: m.knot_min_rho,
            shape_anisotropy: shape_anisotropy,

            orbital_amplitude: avg_orbital_amplitude,
            max_vorticity: m.max_vorticity_mag,
            compton_freq: compton_freq,
            quantum_circulation: m.quantum_circulation,
            radial_probability: m.radial_bins.to_vec(),

            probability_volume: None, sample_count: 1,
            ang_mom_x: m.ang_mom_x, ang_mom_y: m.ang_mom_y, ang_mom_z: m.ang_mom_z,
            ambient_center_noise: m.ambient_center_noise as f64,

            total_system_energy: total_energy_lu as f32,
            energy_loss: 0.0,
            g_measured: -m.gravitational_flux / (4.0 * std::f32::consts::PI * m.knot_weight),

            target_mass_error: mass_error_percent,
            target_volume_error: empirical_volume_error,
        };

        // =================================================================
        // 8. LOG AND DISPATCH TO UI
        // =================================================================
        self.logger.log_tick(self.physics_passes, &final_metrics);

        self.latest_metrics = Some(final_metrics.clone());
        let (tx_metrics, rx_metrics) = std::sync::mpsc::channel();
        tx_metrics.send(final_metrics).unwrap();
        self.diagnostics_rx = Some(rx_metrics);

        println!("[DIAG: COMPLETE] Physics will stand by for CPU.");
    }

    // 3. Fast-check for background completion during the main loop
    pub fn poll_diagnostics(&mut self) -> Option<EngineMetrics> {
        if let Some(rx) = &self.diagnostics_rx {
            if let Ok(mut raw) = rx.try_recv() {
                self.diagnostics_rx = None;

                // Incorporate background radial bin calculations locally
                if self.frame_counter > 1500 {
                    for i in 0..120 { self.persistent_radial_bins[i] += raw.radial_probability[i]; }
                    self.persistent_radial_samples += 1;
                }
                raw.radial_probability = self.persistent_radial_bins.to_vec();
                raw.sample_count = self.persistent_radial_samples.max(1);

                // Baseline and Delta application
                if self.baseline_metrics.is_none() && self.frame_counter >= 100 {
                    self.baseline_metrics = Some(raw.clone());
                    return Some(raw);
                }

                if let Some(baseline) = &self.baseline_metrics {
                    let previous_energy = if let Some(ref m) = self.latest_metrics { m.total_system_energy } else { raw.total_system_energy };
                    raw.energy_loss = raw.total_system_energy - previous_energy;

                    raw.vacuum_mom_z -= baseline.vacuum_mom_z;
                    raw.knot_mom_z -= baseline.knot_mom_z;
                    raw.boundary_echo -= baseline.boundary_echo;
                }

                self.latest_metrics = Some(raw.clone());
                return Some(raw);
            }
        }
        None
    }

    pub fn read_telemetry_probe(&mut self) -> ShaderTelemetry {
        let struct_size = std::mem::size_of::<ShaderTelemetry>() as wgpu::BufferAddress;

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Telemetry Forensic Copy")
        });

        // Copy high-speed storage buffer data down to host-mappable staging memory
        encoder.copy_buffer_to_buffer(&self.telemetry_buffer, 0, &self.telemetry_staging_buffer, 0, struct_size);
        self.queue.submit(Some(encoder.finish()));

        // Async channel link to hook the GPU complete signal
        let (tx, rx) = std::sync::mpsc::channel();
        self.telemetry_staging_buffer.slice(..).map_async(wgpu::MapMode::Read, move |_| {
            tx.send(()).unwrap();
        });

        // Sync poll execution context matching your established code syntax
        let _ = self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        rx.recv().unwrap();

        // Extract raw byte array map into a typed stack instance
        let probe_data: ShaderTelemetry = {
            let mapped_range = self.telemetry_staging_buffer.slice(..).get_mapped_range();
            *bytemuck::from_bytes(&mapped_range[..])
        };
        self.telemetry_staging_buffer.unmap();

        // Reset the trigger and telemetry properties directly inside VRAM for the next pass
        if probe_data.tripwire_triggered == 1 {
            self.queue.write_buffer(&self.telemetry_buffer, 0, bytemuck::cast_slice(&[ShaderTelemetry::default()]));
        }

        probe_data
    }

    // Dumps the exact 2.6GB fluid grid state to a raw binary file.
    // pub fn save_checkpoint(&self, filename: &str) {
    //     println!("💾 Saving D3Q39 checkpoint to {}...", filename);
    //
    //     // 256 * 256 * 256 * 39 directions * 4 bytes (f32) = 2,617,245,696 bytes
    //     let buffer_size = (self.width * self.height * self.depth * 40 * 4) as wgpu::BufferAddress;
    //
    //     // Allocate a temporary staging buffer
    //     let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
    //         label: Some("Checkpoint Staging Buffer"),
    //         size: buffer_size,
    //         usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
    //         mapped_at_creation: false,
    //     });
    //
    //     // Command the GPU to copy the fluid grid to the CPU-readable staging buffer
    //     let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    //     encoder.copy_buffer_to_buffer(&self.f_grid_buffer, 0, &staging_buffer, 0, buffer_size);
    //     self.queue.submit(Some(encoder.finish()));
    //
    //     // Synchronous await for the GPU to finish the 2.6GB transfer
    //     let (tx, rx) = std::sync::mpsc::channel();
    //     staging_buffer.slice(..).map_async(wgpu::MapMode::Read, move |result| {
    //         tx.send(result).unwrap();
    //     });
    //     let _ = self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
    //     rx.recv().unwrap().unwrap();
    //
    //     // Stream the raw bytes directly to disk
    //     let data = staging_buffer.slice(..).get_mapped_range();
    //     std::fs::write(filename, &*data).expect("Failed to write checkpoint file to disk");
    //
    //     drop(data);
    //     staging_buffer.unmap();
    //
    //     println!("✅ Checkpoint saved successfully ({} bytes).", buffer_size);
    // }
    //
    // /// Injects a previously saved binary checkpoint directly into the live VRAM fluid grid.
    // pub fn load_checkpoint(&mut self, filename: &str) {
    //     println!("📂 Loading D3Q39 checkpoint from {}...", filename);
    //
    //     let expected_size = (self.width * self.height * self.depth * 39 * 4) as usize;
    //
    //     let data = std::fs::read(filename).expect("Failed to read checkpoint file from disk");
    //
    //     if data.len() != expected_size {
    //         panic!(
    //             "🛑 Checkpoint file size mismatch! Expected {} bytes, but the file is {} bytes. \
    //             Are you trying to load a D3Q27 checkpoint into a D3Q39 grid?",
    //             expected_size, data.len()
    //         );
    //     }
    //
    //     // Blast the byte array straight into VRAM
    //     self.queue.write_buffer(&self.f_grid_buffer, 0, &data);
    //
    //     println!("✅ Checkpoint loaded and VRAM grid overwritten successfully.");
    // }
}