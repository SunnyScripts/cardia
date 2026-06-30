//Ryan Berg 5/16/26
//Objective Physics Sandbox for discrete super fluid space based on quadratic drag
//n=2 ;)
//Derived from the fermi telescope data that converges with cutoff data of the Ultra High Energy Particles like Oh-My-God from U of U
//uses a Gross-Pitaevskii solver
//the manifold natively curves spacetime around the lattice via the Maxwell-Jüttner distribution

use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    event::{ElementState, MouseScrollDelta, WindowEvent}, //KeyEvent
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{Key, NamedKey},
    window::{Window, WindowId},
};
use soup::{SoupEngine};

struct SoupApp {
    engine: SoupEngine,
    is_left_clicked: bool,
    last_mouse_pos: winit::dpi::PhysicalPosition<f64>,
    sim_paused: bool,
    window: Option<Arc<Window>>,
    zero_volume_checks: u32,
}

impl ApplicationHandler for SoupApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let window_attributes = Window::default_attributes()
                .with_title("Soup Quantum Fluid Engine")
                .with_inner_size(winit::dpi::PhysicalSize::new(800, 750));

            self.window = Some(Arc::new(event_loop.create_window(window_attributes).unwrap()));
            println!("▶ Simulation Running Headless with Window Hooks. Press Space to Pause, 'I' to inject.");
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }

            WindowEvent::KeyboardInput { event: key_event, .. } => {
                if key_event.state == ElementState::Pressed {
                    match key_event.logical_key {
                        Key::Named(NamedKey::Space) => {
                            self.sim_paused = !self.sim_paused;
                            println!("{}", if self.sim_paused { "⏸ Simulation Paused." } else { "▶ Simulation Resumed." });
                        }
                        Key::Character(ref ch) if ch == "i" || ch == "I" => {
                            println!("⚡ Manual Injection Triggered via Hotkey.");
                            let cx = self.engine.width as f32 / 2.0;
                            let cy = self.engine.height as f32 / 2.0;
                            let cz = self.engine.depth as f32 / 2.0;
                            let phys = self.engine.physics;

                            self.engine.blueprint.inject_nucleus(cx, cy, cz, 1, 0, &phys);
                            self.engine.queue.write_buffer(&self.engine.blueprint_uniform_buffer, 0, bytemuck::cast_slice(&[self.engine.blueprint]));
                            self.engine.injection_ticks = 20000;
                            self.engine.event_flag = Some("HOTKEY_INJECT_Z1_N0".to_string());
                        }
                        _ => {}
                    }
                }
            }

            WindowEvent::MouseInput { state, button, .. } => {
                if button == winit::event::MouseButton::Left {
                    self.is_left_clicked = state == ElementState::Pressed;
                }
            }

            WindowEvent::CursorMoved { position, .. } => {
                let dx = position.x - self.last_mouse_pos.x;
                let dy = position.y - self.last_mouse_pos.y;
                self.last_mouse_pos = position;

                if self.is_left_clicked {
                    self.engine.camera_yaw -= (dx as f32) * 0.002;
                    self.engine.camera_pitch += (dy as f32) * 0.002;
                    self.engine.camera_pitch = self.engine.camera_pitch.clamp(-1.5, 1.5);
                    self.engine.execute_fast_raymarch_pass(3.0);
                }
            }

            WindowEvent::MouseWheel { delta, .. } => {
                let scroll_amount = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(pos) => pos.y as f32 * 0.1,
                };
                self.engine.camera_radius = (self.engine.camera_radius - scroll_amount * 0.1).clamp(0.5, 10.0);
                self.engine.execute_fast_raymarch_pass(3.0);
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if !self.sim_paused {
            self.engine.step();

            // Pull the underlying GPU variables directly
            let forensics = self.engine.read_telemetry_probe();
            if forensics.tripwire_triggered == 1 {
                println!("\n🚨 [FORENSIC FAULT DETECTED] Node Coordinates: ({}, {}, {})", forensics.debug_x, forensics.debug_y, forensics.debug_z);
                println!("Density: {:.6} | Velocity Vector: ({:.4}, {:.4}, {:.4})", forensics.local_rho, forensics.local_ux, forensics.local_uy, forensics.local_uz);
                println!("Forces -> Korteweg Fx: {:.6} | Phase-Lock Fx: {:.6}", forensics.korteweg_fx, forensics.phase_lock_fx);
                println!("Lorentz Multiplier: {:.4}", forensics.lorentz_factor);

                // 🌟 NEW QUANTUM VACUUM DIAGNOSTICS
                println!("=========================================================================");
                println!("Vacuum Shear (Pi Magnitude) : {:.6}", forensics.max_pi_magnitude);
                println!("Polarized Relaxation (Tau)  : {:.6}", forensics.peak_polarized_tau);
                println!("Cumulative Mass Inversions  : {}", forensics.negative_population_count);
                println!("=========================================================================");

                std::process::exit(1);
            }

            // 🌟 THE AUTO-TERMINATION & DIAGNOSTICS BLOCK
            if let Some(metrics) = self.engine.poll_diagnostics() {
                // We only check for evaporation AFTER the 20,000 tick injection phase ends
                if self.engine.physics_passes > 25_000 {
                    if metrics.knot_volume == 0.0 {
                        self.zero_volume_checks += 1;
                        if self.zero_volume_checks >= 5 {
                            // 5 consecutive checks = 500 physics ticks of absolute zero volume
                            println!("🛑 Knot evaporated. Zero volume detected for 500 ticks. Terminating simulation.");
                            event_loop.exit();
                            return;
                        }
                    } else {
                        // The knot recovered or is stable, reset the death counter
                        self.zero_volume_checks = 0;
                    }
                }
            }

            // 🌟 CHECKPOINT GENERATOR (Generation Mode)
            // if self.engine.physics_passes == 75_000 {
            //     println!("[STATE: MILESTONE] Tick 25,000 reached. Adiabatic settling complete.");
            //     self.engine.save_checkpoint("proton_stable_25k.bin");
            //     println!("🛑 Checkpoint saved. You can now enable 'tuning_mode' in main() to run rapid parameter sweeps.");
            //
            //     // Optional: Uncomment the next line if you want the app to automatically close after generating the file
            //     event_loop.exit();
            // }

            // Hard maximum tick limit
            if self.engine.physics_passes >= 100_000 {
                println!("🛑 Maximum tick limit reached (60,000). Terminating simulation safely.");
                event_loop.exit();
                return;
            }

            // 🌟 AUTOMATIC INJECTION TRIGGER
            if self.engine.physics_passes >= 100 && !self.engine.has_injected
            {
                println!("[STATE: TRANSITION] 100 Passes Reached. Injecting Atom.");
                let cx = self.engine.width as f32 / 2.0;
                let cy = self.engine.height as f32 / 2.0;
                let cz = self.engine.depth as f32 / 2.0;
                let phys = self.engine.physics;

                println!("cz: {}", self.engine.depth as f32 / 2.0);

                self.engine.blueprint.inject_nucleus(cx, cy, cz, 1, 0, &phys);
                self.engine.queue.write_buffer(&self.engine.blueprint_uniform_buffer, 0, bytemuck::cast_slice(&[self.engine.blueprint]));
                self.engine.injection_ticks = 1200; // 20k-tick adiabatic phase
                self.engine.event_flag = Some("AUTO_INJECT_Z1_N0".to_string());
                self.engine.has_injected = true;
            }

            // if self.engine.physics_passes % 100 == 0 {
            //     println!("Tick: {} processed safely.", self.engine.physics_passes);
            // }
        }
    }
}

// 🌟 THE ASYNC BOOT ENVIRONMENT
// This allows us to resolve the adapter and device futures natively via WGPU's built-in block_on helper
async fn create_physics_context() -> (Arc<wgpu::Device>, Arc<wgpu::Queue>, wgpu::Texture, wgpu::TextureView) {
    // 🌟 WGPU 29 FIX: InstanceDescriptor requires explicit fields, Default trait is removed.
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::METAL,
        flags: wgpu::InstanceFlags::default(),
        backend_options: wgpu::BackendOptions::default(),
        display: None,
        memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
    });

    let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }).await.unwrap();

    let hardware_limits = adapter.limits();
    println!("🔓 [SILICON UNLOCKED] Maximum Buffer Size: {} bytes", hardware_limits.max_buffer_size);

    // 🌟 WGPU 29 FIX: DeviceDescriptor initialization with all required new fields
    let (device, queue) = adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("Pure Physics Device"),
            required_features: wgpu::Features::SHADER_F16,
            required_limits: hardware_limits,
            memory_hints: wgpu::MemoryHints::Performance, // Direct optimization mapping
            experimental_features: wgpu::ExperimentalFeatures::default(),
            trace: wgpu::Trace::Off,
        },
    ).await.unwrap();

    let device = Arc::new(device);
    let queue = Arc::new(queue);

    let size = wgpu::Extent3d { width: 800, height: 750, depth_or_array_layers: 1 };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Soup VRAM Texture"), size, mip_level_count: 1, sample_count: 1,
        dimension: wgpu::TextureDimension::D2, format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

    (device, queue, texture, view)
}

fn main() {
    // 🌟 POLLSTER REMOVAL FIX: Use standard futures implementation or a quick macro execution context
    let (device, queue, texture, view_for_engine) = pollster::block_on(create_physics_context());

    // Instantiate the Engine at the true 512x512x512 cubic resolution
    let engine = SoupEngine::new(
        device.clone(),
        queue.clone(),
        view_for_engine,
        texture.clone(),
        376, 376, 376//376
    );

    // let phys_copy = engine.physics;
    // engine.logger.start_new_run(engine.width, engine.height, engine.depth, &phys_copy);

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = SoupApp {
        engine,
        is_left_clicked: false,
        last_mouse_pos: winit::dpi::PhysicalPosition::new(0.0, 0.0),
        sim_paused: false,
        window: None,
        zero_volume_checks: 0,
    };

    event_loop.run_app(&mut app).unwrap();
}