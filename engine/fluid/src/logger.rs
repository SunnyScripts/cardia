use std::fs::{File, OpenOptions};
use std::io::Write;
use chrono::{Local, SecondsFormat};
use sysinfo::System;

use crate::engine::{EngineMetrics, PhysicsState, CS_LATTICE, L_NODE, SCALE_FACTOR};

pub struct RunLogger {
    pub file: Option<File>,
}

impl RunLogger {
    pub fn new() -> Self {
        Self { file: None }
    }

    pub fn start_new_run(
        &mut self,
        width: u32,
        height: u32,
        depth: u32,
        physics: &PhysicsState, // 🌟 Updated to the unified physics struct
    ) {
        let timestamp = Local::now().format("%Y%m%d_%H%M%S").to_string();
        let filename = format!("cardia_run_log_{}.csv", timestamp);

        let mut file = OpenOptions::new().write(true).create(true).truncate(true).open(&filename).unwrap();

        let mut sys = System::new_all();
        sys.refresh_all();

        let cpu_name = sys.cpus().first().map(|cpu| cpu.brand()).unwrap_or("Unknown CPU").to_string();
        let total_mem = sys.total_memory() / 1048576;

        // =========================================================================
        // METADATA HEADER (Purged of FENE-P, updated for GBC Solitons)
        // =========================================================================
        writeln!(file, "# CARDIA ENGINE: EMPIRICAL RUN LOG").unwrap();
        writeln!(file, "# RUN START (LOCAL): {}", Local::now().to_rfc2822()).unwrap();
        writeln!(file, "# HARDWARE PROFILE : {} | {} MB RAM", cpu_name, total_mem).unwrap();
        writeln!(file, "# SHADER MODEL     : D3Q39 Recrusive Regularized KBC Entropic LBM | Ginzburg-Landau Cohesion (GBC) | Topological Soliton").unwrap();
        writeln!(file, "# ---------------------------------------------------------------------------").unwrap();
        writeln!(file, "# GRID DIMENSIONS  : {} x {} x {} ({} Nodes)", width, height, depth, width * height * depth).unwrap();
        writeln!(file, "# FUNDAMENTAL SCALE: L_NODE = {:e}m, LATTICE SPEED of LIGHT: {:.8}", L_NODE, CS_LATTICE).unwrap();
        writeln!(file, "# COARSE GRAINING  : Scale Factor (S) = {:e}", SCALE_FACTOR).unwrap();
        writeln!(file, "# INJECTION PARAMS : Coherence_Length_LU = {:.2} | Axial Push = {:.6} | Polarization = {:.6}", physics.coherence_length_lu, physics.u_axial_coeff, physics.c_polarization).unwrap();
        writeln!(file, "# FLUID CONSTANTS  : Tau = {:.8} | ZPE Amplitude = {:.8}", physics.tau, physics.zpe_amplitude).unwrap();
        writeln!(file, "# GBC TOPOLOGY     : Shan_Chen_G (Quantum Surface Tension) = {:.6} | Phase_Lock_Rate = {:.6}", physics.shan_chen_g, physics.phase_lock_rate).unwrap();
        writeln!(file, "# ===========================================================================").unwrap();

        // 🌟 PURGED: Removed Madelung_Pressure, VC_Pressure, Elastic_Tension, Max_Tensor_Trace, Weissenberg, Deborah
        let headers = [
            "Wall_Clock_Time", "Sim_Tick", "Lattice_Mass", "Vacuum_KE", "Center_X", "Center_Y", "Center_Z",
            "Mach", "Net_Speed_LU", "Heading_Yaw", "Heading_Pitch", // Extracted kinematics
            "Vacuum_Mom_Z", "Knot_Mom_Z", "Boundary_Echo", "Knot_Volume",
            "Knot_Peak_Rho", "Knot_Min_Rho", "Event", "Total_Enstrophy", "Total_Helicity",
            "Casimir_Delta", "Real_Time", "Shape_Anisotropy", "Orbital_Amplitude", "Max_Vorticity",
            "Compton_Freq", "Quantum_Circulation", "Ang_Mom_X", "Ang_Mom_Y", "Ang_Mom_Z",
            "Ambient_Center_Noise", "Total_System_Energy", "Energy_Loss", // Added Energy Loss
            "G_Measured", "Target_Mass_Error", "Target_Volume_Error",
            "Stability_Faults", "Spatial_Correlation", // Added missing UI metrics
            // "Radial_Bins"
        ];

        writeln!(file, "{}", headers.join(",")).unwrap();

        self.file = Some(file);
        println!("📝 Automated Logging Started: {}", filename);
    }

    pub fn log_tick(&mut self, tick: u64, metrics: &EngineMetrics) {
        if let Some(file) = &mut self.file {
            let radial_str = metrics.radial_probability.iter()
                .map(|v| format!("{:.6}", v / metrics.sample_count as f32))
                .collect::<Vec<String>>()
                .join(";");

            let timestamp = Local::now().to_rfc3339_opts(SecondsFormat::Millis, true);

            let row_data = vec![
                timestamp,
                tick.to_string(),
                metrics.mass.clone(),
                metrics.kinetic.clone(),
                format!("{:.4}", metrics.center_x),
                format!("{:.4}", metrics.center_y),
                format!("{:.4}", metrics.center_z),
                metrics.mach.clone(),
                format!("{:.6}", metrics.net_speed),      // 🌟 NEW
                format!("{:.2}", metrics.heading_yaw),    // 🌟 NEW
                format!("{:.2}", metrics.heading_pitch),  // 🌟 NEW
                format!("{:.6}", metrics.vacuum_mom_z),
                format!("{:.6}", metrics.knot_mom_z),
                format!("{:.6}", metrics.boundary_echo),
                metrics.knot_volume.to_string(),
                metrics.knot_peak_rho.to_string(),
                format!("{:.6}", metrics.knot_min_rho),
                metrics.event.clone(),
                format!("{:.6}", metrics.total_enstrophy),
                format!("{:.6}", metrics.total_helicity),
                metrics.casimir_delta.to_string(),
                format!("{:.25}", metrics.real_time),
                metrics.shape_anisotropy.to_string(),
                metrics.orbital_amplitude.to_string(),
                metrics.max_vorticity.to_string(),
                metrics.compton_freq.to_string(),
                metrics.quantum_circulation.to_string(),
                metrics.ang_mom_x.to_string(),
                metrics.ang_mom_y.to_string(),
                metrics.ang_mom_z.to_string(),
                metrics.ambient_center_noise.to_string(),
                metrics.total_system_energy.to_string(),
                metrics.energy_loss.to_string(),          // 🌟 NEW
                metrics.g_measured.to_string(),
                metrics.target_mass_error.to_string(),
                metrics.target_volume_error.to_string(),
                metrics.stability.clone(),                // 🌟 NEW (NaN, Inf count)
                metrics.correlation.clone(),              // 🌟 NEW
                // radial_str,
            ];

            writeln!(file, "{}", row_data.join(",")).unwrap();
        }
    }

    pub fn stop_run(&mut self) {
        self.file = None;
        println!("💾 Log file closed and saved safely.");
    }
}