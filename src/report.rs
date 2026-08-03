use anyhow::Result;
use serde::Serialize;
use std::path::Path;

#[derive(Debug, Clone, Serialize)]
pub struct BenchRow {
    pub file: String,
    pub dataset: String,
    pub snr_db: Option<f32>,

    pub duration_ms: u64,
    pub frames: usize,
    pub speech_frames: u64,
    pub speech_percentage: f32,

    pub precision: f32,
    pub recall: f32,
    pub f1_score: f32,
    pub accuracy: f32,

    pub tp: u64,
    pub fp: u64,
    pub fn_count: u64,
    pub tn: u64,

    pub avg_onset_latency_ms: Option<f64>,
    pub avg_abs_onset_latency_ms: Option<f64>,

    pub resource_samples: usize,
    pub avg_cpu_percent: f32,
    pub max_cpu_percent: f32,
    pub avg_ram_mb: f64,
    pub max_ram_mb: f64,
}

pub fn write_csv(path: &Path, rows: &[BenchRow]) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let mut writer = csv::Writer::from_path(path)?;

    for row in rows {
        writer.serialize(row)?;
    }

    writer.flush()?;

    Ok(())
}
