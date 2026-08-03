use anyhow::{bail, Context, Result};
use clap::Parser;
use serde::Deserialize;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;
use vad_core::{
    audio,
    engine::{process_offline, FRAME_MS, MIN_SEGMENT_MS, SAMPLE_RATE},
    metrics::{
        bool_frames_to_segments, compute_classification, frame_labels, onset_latency, Annotation,
    },
    monitor::ResourceMonitor,
    report::{write_csv, BenchRow},
};

#[derive(Parser, Debug)]
#[command(name = "vad-bench")]
#[command(about = "Offline benchmark for VAD with resource monitoring")]
struct Args {
    /// Путь к manifest.json
    #[arg(long)]
    manifest: PathBuf,

    /// Директория для отчетов
    #[arg(long, default_value = "reports")]
    output: PathBuf,

    /// Интервал мониторинга ресурсов в миллисекундах
    #[arg(long, default_value_t = 200)]
    monitor_interval_ms: u64,

    /// Quality gate: минимальный средний F1
    #[arg(long)]
    min_f1: Option<f32>,

    /// Quality gate: минимальный средний recall
    #[arg(long)]
    min_recall: Option<f32>,

    /// Quality gate: минимальный средний precision
    #[arg(long)]
    min_precision: Option<f32>,

    /// Quality gate: максимальная средняя CPU в процентах
    #[arg(long)]
    max_avg_cpu_percent: Option<f32>,

    /// Quality gate: максимальная пиковая RAM в МБ
    #[arg(long)]
    max_max_ram_mb: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct BenchItem {
    file: PathBuf,
    annotation: PathBuf,
    dataset: String,

    #[serde(default)]
    snr_db: Option<f32>,
}

fn resolve_path(base: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        base.join(path)
    }
}

fn main() -> Result<()> {
    let args = Args::parse();

    let manifest_raw = fs::read_to_string(&args.manifest)
        .with_context(|| format!("Не удалось прочитать manifest: {:?}", args.manifest))?;

    let items: Vec<BenchItem> =
        serde_json::from_str(&manifest_raw).context("Не удалось распарсить manifest.json")?;

    fs::create_dir_all(&args.output)
        .with_context(|| format!("Не удалось создать output dir: {:?}", args.output))?;

    let manifest_dir = args
        .manifest
        .parent()
        .unwrap_or_else(|| Path::new(""))
        .to_path_buf();

    let mut rows: Vec<BenchRow> = Vec::new();

    for item in items {
        let file_path = resolve_path(&manifest_dir, &item.file);
        let annotation_path = resolve_path(&manifest_dir, &item.annotation);

        println!("Processing: {}", file_path.display());

        let samples = audio::load_wav_mono_i16_16k(&file_path)?;
        let annotation = Annotation::load(&annotation_path)?;

        let monitor = ResourceMonitor::start(Duration::from_millis(args.monitor_interval_ms));

        let predictions = process_offline(&samples);

        let resource_summary = monitor.stop();

        let labels = frame_labels(predictions.len(), FRAME_MS, &annotation);
        let metrics = compute_classification(&labels, &predictions);

        let segments = bool_frames_to_segments(&predictions, FRAME_MS, MIN_SEGMENT_MS);
        let latency = onset_latency(&annotation, &segments);

        let duration_ms = (samples.len() as u64 * 1000) / SAMPLE_RATE as u64;

        let speech_frames = predictions.iter().filter(|&&v| v).count() as u64;

        let speech_percentage = if predictions.is_empty() {
            0.0
        } else {
            speech_frames as f32 / predictions.len() as f32 * 100.0
        };

        rows.push(BenchRow {
            file: file_path.display().to_string(),
            dataset: item.dataset,
            snr_db: item.snr_db,

            duration_ms,
            frames: predictions.len(),
            speech_frames,
            speech_percentage,

            precision: metrics.precision,
            recall: metrics.recall,
            f1_score: metrics.f1_score,
            accuracy: metrics.accuracy,

            tp: metrics.tp,
            fp: metrics.fp,
            fn_count: metrics.fn_count,
            tn: metrics.tn,

            avg_onset_latency_ms: latency.map(|v| v.avg_ms),
            avg_abs_onset_latency_ms: latency.map(|v| v.avg_abs_ms),

            resource_samples: resource_summary.samples,
            avg_cpu_percent: resource_summary.avg_cpu_percent,
            max_cpu_percent: resource_summary.max_cpu_percent,
            avg_ram_mb: resource_summary.avg_ram_mb,
            max_ram_mb: resource_summary.max_ram_mb,
        });
    }

    let csv_path = args.output.join("bench.csv");
    write_csv(&csv_path, &rows)?;

    let json_path = args.output.join("bench.json");
    let json = serde_json::to_string_pretty(&rows)?;
    fs::write(&json_path, json)?;

    println!("\nReports:");
    println!("  CSV: {}", csv_path.display());
    println!("  JSON: {}", json_path.display());

    if rows.is_empty() {
        println!("\nNo rows collected.");
        return Ok(());
    }

    let n = rows.len() as f32;
    let n64 = rows.len() as f64;

    let avg_f1 = rows.iter().map(|r| r.f1_score).sum::<f32>() / n;
    let avg_recall = rows.iter().map(|r| r.recall).sum::<f32>() / n;
    let avg_precision = rows.iter().map(|r| r.precision).sum::<f32>() / n;
    let avg_cpu = rows.iter().map(|r| r.avg_cpu_percent).sum::<f32>() / n;
    let avg_ram_mb = rows.iter().map(|r| r.avg_ram_mb).sum::<f64>() / n64;
    let max_ram_mb = rows.iter().map(|r| r.max_ram_mb).fold(0.0f64, f64::max);

    println!("\nSummary:");
    println!("  files:              {}", rows.len());
    println!("  avg F1:             {:.4}", avg_f1);
    println!("  avg precision:      {:.4}", avg_precision);
    println!("  avg recall:         {:.4}", avg_recall);
    println!("  avg CPU:            {:.2}%", avg_cpu);
    println!("  avg RAM:            {:.2} MB", avg_ram_mb);
    println!("  max RAM:            {:.2} MB", max_ram_mb);

    if let Some(min_f1) = args.min_f1 {
        if avg_f1 < min_f1 {
            bail!("Quality gate failed: avg F1 {:.4} < {:.4}", avg_f1, min_f1);
        }
    }

    if let Some(min_recall) = args.min_recall {
        if avg_recall < min_recall {
            bail!(
                "Quality gate failed: avg recall {:.4} < {:.4}",
                avg_recall,
                min_recall
            );
        }
    }

    if let Some(min_precision) = args.min_precision {
        if avg_precision < min_precision {
            bail!(
                "Quality gate failed: avg precision {:.4} < {:.4}",
                avg_precision,
                min_precision
            );
        }
    }

    if let Some(max_avg_cpu_percent) = args.max_avg_cpu_percent {
        if avg_cpu > max_avg_cpu_percent {
            bail!(
                "Quality gate failed: avg CPU {:.2}% > {:.2}%",
                avg_cpu,
                max_avg_cpu_percent
            );
        }
    }

    if let Some(max_max_ram_mb) = args.max_max_ram_mb {
        if max_ram_mb > max_max_ram_mb {
            bail!(
                "Quality gate failed: max RAM {:.2} MB > {:.2} MB",
                max_ram_mb,
                max_max_ram_mb
            );
        }
    }

    Ok(())
}
