use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;
use sysinfo::{Pid, ProcessRefreshKind, ProcessesToUpdate, System};

#[derive(Debug, Clone, Copy)]
pub struct ResourceSample {
    pub cpu_percent: f32,
    pub ram_bytes: u64,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ResourceSummary {
    pub samples: usize,
    pub avg_cpu_percent: f32,
    pub max_cpu_percent: f32,
    pub avg_ram_mb: f64,
    pub max_ram_mb: f64,
}

pub struct ResourceMonitor {
    running: Arc<AtomicBool>,
    samples: Arc<Mutex<Vec<ResourceSample>>>,
    handle: Option<JoinHandle<()>>,
}

impl ResourceMonitor {
    pub fn start(interval: Duration) -> Self {
        let running = Arc::new(AtomicBool::new(true));
        let samples = Arc::new(Mutex::new(Vec::new()));

        let thread_running = running.clone();
        let thread_samples = samples.clone();

        let handle = thread::spawn(move || {
            let pid = Pid::from(std::process::id() as usize);
            let mut sys = System::new_all();

            sys.refresh_processes_specifics(
                ProcessesToUpdate::Some(&[pid]),
                false,
                ProcessRefreshKind::everything().with_cpu().with_memory(),
            );

            let interval = interval.max(Duration::from_millis(50));

            while thread_running.load(Ordering::Relaxed) {
                thread::sleep(interval);

                sys.refresh_processes_specifics(
                    ProcessesToUpdate::Some(&[pid]),
                    false,
                    ProcessRefreshKind::everything().with_cpu().with_memory(),
                );

                if let Some(process) = sys.process(pid) {
                    let sample = ResourceSample {
                        cpu_percent: process.cpu_usage(),
                        ram_bytes: process.memory(),
                    };

                    if let Ok(mut guard) = thread_samples.lock() {
                        guard.push(sample);
                    }
                }
            }
        });

        Self {
            running,
            samples,
            handle: Some(handle),
        }
    }

    pub fn stop(mut self) -> ResourceSummary {
        self.running.store(false, Ordering::Relaxed);

        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }

        let samples = match self.samples.lock() {
            Ok(guard) => guard.clone(),
            Err(_) => Vec::new(),
        };

        if samples.is_empty() {
            return ResourceSummary::default();
        }

        let count = samples.len();

        let avg_cpu_percent = samples.iter().map(|s| s.cpu_percent).sum::<f32>() / count as f32;

        let max_cpu_percent = samples.iter().map(|s| s.cpu_percent).fold(0.0f32, f32::max);

        let avg_ram_mb = samples.iter().map(|s| s.ram_bytes as f64).sum::<f64>()
            / count as f64
            / 1024.0
            / 1024.0;

        let max_ram_mb =
            samples.iter().map(|s| s.ram_bytes).max().unwrap_or(0) as f64 / 1024.0 / 1024.0;

        ResourceSummary {
            samples: count,
            avg_cpu_percent,
            max_cpu_percent,
            avg_ram_mb,
            max_ram_mb,
        }
    }
}
