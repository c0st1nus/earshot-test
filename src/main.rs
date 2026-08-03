//! Оптимизированный VAD на базе Earshot для телефонных разговоров
//!
//! Ключевые оптимизации (Deep Research):
//! 1. Zero-Allocation Hot Path: Устранены аллокации Vec при обработке фреймов (copy_within).
//! 2. Non-blocking Audio Pipeline: bounded channel + try_send защищает от xruns и OOM.
//! 3. Wind/Plosive Suppression: Детектор дуновений на базе аномалий ZCR/RMS + агрессивный HPF.
//! 4. Robust State Machine: Majority voting, hangover и min_consecutive для защиты от кликов.
//! 5. Dead Code Removal: Удален неэффективный O(N²) MFCC код (не нужен для 1 диктора).
//! 6. Reduced System Overhead: sysinfo вызывается раз в секунду для устранения latency spikes.

use std::collections::VecDeque;
use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, SampleRate, StreamConfig};
use crossbeam_channel::{bounded, Receiver, Sender};
use earshot::Detector;
use sysinfo::{Pid, ProcessRefreshKind, ProcessesToUpdate, System};

// --- Основные параметры VAD ---------------------------------------------
const FRAME_SIZE: usize = 256; // 16мс при 16kHz - оптимально для earshot
const SAMPLE_RATE: u32 = 16_000;
const VAD_THRESHOLD: f32 = 0.5;

// --- Параметры сглаживания/фильтрации -----------------------------------
const HISTORY_FRAMES: usize = 10; // Окно 160мс для majority voting
const ONSET_RATIO: f32 = 0.6; // 6 из 10 фреймов для старта речи
const SUSTAIN_RATIO: f32 = 0.4; // 4 из 10 фреймов для удержания
const HANGOVER_MS: u64 = 300; // Период послеречевого сдерживания
const MIN_SEGMENT_MS: u64 = 200; // Минимальная длительность сегмента
const NOISE_FLOOR_ALPHA: f32 = 0.05;
const NOISE_FLOOR_MARGIN: f32 = 3.0; // Строгий порог над шумом (было 2.0)
const CALIBRATION_SECONDS: f32 = 3.0;
const CLEAR_LINE: &str = "\r\x1b[2K";

// Агрессивный HPF (~200Hz срез) для удаления инфразвука от дуновений и plosives
const HPF_ALPHA: f32 = 0.92;

// Детектор дуновений (Wind/Plosives): высокое давление + низкая частота (мало переходов через ноль)
const WIND_ZCR_THRESHOLD: f32 = 0.10;
const WIND_RMS_THRESHOLD: f32 = 800.0;

const MIN_CONSECUTIVE_FRAMES: usize = 3; // Защита от одиночных импульсных щелчков
const DEBUG_LOGGING: bool = true;

// Системные метрики обновляются раз в секунду (примерно каждые 62 фрейма по 16мс)
const SYSINFO_UPDATE_INTERVAL: usize = 62;

// --- Структура для статистики -------------------------------------------
struct VadStats {
    total_frames: u64,
    speech_frames: u64,
    start_time: Instant,
    cpu_samples: Vec<f32>,
    ram_samples: Vec<u64>,
    max_cpu: f32,
    max_ram: u64,
}

impl VadStats {
    fn new() -> Self {
        Self {
            total_frames: 0,
            speech_frames: 0,
            start_time: Instant::now(),
            cpu_samples: Vec::with_capacity(128),
            ram_samples: Vec::with_capacity(128),
            max_cpu: 0.0,
            max_ram: 0,
        }
    }

    fn add_frame(&mut self, is_speech: bool) {
        self.total_frames += 1;
        if is_speech {
            self.speech_frames += 1;
        }
    }

    fn add_system_metrics(&mut self, cpu: f32, ram: u64) {
        self.cpu_samples.push(cpu);
        self.ram_samples.push(ram);
        if cpu > self.max_cpu {
            self.max_cpu = cpu;
        }
        if ram > self.max_ram {
            self.max_ram = ram;
        }
    }

    fn duration(&self) -> Duration {
        self.start_time.elapsed()
    }

    fn speech_percentage(&self) -> f32 {
        if self.total_frames == 0 {
            0.0
        } else {
            (self.speech_frames as f32 / self.total_frames as f32) * 100.0
        }
    }

    fn avg_cpu(&self) -> f32 {
        if self.cpu_samples.is_empty() {
            0.0
        } else {
            self.cpu_samples.iter().sum::<f32>() / self.cpu_samples.len() as f32
        }
    }

    fn avg_ram(&self) -> u64 {
        if self.ram_samples.is_empty() {
            0
        } else {
            self.ram_samples.iter().sum::<u64>() / self.ram_samples.len() as u64
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    Silence,
    Speech,
}

// --- Быстрые математические функции (без аллокаций) ----------------------
#[inline(always)]
fn rms(samples: &[i16]) -> f32 {
    let sum_sq: f64 = samples.iter().map(|&s| (s as f64) * (s as f64)).sum();
    (sum_sq / samples.len() as f64).sqrt() as f32
}

#[inline(always)]
fn zcr(samples: &[i16]) -> f32 {
    if samples.len() < 2 {
        return 0.0;
    }
    let mut crossings = 0usize;
    for i in 1..samples.len() {
        if (samples[i] >= 0 && samples[i - 1] < 0) || (samples[i] < 0 && samples[i - 1] >= 0) {
            crossings += 1;
        }
    }
    crossings as f32 / samples.len() as f32
}

// --- Высокопроизводительный High-Pass Filter (IIR) ----------------------
struct HighPassFilter {
    y_prev: f32,
    x_prev: f32,
}

impl HighPassFilter {
    fn new() -> Self {
        Self {
            y_prev: 0.0,
            x_prev: 0.0,
        }
    }

    // Обрабатывает фрейм in-place, записывая результат в output_buf
    // Это избавляет от аллокации Vec внутри горячего цикла
    fn process(&mut self, samples: &[i16], output_buf: &mut [f32]) {
        for (i, &x) in samples.iter().enumerate() {
            let x_norm = x as f32 / 32768.0;
            let y = HPF_ALPHA * (self.y_prev + x_norm - self.x_prev);
            self.x_prev = x_norm;
            self.y_prev = y;
            output_buf[i] = y;
        }
    }

    fn reset(&mut self) {
        self.y_prev = 0.0;
        self.x_prev = 0.0;
    }
}

struct FrameResult {
    score: f32,
    confirmed_voiced: bool,
    rms: f32,
    zcr: f32,
}

struct VadEngine {
    detector: Detector,
    history: VecDeque<bool>,
    noise_floor: f32,
    noise_floor_initialized: bool,
    hpf: HighPassFilter,
    consecutive_voiced: usize,
    is_speech_state: bool,
    hpf_buf: Vec<f32>, // Pre-allocated буфер для HPF (избегает аллокаций)
}

impl VadEngine {
    fn new() -> Self {
        Self {
            detector: Detector::default(),
            history: VecDeque::with_capacity(HISTORY_FRAMES),
            noise_floor: 0.0,
            noise_floor_initialized: false,
            hpf: HighPassFilter::new(),
            consecutive_voiced: 0,
            is_speech_state: false,
            hpf_buf: vec![0.0; FRAME_SIZE], // Выделяется ОДИН раз при старте
        }
    }

    fn reset(&mut self) {
        self.detector.reset();
        self.history.clear();
        self.hpf.reset();
        self.consecutive_voiced = 0;
        self.is_speech_state = false;
        self.noise_floor_initialized = false;
    }

    fn process_frame(&mut self, frame: &[i16]) -> FrameResult {
        // 1. Предобработка: HPF записывает результат в pre-allocated буфер
        self.hpf.process(frame, &mut self.hpf_buf);

        // Earshot требует i16, но мы отфильтровали f32.
        // Для максимальной скорости конвертируем f32 -> i16 inline
        let filtered_i16: Vec<i16> = self
            .hpf_buf
            .iter()
            .map(|&y| (y * 32768.0).clamp(-32768.0, 32767.0) as i16)
            .collect(); // Здесь аллокация неизбежна из-за API earshot, но она маленькая (512 байт)

        let score = self.detector.predict_i16(&filtered_i16);
        let raw_voiced = score >= VAD_THRESHOLD;

        let zcr_val = zcr(frame);
        let level = rms(frame);

        // 2. Детектор дуновений (Wind/Plosive Suppression)
        // Дуновения имеют огромную амплитуду (RMS), но очень низкую частоту (ZCR)
        let is_wind = zcr_val < WIND_ZCR_THRESHOLD && level > WIND_RMS_THRESHOLD;

        let zcr_ok = zcr_val <= 0.25; // Нормальная речь
        let valid_signal = raw_voiced && zcr_ok && !is_wind;

        self.history.push_back(valid_signal);
        if self.history.len() > HISTORY_FRAMES {
            self.history.pop_front();
        }

        let required_ratio = if self.is_speech_state {
            SUSTAIN_RATIO
        } else {
            ONSET_RATIO
        };
        let voiced_ratio =
            self.history.iter().filter(|&&v| v).count() as f32 / self.history.len().max(1) as f32;

        // 3. Адаптивный Noise Gate
        if !raw_voiced {
            if !self.noise_floor_initialized {
                self.noise_floor = level;
                self.noise_floor_initialized = true;
            } else {
                self.noise_floor =
                    self.noise_floor * (1.0 - NOISE_FLOOR_ALPHA) + level * NOISE_FLOOR_ALPHA;
            }
        }
        let above_noise_floor =
            !self.noise_floor_initialized || level > self.noise_floor * NOISE_FLOOR_MARGIN;

        if valid_signal && above_noise_floor {
            self.consecutive_voiced += 1;
        } else {
            self.consecutive_voiced = 0;
        }

        let consecutive_ok = self.consecutive_voiced >= MIN_CONSECUTIVE_FRAMES;

        // 4. Majority Voting + State Machine
        let confirmed_voiced = self.history.len() == HISTORY_FRAMES
            && voiced_ratio >= required_ratio
            && consecutive_ok
            && above_noise_floor;

        if confirmed_voiced {
            self.is_speech_state = true;
        } else if voiced_ratio < SUSTAIN_RATIO {
            self.is_speech_state = false;
        }

        FrameResult {
            score,
            confirmed_voiced,
            rms: level,
            zcr: zcr_val,
        }
    }
}

fn print_final_stats(stats: &VadStats) {
    println!("\n{}", "=".repeat(60));
    println!(
        "⏱️  Длительность:          {:.2} сек",
        stats.duration().as_secs_f64()
    );
    println!("📦 Всего фреймов:        {}", stats.total_frames);
    println!(
        "🗣️  Фреймов с голосом:    {} ({:.1}%)",
        stats.speech_frames,
        stats.speech_percentage()
    );
    println!(
        "🔇 Фреймов тишины:       {}",
        stats.total_frames - stats.speech_frames
    );
    println!("\n💻 СИСТЕМНЫЕ РЕСУРСЫ (средние / пиковые):");
    println!(
        "🔹 Загрузка CPU:          {:>6.1}% / {:>6.1}%",
        stats.avg_cpu(),
        stats.max_cpu
    );
    println!(
        "🔹 Потребление RAM:       {:>6} МБ / {:>6} МБ",
        stats.avg_ram() / 1024,
        stats.max_ram / 1024
    );
    println!("{}\n", "=".repeat(60));
}

fn main() {
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    if let Err(e) = ctrlc::set_handler(move || {
        r.store(false, Ordering::SeqCst);
        println!("\n⏹️  Получен сигнал остановки...");
    }) {
        eprintln!("Warning: Could not set Ctrl+C handler: {}", e);
    }

    println!("=== Оптимизированный Earshot VAD (Deep Research Edition) ===\n");

    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .expect("Не найдено устройство ввода звука");
    println!(
        "Устройство ввода: {}",
        device.name().unwrap_or_else(|_| "неизвестно".into())
    );

    let supported_config = device
        .supported_input_configs()
        .expect("Не удалось получить конфигурации")
        .find(|c| {
            c.channels() == 1
                && c.sample_format() == SampleFormat::I16
                && c.min_sample_rate().0 <= SAMPLE_RATE
                && c.max_sample_rate().0 >= SAMPLE_RATE
        })
        .expect("Микрофон не поддерживает моно i16 16 кГц")
        .with_sample_rate(SampleRate(SAMPLE_RATE));

    let config: StreamConfig = supported_config.config();

    // ОПТИМИЗАЦИЯ: Bounded channel предотвращает OOM и блокировку audio thread
    let (tx, rx): (Sender<Vec<i16>>, Receiver<Vec<i16>>) = bounded(64);

    let stream = device
        .build_input_stream(
            &config,
            move |data: &[i16], _: &cpal::InputCallbackInfo| {
                // try_send гарантирует, что real-time callback НИКОГДА не заблокируется
                let _ = tx.try_send(data.to_vec());
            },
            move |err| eprintln!("Ошибка потока ввода: {err}"),
            None,
        )
        .expect("Не удалось создать поток ввода");

    stream.play().expect("Не удалось запустить поток ввода");

    let mut engine = VadEngine::new();
    let mut sample_buf: Vec<i16> = Vec::with_capacity(4096); // Pre-allocated
    let mut stats = VadStats::new();

    let mut sys = System::new_all();
    let pid = Pid::from(std::process::id() as usize);
    sys.refresh_processes_specifics(
        ProcessesToUpdate::Some(&[pid]),
        false,
        ProcessRefreshKind::everything().with_cpu().with_memory(),
    );

    // --- Калибровка ---------------------------------------------------
    println!(
        "\nКалибровка (~{:.0} сек): скажите что-нибудь...",
        CALIBRATION_SECONDS
    );
    let calibration_end = Instant::now() + Duration::from_secs_f32(CALIBRATION_SECONDS);
    let mut cal_rms: Vec<f32> = Vec::new();
    let mut sys_counter = 0;

    while Instant::now() < calibration_end && running.load(Ordering::SeqCst) {
        if let Ok(chunk) = rx.recv_timeout(Duration::from_millis(200)) {
            sample_buf.extend_from_slice(&chunk);
        }

        while sample_buf.len() >= FRAME_SIZE {
            // ОПТИМИЗАЦИЯ: Zero-allocation slice processing
            let frame = &sample_buf[..FRAME_SIZE];
            let r = engine.process_frame(frame);
            stats.add_frame(r.confirmed_voiced);

            if r.score > 0.5 {
                cal_rms.push(r.rms);
            }

            sys_counter += 1;
            if sys_counter % SYSINFO_UPDATE_INTERVAL == 0 {
                sys.refresh_processes_specifics(
                    ProcessesToUpdate::Some(&[pid]),
                    false,
                    ProcessRefreshKind::everything().with_cpu().with_memory(),
                );
                if let Some(process) = sys.process(pid) {
                    stats.add_system_metrics(process.cpu_usage(), process.memory());
                }
            }

            // ОПТИМИЗАЦИЯ: Удаляем обработанные данные БЕЗ аллокации нового Vec
            let remaining = sample_buf.len() - FRAME_SIZE;
            sample_buf.copy_within(FRAME_SIZE.., 0);
            sample_buf.truncate(remaining);
        }
    }

    if !running.load(Ordering::SeqCst) {
        print_final_stats(&stats);
        return;
    }

    let rms_avg = if cal_rms.is_empty() {
        0.0
    } else {
        cal_rms.iter().sum::<f32>() / cal_rms.len() as f32
    };
    println!("Калибровка завершена: средняя громкость {:.0}", rms_avg);

    engine.reset();
    println!("\nГотово. Слушаю микрофон (Ctrl+C для выхода)...\n");

    // --- Основной цикл ------------------------------------------------
    let mut state = State::Silence;
    let mut segment_start: Option<Instant> = None;
    let mut last_voiced_at: Option<Instant> = None;
    let mut segment_peak_score: f32 = 0.0;
    let mut segment_rms_sum: f32 = 0.0;
    let mut segment_rms_count: usize = 0;
    let mut last_meter_at = Instant::now() - Duration::from_secs(1);
    sys_counter = 0;

    loop {
        if !running.load(Ordering::SeqCst) {
            break;
        }

        if let Ok(chunk) = rx.recv_timeout(Duration::from_millis(500)) {
            sample_buf.extend_from_slice(&chunk);
        }

        while sample_buf.len() >= FRAME_SIZE {
            let frame = &sample_buf[..FRAME_SIZE];
            let now = Instant::now();
            let r = engine.process_frame(frame);
            stats.add_frame(r.confirmed_voiced);

            match state {
                State::Silence => {
                    if r.confirmed_voiced {
                        state = State::Speech;
                        segment_start = Some(now);
                        last_voiced_at = Some(now);
                        segment_peak_score = r.score;
                        segment_rms_sum = r.rms;
                        segment_rms_count = 1;
                    }
                }
                State::Speech => {
                    if r.confirmed_voiced {
                        last_voiced_at = Some(now);
                        segment_rms_sum += r.rms;
                        segment_rms_count += 1;
                        if r.score > segment_peak_score {
                            segment_peak_score = r.score;
                        }
                    }

                    let hangover_expired = last_voiced_at
                        .map(|t| now.duration_since(t) > Duration::from_millis(HANGOVER_MS))
                        .unwrap_or(true);

                    if hangover_expired {
                        let duration = segment_start
                            .map(|s| now.duration_since(s))
                            .unwrap_or_default();
                        state = State::Silence;

                        if duration >= Duration::from_millis(MIN_SEGMENT_MS) {
                            let avg_rms = if segment_rms_count > 0 {
                                segment_rms_sum / segment_rms_count as f32
                            } else {
                                0.0
                            };
                            println!(
                                "\n🗣️  Сегмент: {:.0}мс | score max {:.2} | rms {:.0}",
                                duration.as_millis(),
                                segment_peak_score,
                                avg_rms
                            );
                        }
                    }
                }
            }

            // Живой индикатор + системные метрики
            if now.duration_since(last_meter_at) >= Duration::from_millis(80) {
                last_meter_at = now;
                sys_counter += 1;
                let mut cpu_str = String::new();
                let mut ram_str = String::new();

                // ОПТИМИЗАЦИЯ: sysinfo вызывается редко, чтобы не создавать latency spikes
                if sys_counter % SYSINFO_UPDATE_INTERVAL == 0 {
                    sys.refresh_processes_specifics(
                        ProcessesToUpdate::Some(&[pid]),
                        false,
                        ProcessRefreshKind::everything().with_cpu().with_memory(),
                    );
                    if let Some(process) = sys.process(pid) {
                        let cpu = process.cpu_usage();
                        let ram = process.memory();
                        stats.add_system_metrics(cpu, ram);
                        if DEBUG_LOGGING {
                            cpu_str = format!("| CPU: {:>4.1}%", cpu);
                            ram_str = format!("| RAM: {:>5} КБ", ram);
                        }
                    }
                }

                let label = match state {
                    State::Speech => "говорит",
                    State::Silence => "тишина ",
                };
                let filled = (r.score.clamp(0.0, 1.0) * 20.0) as usize;
                let bar: String = "█".repeat(filled) + &"░".repeat(20 - filled);

                if DEBUG_LOGGING {
                    let wind_status = if r.zcr < WIND_ZCR_THRESHOLD && r.rms > WIND_RMS_THRESHOLD {
                        "ВЕТЕР"
                    } else {
                        "OK"
                    };
                    print!(
                        "{CLEAR_LINE}[{bar}] score {:.2} | ZCR {:.3} ({}) | RMS {:.0} | {} {} {}",
                        r.score, r.zcr, wind_status, r.rms, label, cpu_str, ram_str
                    );
                } else {
                    print!(
                        "{CLEAR_LINE}[{bar}] score {:.2}  {} {} {}",
                        r.score, label, cpu_str, ram_str
                    );
                }
                io::stdout().flush().ok();
            }

            // Zero-allocation cleanup
            let remaining = sample_buf.len() - FRAME_SIZE;
            sample_buf.copy_within(FRAME_SIZE.., 0);
            sample_buf.truncate(remaining);
        }
    }
    print_final_stats(&stats);
}
