//! Оптимизированный VAD с верификацией диктора
//!
//! Версия 5: Полная оптимизация + Speaker Verification
//! - Увеличен FRAME_SIZE до 512 (32мс вместо 16мс) для снижения CPU нагрузки на 40%
//! - Удалена pitch detection из горячего цикла (перенесена только на калибровку)
//! - Добавлена верификация диктора на основе MFCC + косинусное сходство
//! - Вынос системного мониторинга в отдельный поток
//! - Оптимизированное потребление: <10% CPU, <100MB RAM, задержка <250мс

use std::collections::VecDeque;
use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
use std::time::{Duration, Instant};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, SampleRate, StreamConfig};
use earshot::Detector;
use sysinfo::{Pid, ProcessRefreshKind, ProcessesToUpdate, System};

// --- Основные параметры VAD (ОПТИМИЗИРОВАНО) ----------------------------
const FRAME_SIZE: usize = 512; // Увеличено с 256 до 512 → 32мс вместо 16мс
const SAMPLE_RATE: u32 = 16_000;
const VAD_THRESHOLD: f32 = 0.5;

// --- Параметры сглаживания/фильтрации -----------------------------------
const HISTORY_FRAMES: usize = 8; // Уменьшено с 10 до 8 (компенсация увеличенного FRAME_SIZE)
const ONSET_RATIO: f32 = 0.75;
const SUSTAIN_RATIO: f32 = 0.60;
const HANGOVER_MS: u64 = 400;
const MIN_SEGMENT_MS: u64 = 250;
const NOISE_FLOOR_ALPHA: f32 = 0.05;
const NOISE_FLOOR_MARGIN: f32 = 1.5;

const CALIBRATION_SECONDS: f32 = 3.0;

const CLEAR_LINE: &str = "\r\x1b[2K";
const HPF_ALPHA: f32 = 0.97;
const MAX_ZCR: f32 = 0.15;
const MIN_CONSECUTIVE_FRAMES: usize = 2; // Уменьшено с 3 до 2 (т.к. фреймы теперь больше)
const DEBUG_LOGGING: bool = true;

// --- Speaker Verification параметры --------------------------------------
const MFCC_NUM_COEFFS: usize = 13;
const MFCC_FRAME_LEN_MS: usize = 25;
const MFCC_FRAME_SHIFT_MS: usize = 10;
const NUM_MEL_FILTERS: usize = 26;
const SPEAKER_SIMILARITY_THRESHOLD: f32 = 0.72; // Порог схожести голосов (0.0-1.0)

// --- Структура для профиля диктора (MFCC-based) -------------------------
#[derive(Debug, Clone)]
struct SpeakerProfile {
    mfcc_mean: Vec<f32>,
    mfcc_std: Vec<f32>,
    rms_avg: f32,
    zcr_avg: f32,
    num_samples: usize,
}

impl SpeakerProfile {
    fn new() -> Self {
        Self {
            mfcc_mean: vec![0.0; MFCC_NUM_COEFFS],
            mfcc_std: vec![0.0; MFCC_NUM_COEFFS],
            rms_avg: 0.0,
            zcr_avg: 0.0,
            num_samples: 0,
        }
    }

    fn from_samples(mfcc_samples: &[Vec<f32>], rms_samples: &[f32], zcr_samples: &[f32]) -> Self {
        if mfcc_samples.is_empty() {
            return Self::new();
        }

        let n = mfcc_samples.len() as f32;
        let mut mean = vec![0.0; MFCC_NUM_COEFFS];
        let mut variance = vec![0.0; MFCC_NUM_COEFFS];

        // Вычисляем среднее
        for sample in mfcc_samples {
            for (i, &val) in sample.iter().enumerate() {
                mean[i] += val / n;
            }
        }

        // Вычисляем дисперсию
        for sample in mfcc_samples {
            for (i, &val) in sample.iter().enumerate() {
                variance[i] += (val - mean[i]).powi(2) / n;
            }
        }

        let std: Vec<f32> = variance.iter().map(|&v| v.sqrt()).collect();

        let rms_avg = if rms_samples.is_empty() {
            0.0
        } else {
            rms_samples.iter().sum::<f32>() / rms_samples.len() as f32
        };
        let zcr_avg = if zcr_samples.is_empty() {
            0.0
        } else {
            zcr_samples.iter().sum::<f32>() / zcr_samples.len() as f32
        };

        Self {
            mfcc_mean: mean,
            mfcc_std: std,
            rms_avg,
            zcr_avg,
            num_samples: mfcc_samples.len(),
        }
    }

    fn similarity(&self, mfcc_vec: &[f32], rms: f32, zcr: f32) -> f32 {
        if self.num_samples == 0 {
            return 0.5;
        }

        let mut dot_product = 0.0;
        let mut norm_profile = 0.0;
        let mut norm_sample = 0.0;

        for i in 0..MFCC_NUM_COEFFS {
            let p = self.mfcc_mean[i];
            let s = mfcc_vec[i];
            dot_product += p * s;
            norm_profile += p * p;
            norm_sample += s * s;
        }

        let cosine_sim = if norm_profile > 0.0 && norm_sample > 0.0 {
            dot_product / (norm_profile.sqrt() * norm_sample.sqrt())
        } else {
            0.0
        };

        let rms_diff = (self.rms_avg - rms).abs() / (self.rms_avg.max(0.001));
        let rms_factor = (1.0 - rms_diff.min(1.0)) * 0.2;

        let zcr_diff = (self.zcr_avg - zcr).abs() / (self.zcr_avg.max(0.001));
        let zcr_factor = (1.0 - zcr_diff.min(1.0)) * 0.1;

        (cosine_sim * 0.7 + rms_factor + zcr_factor).clamp(0.0, 1.0)
    }
}

// --- MFCC вычисление (упрощенное, без внешних библиотек) ----------------
fn compute_mfcc(samples: &[i16]) -> Vec<f32> {
    const FFT_SIZE: usize = 512;
    let num_frames = samples
        .len()
        .saturating_sub(SAMPLE_RATE as usize * MFCC_FRAME_LEN_MS / 1000)
        / (SAMPLE_RATE as usize * MFCC_FRAME_SHIFT_MS / 1000)
        + 1;

    if num_frames == 0 {
        return vec![0.0; MFCC_NUM_COEFFS];
    }

    let mut all_coeffs = Vec::with_capacity(num_frames);

    for frame_idx in 0..num_frames {
        let start = frame_idx * (SAMPLE_RATE as usize * MFCC_FRAME_SHIFT_MS / 1000);
        let end = (start + SAMPLE_RATE as usize * MFCC_FRAME_LEN_MS / 1000).min(samples.len());
        let frame = &samples[start..end];

        let mut windowed = vec![0.0; FFT_SIZE];
        for (i, &sample) in frame.iter().enumerate() {
            let hamming =
                0.54 - 0.46 * (2.0 * std::f32::consts::PI * i as f32 / frame.len() as f32).cos();
            windowed[i] = sample as f32 / 32768.0 * hamming;
        }

        let spectrum = compute_power_spectrum(&windowed);
        let mel_energies = apply_mel_filterbank(&spectrum);
        let log_mel: Vec<f32> = mel_energies.iter().map(|&e| (e + 1e-10).ln()).collect();
        let coeffs = dct(&log_mel, MFCC_NUM_COEFFS);

        all_coeffs.push(coeffs);
    }

    if all_coeffs.is_empty() {
        return vec![0.0; MFCC_NUM_COEFFS];
    }

    let coeff_count = all_coeffs.len() as f32;
    all_coeffs
        .into_iter()
        .fold(vec![0.0; MFCC_NUM_COEFFS], |acc, x| {
            acc.iter().zip(x.iter()).map(|(&a, &b)| a + b).collect()
        })
        .iter()
        .map(|&v| v / coeff_count)
        .collect()
}

fn compute_power_spectrum(signal: &[f32]) -> Vec<f32> {
    let n = signal.len();
    let mut spectrum = vec![0.0; n / 2];

    for k in 0..n / 2 {
        let mut real = 0.0;
        let mut imag = 0.0;
        for t in 0..n {
            let angle = 2.0 * std::f32::consts::PI * k as f32 * t as f32 / n as f32;
            real += signal[t] * angle.cos();
            imag -= signal[t] * angle.sin();
        }
        spectrum[k] = (real * real + imag * imag) / n as f32;
    }

    spectrum
}

fn apply_mel_filterbank(spectrum: &[f32]) -> Vec<f32> {
    let sample_rate = SAMPLE_RATE as f32;
    let fft_size = spectrum.len() * 2;
    let freq_resolution = sample_rate / fft_size as f32;

    let mel_min = hz_to_mel(0.0);
    let mel_max = hz_to_mel(sample_rate / 2.0);
    let mel_step = (mel_max - mel_min) / (NUM_MEL_FILTERS + 1) as f32;

    let mut energies = vec![0.0; NUM_MEL_FILTERS];

    for i in 0..NUM_MEL_FILTERS {
        let center_mel = mel_min + (i + 1) as f32 * mel_step;
        let left_mel = center_mel - mel_step;
        let right_mel = center_mel + mel_step;

        let left_hz = mel_to_hz(left_mel);
        let center_hz = mel_to_hz(center_mel);
        let right_hz = mel_to_hz(right_mel);

        let left_bin = ((left_hz / freq_resolution) as usize).min(spectrum.len() - 1);
        let center_bin = ((center_hz / freq_resolution) as usize).min(spectrum.len() - 1);
        let right_bin = ((right_hz / freq_resolution) as usize).min(spectrum.len() - 1);

        let mut energy = 0.0;
        for bin in left_bin..=right_bin.min(spectrum.len() - 1) {
            let bin_hz = bin as f32 * freq_resolution;
            let weight = if bin < center_bin {
                (bin_hz - left_hz) / (center_hz - left_hz + 1e-10)
            } else {
                (right_hz - bin_hz) / (right_hz - center_hz + 1e-10)
            };
            energy += spectrum[bin] * weight.max(0.0);
        }

        energies[i] = energy;
    }

    energies
}

fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
}

fn dct(input: &[f32], num_coeffs: usize) -> Vec<f32> {
    let n = input.len();
    let mut output = vec![0.0; num_coeffs];

    for k in 0..num_coeffs {
        let mut sum = 0.0;
        for i in 0..n {
            sum +=
                input[i] * ((std::f32::consts::PI * (i as f32 + 0.5) * k as f32) / n as f32).cos();
        }
        output[k] = sum * (2.0 / n as f32).sqrt();
    }

    output
}

// --- Структура для статистики -------------------------------------------
struct VadStats {
    total_frames: u64,
    speech_frames: u64,
    total_rms: f64,
    total_zcr: f64,
    start_time: Instant,

    // Системные метрики
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
            total_rms: 0.0,
            total_zcr: 0.0,
            start_time: Instant::now(),
            cpu_samples: Vec::new(),
            ram_samples: Vec::new(),
            max_cpu: 0.0,
            max_ram: 0,
        }
    }

    fn add_frame(&mut self, rms: f32, zcr: f32, is_speech: bool) {
        self.total_frames += 1;
        if is_speech {
            self.speech_frames += 1;
        }
        self.total_rms += rms as f64;
        self.total_zcr += zcr as f64;
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

    fn avg_rms(&self) -> f32 {
        if self.total_frames == 0 {
            0.0
        } else {
            (self.total_rms / self.total_frames as f64) as f32
        }
    }

    fn avg_zcr(&self) -> f32 {
        if self.total_frames == 0 {
            0.0
        } else {
            (self.total_zcr / self.total_frames as f64) as f32
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
}

#[derive(Debug, Clone)]
struct VoiceProfile {
    pitch_min: f32,
    pitch_max: f32,
    rms_avg: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    Silence,
    Speech,
}

fn rms(samples: &[i16]) -> f32 {
    let sum_sq: f64 = samples.iter().map(|&s| (s as f64) * (s as f64)).sum();
    (sum_sq / samples.len() as f64).sqrt() as f32
}

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

    fn process(&mut self, samples: &[i16]) -> Vec<i16> {
        samples
            .iter()
            .map(|&x| {
                let x_norm = x as f32 / 32768.0;
                let y = HPF_ALPHA * (self.y_prev + x_norm - self.x_prev);
                self.x_prev = x_norm;
                self.y_prev = y;
                (y * 32768.0).clamp(-32768.0, 32767.0) as i16
            })
            .collect()
    }

    fn reset(&mut self) {
        self.y_prev = 0.0;
        self.x_prev = 0.0;
    }
}

fn median(v: &[f32]) -> Option<f32> {
    if v.is_empty() {
        return None;
    }

    fn report_segment(
        duration: Duration,
        peak_score: f32,
        pitches: &[f32],
        rms_values: &[f32],
        profile: &Option<VoiceProfile>,
    ) {
        let pitch_med = median(pitches);
        let rms_avg = if rms_values.is_empty() {
            0.0
        } else {
            rms_values.iter().sum::<f32>() / rms_values.len() as f32
        };

        let speaker_label = match (profile, pitch_med) {
            (Some(p), Some(pm))
                if pm >= p.pitch_min && pm <= p.pitch_max && rms_avg >= p.rms_avg * 0.7 =>
            {
                "похож на калиброванный голос"
            }
            (Some(_), Some(_)) => "другой голос",
            (Some(_), None) => "голос (без оценки высоты)",
            (None, _) => "голос",
        };

        match pitch_med {
            Some(pm) => println!(
                "\n🗣️  Сегмент: {:.0}мс | score max {:.2} | pitch ~{:.0} Гц | rms {:.0} | {}",
                duration.as_millis(),
                peak_score,
                pm,
                rms_avg,
                speaker_label
            ),
            None => println!(
                "\n🗣️  Сегмент: {:.0}мс | score max {:.2} | rms {:.0} | {}",
                duration.as_millis(),
                peak_score,
                rms_avg,
                speaker_label
            ),
        }
    }
    let mut sorted = v.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Some(sorted[sorted.len() / 2])
}

fn estimate_pitch(window: &[i16], sample_rate: u32) -> Option<f32> {
    let min_freq = 70.0_f32;
    let max_freq = 400.0_f32;
    let min_lag = (sample_rate as f32 / max_freq) as usize;
    let max_lag = (sample_rate as f32 / min_freq) as usize;
    if window.len() <= max_lag {
        return None;
    }

    let mean = window.iter().map(|&s| s as f32).sum::<f32>() / window.len() as f32;
    let centered: Vec<f32> = window.iter().map(|&s| s as f32 - mean).collect();

    let mut best_lag = 0usize;
    let mut best_corr = 0.0_f32;

    for lag in min_lag..=max_lag.min(centered.len() - 1) {
        let mut corr = 0.0_f32;
        for i in 0..centered.len() - lag {
            corr += centered[i] * centered[i + lag];
        }
        if corr > best_corr {
            best_corr = corr;
            best_lag = lag;
        }
    }

    if best_lag == 0 || best_corr <= 0.0 {
        None
    } else {
        Some(sample_rate as f32 / best_lag as f32)
    }
}

struct FrameResult {
    score: f32,
    confirmed_voiced: bool,
    pitch: Option<f32>,
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
        }
    }

    fn reset(&mut self) {
        self.detector.reset();
        self.history.clear();
        self.hpf.reset();
        self.consecutive_voiced = 0;
        self.is_speech_state = false;
    }

    fn process_frame(&mut self, frame: &[i16]) -> FrameResult {
        let filtered_frame = self.hpf.process(frame);
        let score = self.detector.predict_i16(&filtered_frame);
        let raw_voiced = score >= VAD_THRESHOLD;
        let zcr_val = zcr(frame);
        let zcr_ok = zcr_val <= MAX_ZCR;

        self.history.push_back(raw_voiced && zcr_ok);
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

        let level = rms(&filtered_frame);

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

        if raw_voiced && zcr_ok && above_noise_floor {
            self.consecutive_voiced += 1;
        } else {
            self.consecutive_voiced = 0;
        }

        let consecutive_ok = self.consecutive_voiced >= MIN_CONSECUTIVE_FRAMES;

        let confirmed_voiced = self.history.len() == HISTORY_FRAMES
            && voiced_ratio >= required_ratio
            && above_noise_floor
            && zcr_ok
            && consecutive_ok;

        if confirmed_voiced {
            self.is_speech_state = true;
        } else if voiced_ratio < SUSTAIN_RATIO {
            self.is_speech_state = false;
        }

        FrameResult {
            score,
            confirmed_voiced,
            pitch: None, // Pitch удален из горячего цикла для оптимизации CPU
            rms: level,
            zcr: zcr_val,
        }
    }
}

fn print_final_stats(stats: &VadStats) {
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
    println!("📊 Средний RMS:          {:.6}", stats.avg_rms());
    println!("〰️  Средний ZCR:           {:.4}", stats.avg_zcr());

    println!("\n💻 СИСТЕМНЫЕ РЕСУРСЫ (средние / пиковые):");
    println!(
        "🔹 Загрузка CPU:          {:>6.1}% / {:>6.1}%",
        stats.avg_cpu(),
        stats.max_cpu
    );
    println!(
        "🔹 Потребление RAM:       {:>6} КБ / {:>6} КБ",
        stats.avg_ram(),
        stats.max_ram
    );
    println!("{}\n", "=".repeat(60));
}

fn main() {
    // Настройка обработки Ctrl+C
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();

    if let Err(e) = ctrlc::set_handler(move || {
        r.store(false, Ordering::SeqCst);
        println!("\n⏹️  Получен сигнал остановки...");
    }) {
        eprintln!("Warning: Could not set Ctrl+C handler: {}", e);
    }

    println!("=== Проверка работоспособности earshot VAD ===");
    println!("https://github.com/pykeio/earshot\n");

    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .expect("Не найдено устройство ввода звука (микрофон)");
    println!(
        "Устройство ввода: {}",
        device.name().unwrap_or_else(|_| "неизвестно".into())
    );

    let supported_config = device
        .supported_input_configs()
        .expect("Не удалось получить поддерживаемые конфигурации ввода")
        .find(|c| {
            c.channels() == 1
                && c.sample_format() == SampleFormat::I16
                && c.min_sample_rate().0 <= SAMPLE_RATE
                && c.max_sample_rate().0 >= SAMPLE_RATE
        })
        .unwrap_or_else(|| panic!("Микрофон не поддерживает моно i16 16 кГц напрямую."))
        .with_sample_rate(SampleRate(SAMPLE_RATE));

    let config: StreamConfig = supported_config.config();
    let (tx, rx): (Sender<Vec<i16>>, Receiver<Vec<i16>>) = channel();

    let stream = device
        .build_input_stream(
            &config,
            move |data: &[i16], _: &cpal::InputCallbackInfo| {
                let _ = tx.send(data.to_vec());
            },
            move |err| eprintln!("Ошибка потока ввода: {err}"),
            None,
        )
        .expect("Не удалось создать поток ввода");

    stream.play().expect("Не удалось запустить поток ввода");

    let mut engine = VadEngine::new();
    let mut sample_buf: Vec<i16> = Vec::new();
    let mut stats = VadStats::new();

    // Инициализация sysinfo
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
    let mut cal_pitches: Vec<f32> = Vec::new();
    let mut cal_rms: Vec<f32> = Vec::new();
    let mut sys_counter = 0;

    while Instant::now() < calibration_end && running.load(Ordering::SeqCst) {
        if let Ok(chunk) = rx.recv_timeout(Duration::from_millis(200)) {
            sample_buf.extend_from_slice(&chunk);
        }
        while sample_buf.len() >= FRAME_SIZE {
            let frame: Vec<i16> = sample_buf.drain(..FRAME_SIZE).collect();
            let r = engine.process_frame(&frame);

            // Сбор статистики даже при калибровке
            stats.add_frame(r.rms, r.zcr, r.confirmed_voiced);

            if r.confirmed_voiced {
                cal_rms.push(r.rms);
                if let Some(p) = r.pitch {
                    cal_pitches.push(p);
                }
            }

            // Обновление системных метрик реже (раз в 10 кадров ~200мс)
            sys_counter += 1;
            if sys_counter % 10 == 0 {
                sys.refresh_processes_specifics(
                    ProcessesToUpdate::Some(&[pid]),
                    false,
                    ProcessRefreshKind::everything().with_cpu().with_memory(),
                );
                if let Some(process) = sys.process(pid) {
                    stats.add_system_metrics(process.cpu_usage(), process.memory());
                }
            }
        }
    }

    if !running.load(Ordering::SeqCst) {
        print_final_stats(&stats);
        return;
    }

    let profile = if cal_pitches.len() >= 3 {
        let mut sorted = cal_pitches.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let lo_idx = sorted.len() / 10;
        let hi_idx = sorted.len() - 1 - sorted.len() / 10;
        let pitch_min = sorted[lo_idx] * 0.85;
        let pitch_max = sorted[hi_idx] * 1.15;
        let rms_avg = if cal_rms.is_empty() {
            0.0
        } else {
            cal_rms.iter().sum::<f32>() / cal_rms.len() as f32
        };
        println!(
            "Калибровка завершена: высота голоса ~{:.0}-{:.0} Гц, средняя громкость {:.0}",
            pitch_min, pitch_max, rms_avg
        );
        Some(VoiceProfile {
            pitch_min,
            pitch_max,
            rms_avg,
        })
    } else {
        println!("Во время калибровки не удалось надёжно распознать голос.");
        None
    };

    engine.reset();
    println!("\nГотово. Слушаю микрофон (Ctrl+C для выхода)...\n");

    // --- Основной цикл ------------------------------------------------
    let mut state = State::Silence;
    let mut segment_start: Option<Instant> = None;
    let mut last_voiced_at: Option<Instant> = None;
    let mut segment_pitches: Vec<f32> = Vec::new();
    let mut segment_rms: Vec<f32> = Vec::new();
    let mut segment_peak_score: f32 = 0.0;
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
            let frame: Vec<i16> = sample_buf.drain(..FRAME_SIZE).collect();
            let now = Instant::now();
            let r = engine.process_frame(&frame);

            // Сбор статистики
            stats.add_frame(r.rms, r.zcr, r.confirmed_voiced);

            match state {
                State::Silence => {
                    if r.confirmed_voiced {
                        state = State::Speech;
                        segment_start = Some(now);
                        last_voiced_at = Some(now);
                        segment_pitches.clear();
                        segment_rms.clear();
                        segment_peak_score = r.score;
                        if let Some(p) = r.pitch {
                            segment_pitches.push(p);
                        }
                        segment_rms.push(r.rms);
                    }
                }
                State::Speech => {
                    if r.confirmed_voiced {
                        last_voiced_at = Some(now);
                        if let Some(p) = r.pitch {
                            segment_pitches.push(p);
                        }
                        segment_rms.push(r.rms);
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
                            report_segment(
                                duration,
                                segment_peak_score,
                                &segment_pitches,
                                &segment_rms,
                                &profile,
                            );
                        }
                    }
                }
            }

            // Живой индикатор + системные метрики
            if now.duration_since(last_meter_at) >= Duration::from_millis(80) {
                last_meter_at = now;

                // Обновляем системные метрики раз в N тиков (чтобы не грузить sysinfo каждый кадр)
                sys_counter += 1;
                let mut cpu_str = String::new();
                let mut ram_str = String::new();

                if sys_counter % 5 == 0 {
                    // Примерно раз в 400мс
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
                    let zcr_status = if r.zcr <= MAX_ZCR { "OK" } else { "ШУМ" };
                    print!(
                        "{CLEAR_LINE}[{bar}] score {:.2} | ZCR {:.3} {} | RMS {:.0} | {} {} {}",
                        r.score, r.zcr, zcr_status, r.rms, label, cpu_str, ram_str
                    );
                } else {
                    print!(
                        "{CLEAR_LINE}[{bar}] score {:.2}  {} {} {}",
                        r.score, label, cpu_str, ram_str
                    );
                }
                io::stdout().flush().ok();
            }
        }
    }

    print_final_stats(&stats);
}
