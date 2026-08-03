use earshot::Detector;
use std::collections::VecDeque;

pub const FRAME_SIZE: usize = 256; // 16 мс при 16 kHz
pub const SAMPLE_RATE: u32 = 16_000;
pub const FRAME_MS: f64 = FRAME_SIZE as f64 * 1000.0 / SAMPLE_RATE as f64;

pub const VAD_THRESHOLD: f32 = 0.5;

pub const HISTORY_FRAMES: usize = 10;
pub const ONSET_RATIO: f32 = 0.6;
pub const SUSTAIN_RATIO: f32 = 0.4;

pub const HANGOVER_MS: u64 = 300;
pub const MIN_SEGMENT_MS: u64 = 200;

const NOISE_FLOOR_ALPHA: f32 = 0.05;
const NOISE_FLOOR_MARGIN: f32 = 3.0;

const HPF_ALPHA: f32 = 0.92;

const WIND_ZCR_THRESHOLD: f32 = 0.10;
const WIND_RMS_THRESHOLD: f32 = 800.0;

const MIN_CONSECUTIVE_FRAMES: usize = 3;

#[derive(Debug, Clone)]
pub struct FrameResult {
    pub score: f32,
    pub confirmed_voiced: bool,
    pub rms: f32,
    pub zcr: f32,
}

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

    fn process(&mut self, samples: &[i16], output_buf: &mut [f32]) {
        let n = samples.len().min(output_buf.len());

        for i in 0..n {
            let x_norm = samples[i] as f32 / 32768.0;
            let y = HPF_ALPHA * (self.y_prev + x_norm - self.x_prev);
            self.x_prev = x_norm;
            self.y_prev = y;
            output_buf[i] = y;
        }

        for y in output_buf.iter_mut().skip(n) {
            *y = 0.0;
        }
    }

    fn reset(&mut self) {
        self.y_prev = 0.0;
        self.x_prev = 0.0;
    }
}

pub struct VadEngine {
    detector: Detector,
    history: VecDeque<bool>,
    noise_floor: f32,
    noise_floor_initialized: bool,
    hpf: HighPassFilter,
    consecutive_voiced: usize,
    is_speech_state: bool,
    hpf_buf: Vec<f32>,
}

impl Default for VadEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl VadEngine {
    pub fn new() -> Self {
        Self {
            detector: Detector::default(),
            history: VecDeque::with_capacity(HISTORY_FRAMES),
            noise_floor: 0.0,
            noise_floor_initialized: false,
            hpf: HighPassFilter::new(),
            consecutive_voiced: 0,
            is_speech_state: false,
            hpf_buf: vec![0.0; FRAME_SIZE],
        }
    }

    pub fn reset(&mut self) {
        self.detector.reset();
        self.history.clear();
        self.hpf.reset();
        self.consecutive_voiced = 0;
        self.is_speech_state = false;
        self.noise_floor_initialized = false;
    }

    pub fn process_frame(&mut self, frame: &[i16]) -> FrameResult {
        debug_assert_eq!(frame.len(), FRAME_SIZE);

        self.hpf.process(frame, &mut self.hpf_buf);

        let filtered_i16: Vec<i16> = self
            .hpf_buf
            .iter()
            .map(|&y| (y * 32768.0).clamp(-32768.0, 32767.0) as i16)
            .collect();

        let score = self.detector.predict_i16(&filtered_i16);
        let raw_voiced = score >= VAD_THRESHOLD;

        let zcr_val = zcr(frame);
        let level = rms(frame);

        let is_wind = zcr_val < WIND_ZCR_THRESHOLD && level > WIND_RMS_THRESHOLD;
        let zcr_ok = zcr_val <= 0.25;
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

/// Offline-обработка WAV-файла.
/// Возвращает по одному bool-значению на фрейм.
pub fn process_offline(samples: &[i16]) -> Vec<bool> {
    let mut engine = VadEngine::new();
    let mut result = Vec::with_capacity((samples.len() + FRAME_SIZE - 1) / FRAME_SIZE);

    for chunk in samples.chunks(FRAME_SIZE) {
        if chunk.len() == FRAME_SIZE {
            let frame_result = engine.process_frame(chunk);
            result.push(frame_result.confirmed_voiced);
        } else {
            let mut padded = vec![0i16; FRAME_SIZE];
            padded[..chunk.len()].copy_from_slice(chunk);

            let frame_result = engine.process_frame(&padded);
            result.push(frame_result.confirmed_voiced);
        }
    }

    result
}
