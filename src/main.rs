//! Небольшой проект для проверки работоспособности VAD-библиотеки earshot
//! (https://github.com/pykeio/earshot).
//!
//! Что делает:
//! 1. Слушает микрофон и прогоняет аудио через earshot::Detector.
//! 2. В первые несколько секунд проводит "калибровку": строит грубый
//!    голосовой профиль (диапазон высоты тона + громкость) человека,
//!    который заговорил первым.
//! 3. Дальше в реальном времени выделяет реплики (сегменты речи) и для
//!    каждой целиком - а не по отдельным 16-мс кадрам - решает, похожа ли
//!    она на калиброванный голос.
//!
//! Версия 3: Улучшенная фильтрация шумов (Вариант A):
//! - High-Pass Filter (80 Гц) для отсечения ветра и низкочастотного гула
//! - Zero Crossing Rate (ZCR) для фильтрации белого шума
//! - Гистерезис порогов (onset/sustain) для стабильности
//! - Требование N кадров подряд для срабатывания
//! - Отладочное логирование параметров
//!
//! ВАЖНО: earshot - это исключительно детектор голосовой активности (VAD),
//! он НЕ делает идентификацию/верификацию диктора. "Подстройка под голос"
//! здесь - простая эвристика поверх VAD (высота тона + громкость), а не
//! настоящее распознавание диктора.

use std::collections::VecDeque;
use std::io::{self, Write};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::time::{Duration, Instant};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, SampleRate, StreamConfig};
use earshot::Detector;

// --- Основные параметры VAD ---------------------------------------------

/// earshot требует кадры ровно по 256 сэмплов (16 мс) при 16 кГц.
const FRAME_SIZE: usize = 256;
const SAMPLE_RATE: u32 = 16_000;
/// Порог по умолчанию, рекомендованный документацией earshot.
const VAD_THRESHOLD: f32 = 0.5;

// --- Параметры сглаживания/фильтрации (УЛУЧШЕННАЯ ВЕРСИЯ) ---------------

/// Сколько последних кадров учитывать при решении "точно ли началась речь".
/// 10 кадров по 16 мс = ~160 мс контекста (увеличено с 6 для стабильности).
const HISTORY_FRAMES: usize = 10;
/// Доля кадров в этом окне для ВКЛЮЧЕНИЯ (onset) - строже чем раньше.
const ONSET_RATIO: f32 = 0.75;
/// Доля кадров в этом окне для УДЕРЖАНИЯ (sustain) - мягче, чтобы речь не прерывалась.
const SUSTAIN_RATIO: f32 = 0.60;
/// Сколько миллисекунд тишины подряд нужно, чтобы закрыть реплику.
const HANGOVER_MS: u64 = 400;
/// Реплики короче этого считаются шумом/щелчком и отбрасываются целиком.
const MIN_SEGMENT_MS: u64 = 250; // увеличено с 200
/// Скорость адаптации фонового шумового уровня (эксп. скользящее среднее).
const NOISE_FLOOR_ALPHA: f32 = 0.05;
/// Во сколько раз кадр должен быть громче шумового пола.
const NOISE_FLOOR_MARGIN: f32 = 1.5; // увеличено с 1.3

/// Окно для оценки высоты тона автокорреляцией.
const PITCH_WINDOW_LEN: usize = 1024;
/// Сколько секунд слушать при калибровке.
const CALIBRATION_SECONDS: f32 = 3.0;

/// ANSI: очистить текущую строку терминала.
const CLEAR_LINE: &str = "\r\x1b[2K";

/// Коэффициент для HPF (High-Pass Filter) - частота среза ~80 Гц.
/// Формула: alpha = rc / (rc + dt), где rc = 1/(2*pi*fc), dt = 1/sample_rate
/// Для fc=80 Гц при 16 кГц: alpha ≈ 0.97
const HPF_ALPHA: f32 = 0.97;

/// Максимальный ZCR для голоса (при 16 кГц). Выше - скорее всего шум.
/// Голос: ~100-500 пересечений/сек, ветер/шум: >1000.
const MAX_ZCR: f32 = 0.15; // доля от частоты дискретизации (0.15 * 16000 = 2400)

/// Минимальное количество кадров подряд со score > порога для срабатывания.
const MIN_CONSECUTIVE_FRAMES: usize = 3;

/// Включить отладочное логирование (ZCR, RMS, статус).
const DEBUG_LOGGING: bool = true;

/// Грубый "голосовой профиль" человека, инициализировавшего диалог.
#[derive(Debug, Clone)]
struct VoiceProfile {
    pitch_min: f32,
    pitch_max: f32,
    rms_avg: f32,
}

fn rms(samples: &[i16]) -> f32 {
    let sum_sq: f64 = samples.iter().map(|&s| (s as f64) * (s as f64)).sum();
    (sum_sq / samples.len() as f64).sqrt() as f32
}

/// Вычисляет Zero Crossing Rate (доля пересечений нуля).
/// Для голоса типично 0.01-0.05, для шума/ветра > 0.1.
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

/// Простейший IIR high-pass фильтр первого порядка.
/// Отсекает низкие частоты (ветер, гул) ниже ~80 Гц.
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
                let x_norm = x as f32 / 32768.0; // нормализуем к [-1, 1]
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
    let mut sorted = v.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Some(sorted[sorted.len() / 2])
}

/// Простейшая оценка высоты основного тона автокорреляцией во временной
/// области. Годится для демонстрации, не для продакшена.
fn estimate_pitch(window: &[i16], sample_rate: u32) -> Option<f32> {
    let min_freq = 70.0_f32; // нижняя граница (низкий мужской голос)
    let max_freq = 400.0_f32; // верхняя граница (высокий женский/детский голос)
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

/// Результат обработки одного 16-мс кадра.
struct FrameResult {
    score: f32,
    /// Сглаженное, отфильтрованное решение "это точно голос" - в отличие
    /// от сырого `score >= VAD_THRESHOLD`, которое дёргается кадр от кадра.
    confirmed_voiced: bool,
    pitch: Option<f32>,
    rms: f32,
    zcr: f32,
}

/// Обёртка над earshot::Detector, которая добавляет сглаживание onset,
/// оценку высоты тона и адаптивный шумовой пол. Используется одинаково
/// и на калибровке, и в основном цикле - это и даёт стабильность:
/// одна и та же логика фильтрации применяется везде.
struct VadEngine {
    detector: Detector,
    history: VecDeque<bool>,
    pitch_window: Vec<i16>,
    noise_floor: f32,
    noise_floor_initialized: bool,
    hpf: HighPassFilter,
    consecutive_voiced: usize,
    is_speech_state: bool, // для гистерезиса
}

impl VadEngine {
    fn new() -> Self {
        Self {
            detector: Detector::default(),
            history: VecDeque::with_capacity(HISTORY_FRAMES),
            pitch_window: Vec::with_capacity(PITCH_WINDOW_LEN),
            noise_floor: 0.0,
            noise_floor_initialized: false,
            hpf: HighPassFilter::new(),
            consecutive_voiced: 0,
            is_speech_state: false,
        }
    }

    /// Сбрасывает внутреннее состояние - вызывать при смене режима работы
    /// (например, после калибровки перед "боевым" прослушиванием).
    fn reset(&mut self) {
        self.detector.reset();
        self.history.clear();
        self.hpf.reset();
        self.consecutive_voiced = 0;
        self.is_speech_state = false;
    }

    fn process_frame(&mut self, frame: &[i16]) -> FrameResult {
        // Применяем HPF для отсечения низкочастотного шума (ветер)
        let filtered_frame = self.hpf.process(frame);
        
        let score = self.detector.predict_i16(&filtered_frame);
        let raw_voiced = score >= VAD_THRESHOLD;

        // Считаем ZCR до фильтрации (оригинальный сигнал)
        let zcr_val = zcr(frame);
        
        // Проверка ZCR: если слишком высокий - скорее всего шум
        let zcr_ok = zcr_val <= MAX_ZCR;

        self.history.push_back(raw_voiced && zcr_ok);
        if self.history.len() > HISTORY_FRAMES {
            self.history.pop_front();
        }
        
        // Гистерезис: разные пороги для включения и удержания
        let required_ratio = if self.is_speech_state {
            SUSTAIN_RATIO
        } else {
            ONSET_RATIO
        };
        
        let voiced_ratio =
            self.history.iter().filter(|&&v| v).count() as f32 / self.history.len().max(1) as f32;

        self.pitch_window.extend_from_slice(&filtered_frame);
        if self.pitch_window.len() > PITCH_WINDOW_LEN {
            let excess = self.pitch_window.len() - PITCH_WINDOW_LEN;
            self.pitch_window.drain(..excess);
        }
        let pitch = if self.pitch_window.len() >= PITCH_WINDOW_LEN {
            estimate_pitch(&self.pitch_window, SAMPLE_RATE)
        } else {
            None
        };

        let level = rms(&filtered_frame);

        // Пока сырой VAD явно не видит голоса - медленно подстраиваем
        // оценку фонового шума. Помогает не путать ровный гул/шипение
        // на грани порога с настоящим голосом.
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

        // Требуем N кадров подряд для срабатывания
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

        // Обновляем состояние для гистерезиса
        if confirmed_voiced {
            self.is_speech_state = true;
        } else if voiced_ratio < SUSTAIN_RATIO {
            self.is_speech_state = false;
        }

        FrameResult {
            score,
            confirmed_voiced,
            pitch,
            rms: level,
            zcr: zcr_val,
        }
    }
}

#[derive(PartialEq)]
enum State {
    Silence,
    Speech,
}

/// Печатает итог по завершённой реплике: длительность, пиковый score,
/// медианную высоту тона за всю реплику (а не за один кадр - это и убирает
/// прежнее "мигание" между похоже/не похоже) и вердикт по профилю.
fn report_segment(
    duration: Duration,
    peak_score: f32,
    pitches: &[f32],
    levels: &[f32],
    profile: &Option<VoiceProfile>,
) {
    let pitch_median = median(pitches);
    let level_avg = if levels.is_empty() {
        0.0
    } else {
        levels.iter().sum::<f32>() / levels.len() as f32
    };

    let verdict = match (profile, pitch_median) {
        (Some(p), Some(f)) => {
            let pitch_ok = f >= p.pitch_min && f <= p.pitch_max;
            let level_ok = p.rms_avg <= 0.0 || level_avg >= p.rms_avg * 0.2;
            if pitch_ok && level_ok {
                "похоже на калиброванный голос"
            } else {
                "НЕ похоже на калиброванный голос"
            }
        }
        (Some(_), None) => "профиль есть, но тон не определён",
        (None, _) => "профиль не калибровался",
    };

    print!("{CLEAR_LINE}");
    match pitch_median {
        Some(f) => println!(
            "🗣  реплика {:>4.1} сек, пик score {:.2}, тон ~{:>3.0} Гц — {}",
            duration.as_secs_f32(),
            peak_score,
            f,
            verdict
        ),
        None => println!(
            "🗣  реплика {:>4.1} сек, пик score {:.2}, тон не определён — {}",
            duration.as_secs_f32(),
            peak_score,
            verdict
        ),
    }
}

fn main() {
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

    // earshot жёстко требует моно, 16 кГц, i16, кадр 256 сэмплов.
    // Просим у устройства именно такой формат напрямую; если микрофон его
    // не поддерживает - честно останавливаемся, а не тихо портим звук
    // самодельным ресемплингом.
    let supported_config = device
        .supported_input_configs()
        .expect("Не удалось получить поддерживаемые конфигурации ввода")
        .find(|c| {
            c.channels() == 1
                && c.sample_format() == SampleFormat::I16
                && c.min_sample_rate().0 <= SAMPLE_RATE
                && c.max_sample_rate().0 >= SAMPLE_RATE
        })
        .unwrap_or_else(|| {
            panic!(
                "Микрофон не поддерживает моно i16 16 кГц напрямую.\n\
                 Нужен ресемплинг (например, через crate `rubato`) - \
                 в этом демо-проекте он не реализован."
            )
        })
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

    // --- Калибровка ---------------------------------------------------
    println!(
        "\nКалибровка (~{:.0} сек): скажите что-нибудь - под этот голос \
         программа будет подстраиваться...",
        CALIBRATION_SECONDS
    );

    let calibration_end = Instant::now() + Duration::from_secs_f32(CALIBRATION_SECONDS);
    let mut cal_pitches: Vec<f32> = Vec::new();
    let mut cal_rms: Vec<f32> = Vec::new();

    while Instant::now() < calibration_end {
        if let Ok(chunk) = rx.recv_timeout(Duration::from_millis(200)) {
            sample_buf.extend_from_slice(&chunk);
        }
        while sample_buf.len() >= FRAME_SIZE {
            let frame: Vec<i16> = sample_buf.drain(..FRAME_SIZE).collect();
            let r = engine.process_frame(&frame);
            // Используем ту же сглаженную "confirmed_voiced", что и в
            // боевом режиме - так калибровка меньше цепляет случайный шум.
            if r.confirmed_voiced {
                cal_rms.push(r.rms);
                if let Some(p) = r.pitch {
                    cal_pitches.push(p);
                }
            }
        }
    }

    let profile = if cal_pitches.len() >= 3 {
        let mut sorted = cal_pitches.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        // Немного обрезаем края, чтобы один выброс не испортил диапазон.
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
        println!(
            "Во время калибровки не удалось надёжно распознать голос \
             (слишком тихо или слишком коротко)."
        );
        println!("Дальше программа будет только детектировать голос, без сравнения с профилем.");
        None
    };

    // Сбрасываем внутреннее состояние детектора и историю перед "боевым"
    // режимом - калибровка уже прогрела их несколькими секундами аудио.
    engine.reset();

    println!("\nГотово. Слушаю микрофон (Ctrl+C для выхода)...\n");

    // --- Основной цикл: разбиение на реплики (сегменты речи) ------------
    let mut state = State::Silence;
    let mut segment_start: Option<Instant> = None;
    let mut last_voiced_at: Option<Instant> = None;
    let mut segment_pitches: Vec<f32> = Vec::new();
    let mut segment_rms: Vec<f32> = Vec::new();
    let mut segment_peak_score: f32 = 0.0;
    let mut last_meter_at = Instant::now() - Duration::from_secs(1);

    loop {
        if let Ok(chunk) = rx.recv_timeout(Duration::from_millis(500)) {
            sample_buf.extend_from_slice(&chunk);
        }

        while sample_buf.len() >= FRAME_SIZE {
            let frame: Vec<i16> = sample_buf.drain(..FRAME_SIZE).collect();
            let now = Instant::now();
            let r = engine.process_frame(&frame);

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
                        // Короче MIN_SEGMENT_MS - молча отбрасываем как щелчок/шум.
                    }
                }
            }

            // Живой однострочный индикатор с отладочной информацией -
            // не более 12-13 раз в секунду, чтобы не грузить терминал.
            if now.duration_since(last_meter_at) >= Duration::from_millis(80) {
                last_meter_at = now;
                let label = match state {
                    State::Speech => "говорит",
                    State::Silence => "тишина ",
                };
                let filled = (r.score.clamp(0.0, 1.0) * 20.0) as usize;
                let bar: String = "█".repeat(filled) + &"░".repeat(20 - filled);
                
                if DEBUG_LOGGING {
                    // Показываем ZCR и RMS для отладки
                    let zcr_status = if r.zcr <= MAX_ZCR { "OK" } else { "ШУМ" };
                    print!(
                        "{CLEAR_LINE}[{bar}] score {:.2} | ZCR {:.3} {} | RMS {:.0} | {}",
                        r.score, r.zcr, zcr_status, r.rms, label
                    );
                } else {
                    print!("{CLEAR_LINE}[{bar}] score {:.2}  {label}", r.score);
                }
                io::stdout().flush().ok();
            }
        }
    }
}
