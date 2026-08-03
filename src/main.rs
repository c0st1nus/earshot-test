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
//! Версия 2: добавлены сглаживание срабатывания (onset), "hangover"
//! (пауза внутри фразы не рвёт её на части), фильтр коротких щелчков и
//! адаптивный шумовой пол - см. комментарии у констант ниже.
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

// --- Параметры сглаживания/фильтрации (то, что просили доработать) ------

/// Сколько последних кадров учитывать при решении "точно ли началась речь".
/// 6 кадров по 16 мс = ~96 мс контекста.
const HISTORY_FRAMES: usize = 6;
/// Доля кадров в этом окне, которая должна быть "голосовой" по сырому
/// VAD-скору, чтобы считать речь подтверждённой (гасит единичные всплески).
const ONSET_RATIO: f32 = 0.65;
/// Сколько миллисекунд тишины подряд нужно, чтобы закрыть реплику.
/// Не даёт короткой паузе/вдоху внутри фразы разорвать её на части.
const HANGOVER_MS: u64 = 400;
/// Реплики короче этого считаются шумом/щелчком и отбрасываются целиком.
const MIN_SEGMENT_MS: u64 = 200;
/// Скорость адаптации фонового шумового уровня (эксп. скользящее среднее).
const NOISE_FLOOR_ALPHA: f32 = 0.05;
/// Во сколько раз кадр должен быть громче шумового пола, чтобы не
/// считаться просто фоновым гулом на грани VAD-порога.
const NOISE_FLOOR_MARGIN: f32 = 1.3;

/// Окно для оценки высоты тона автокорреляцией (побольше, чем 1 кадр,
/// иначе не поместится даже пара периодов низкого мужского голоса).
const PITCH_WINDOW_LEN: usize = 1024;
/// Сколько секунд слушать при калибровке голоса инициатора диалога.
const CALIBRATION_SECONDS: f32 = 3.0;

/// ANSI: очистить текущую строку терминала и вернуть курсор в начало.
/// Нужно для "живого" однострочного индикатора вместо простыни принтов.
/// Если терминал не поддерживает ANSI (редкие случаи на Windows) - просто
/// будет немного мусора в выводе, на работу это не влияет.
const CLEAR_LINE: &str = "\r\x1b[2K";

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
}

impl VadEngine {
    fn new() -> Self {
        Self {
            detector: Detector::default(),
            history: VecDeque::with_capacity(HISTORY_FRAMES),
            pitch_window: Vec::with_capacity(PITCH_WINDOW_LEN),
            noise_floor: 0.0,
            noise_floor_initialized: false,
        }
    }

    /// Сбрасывает внутреннее состояние - вызывать при смене режима работы
    /// (например, после калибровки перед "боевым" прослушиванием).
    fn reset(&mut self) {
        self.detector.reset();
        self.history.clear();
    }

    fn process_frame(&mut self, frame: &[i16]) -> FrameResult {
        let score = self.detector.predict_i16(frame);
        let raw_voiced = score >= VAD_THRESHOLD;

        self.history.push_back(raw_voiced);
        if self.history.len() > HISTORY_FRAMES {
            self.history.pop_front();
        }
        let voiced_ratio =
            self.history.iter().filter(|&&v| v).count() as f32 / self.history.len().max(1) as f32;

        self.pitch_window.extend_from_slice(frame);
        if self.pitch_window.len() > PITCH_WINDOW_LEN {
            let excess = self.pitch_window.len() - PITCH_WINDOW_LEN;
            self.pitch_window.drain(..excess);
        }
        let pitch = if self.pitch_window.len() >= PITCH_WINDOW_LEN {
            estimate_pitch(&self.pitch_window, SAMPLE_RATE)
        } else {
            None
        };

        let level = rms(frame);

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

        let confirmed_voiced = self.history.len() == HISTORY_FRAMES
            && voiced_ratio >= ONSET_RATIO
            && above_noise_floor;

        FrameResult {
            score,
            confirmed_voiced,
            pitch,
            rms: level,
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

            // Живой однострочный индикатор, перезаписывается на месте -
            // не более 12-13 раз в секунду, чтобы не грузить терминал.
            if now.duration_since(last_meter_at) >= Duration::from_millis(80) {
                last_meter_at = now;
                let label = match state {
                    State::Speech => "говорит",
                    State::Silence => "тишина ",
                };
                let filled = (r.score.clamp(0.0, 1.0) * 20.0) as usize;
                let bar: String = "█".repeat(filled) + &"░".repeat(20 - filled);
                print!("{CLEAR_LINE}[{bar}] score {:.2}  {label}", r.score);
                io::stdout().flush().ok();
            }
        }
    }
}
