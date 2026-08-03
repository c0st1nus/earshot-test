use anyhow::{bail, Context, Result};
use hound::{SampleFormat, WavReader};
use std::path::Path;

pub fn load_wav_mono_i16_16k(path: &Path) -> Result<Vec<i16>> {
    let reader = WavReader::open(path)
        .with_context(|| format!("Не удалось открыть WAV: {}", path.display()))?;

    let spec = reader.spec();

    if spec.sample_rate != 16_000 {
        bail!(
            "Файл {} имеет sample_rate={}, а нужно 16000 Hz. \
             Конвертируйте: ffmpeg -i input -ac 1 -ar 16000 -sample_fmt s16 output.wav",
            path.display(),
            spec.sample_rate
        );
    }

    if spec.channels != 1 {
        bail!(
            "Файл {} имеет {} каналов, а нужен моно (1 канал). \
             Конвертируйте: ffmpeg -i input -ac 1 -ar 16000 -sample_fmt s16 output.wav",
            path.display(),
            spec.channels
        );
    }

    if spec.sample_format != SampleFormat::Int || spec.bit_depth != 16 {
        bail!(
            "Файл {} должен быть PCM16 integer. \
             Конвертируйте: ffmpeg -i input -ac 1 -ar 16000 -sample_fmt s16 output.wav",
            path.display()
        );
    }

    let samples = reader
        .into_samples::<i16>()
        .collect::<Result<Vec<_>, _>>()
        .with_context(|| format!("Не удалось прочитать сэмплы из {}", path.display()))?;

    Ok(samples)
}
