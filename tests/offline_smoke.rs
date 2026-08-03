use std::path::Path;
use vad_core::{
    audio,
    engine::{process_offline, FRAME_SIZE},
};

fn write_wav(path: &Path, samples: &[i16]) -> Result<(), hound::Error> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = hound::WavWriter::create(path, spec)?;

    for sample in samples {
        writer.write_sample(*sample)?;
    }

    writer.finalize()?;

    Ok(())
}

#[test]
fn offline_smoke_with_generated_silence() {
    let dir = std::env::temp_dir();
    let wav_path = dir.join("vad_offline_smoke.wav");

    let samples = vec![0i16; 16_000]; // 1 секунда тишины

    write_wav(&wav_path, &samples).expect("Failed to write test WAV");

    let loaded = audio::load_wav_mono_i16_16k(&wav_path).expect("Failed to load test WAV");

    let predictions = process_offline(&loaded);

    assert!(!predictions.is_empty());
    assert!(predictions.len() >= loaded.len() / FRAME_SIZE);
}
