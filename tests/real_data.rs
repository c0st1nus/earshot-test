use std::path::Path;

#[test]
fn real_data_regression() {
    let wav = match std::env::var("VAD_TEST_WAV") {
        Ok(v) => v,
        Err(_) => {
            eprintln!("VAD_TEST_WAV not set, skipping real data test");
            return;
        }
    };

    let annotation = match std::env::var("VAD_TEST_ANN") {
        Ok(v) => v,
        Err(_) => {
            eprintln!("VAD_TEST_ANN not set, skipping real data test");
            return;
        }
    };

    let samples = vad_core::audio::load_wav_mono_i16_16k(Path::new(&wav))
        .expect("Failed to load VAD_TEST_WAV");

    let annotation = vad_core::metrics::Annotation::load(Path::new(&annotation))
        .expect("Failed to load VAD_TEST_ANN");

    let predictions = vad_core::engine::process_offline(&samples);

    let labels =
        vad_core::metrics::frame_labels(predictions.len(), vad_core::engine::FRAME_MS, &annotation);

    let metrics = vad_core::metrics::compute_classification(&labels, &predictions);

    println!("Real data metrics: {:?}", metrics);

    assert!(predictions.len() > 0);

    // Если хотите жесткий gate в тестах, раскомментируйте:
    // assert!(metrics.f1_score > 0.80);
}
