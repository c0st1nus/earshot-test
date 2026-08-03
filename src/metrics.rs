use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
pub struct Annotation {
    pub speech: Vec<SpeechInterval>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SpeechInterval {
    pub start_ms: u64,
    pub end_ms: u64,
}

impl Annotation {
    pub fn load(path: &Path) -> Result<Self> {
        let file = File::open(path)
            .with_context(|| format!("Не удалось открыть аннотацию: {}", path.display()))?;

        serde_json::from_reader(file)
            .with_context(|| format!("Не удалось распарсить аннотацию: {}", path.display()))
    }
}

/// Размечает фреймы как speech / non-speech.
/// Фрейм считается речью, если он перекрыт ground truth интервалом минимум на 50%.
pub fn frame_labels(num_frames: usize, frame_ms: f64, annotation: &Annotation) -> Vec<bool> {
    let mut labels = vec![false; num_frames];

    let mut intervals: Vec<(f64, f64)> = annotation
        .speech
        .iter()
        .map(|interval| (interval.start_ms as f64, interval.end_ms as f64))
        .collect();

    intervals.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));

    let mut active = 0usize;

    for frame in 0..num_frames {
        let frame_start = frame as f64 * frame_ms;
        let frame_end = frame_start + frame_ms;

        while active < intervals.len() && intervals[active].1 <= frame_start {
            active += 1;
        }

        for &(start_ms, end_ms) in intervals.iter().skip(active) {
            if start_ms >= frame_end {
                break;
            }

            let overlap = frame_end.min(end_ms) - frame_start.max(start_ms);

            if overlap > 0.0 && overlap >= frame_ms * 0.5 {
                labels[frame] = true;
                break;
            }
        }
    }

    labels
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct ClassificationMetrics {
    pub tp: u64,
    pub fp: u64,
    pub fn_count: u64,
    pub tn: u64,
    pub precision: f32,
    pub recall: f32,
    pub f1_score: f32,
    pub accuracy: f32,
}

pub fn compute_classification(labels: &[bool], predictions: &[bool]) -> ClassificationMetrics {
    let n = labels.len().min(predictions.len());

    let mut tp = 0u64;
    let mut fp = 0u64;
    let mut fn_count = 0u64;
    let mut tn = 0u64;

    for i in 0..n {
        match (labels[i], predictions[i]) {
            (true, true) => tp += 1,
            (false, true) => fp += 1,
            (true, false) => fn_count += 1,
            (false, false) => tn += 1,
        }
    }

    if predictions.len() > n {
        fp += (predictions.len() - n) as u64;
    }

    if labels.len() > n {
        fn_count += (labels.len() - n) as u64;
    }

    let precision = if tp + fp == 0 {
        0.0
    } else {
        tp as f32 / (tp + fp) as f32
    };

    let recall = if tp + fn_count == 0 {
        0.0
    } else {
        tp as f32 / (tp + fn_count) as f32
    };

    let f1_score = if precision + recall == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    };

    let total = (tp + fp + fn_count + tn) as f32;

    let accuracy = if total == 0.0 {
        0.0
    } else {
        (tp + tn) as f32 / total
    };

    ClassificationMetrics {
        tp,
        fp,
        fn_count,
        tn,
        precision,
        recall,
        f1_score,
        accuracy,
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Segment {
    pub start_ms: u64,
    pub end_ms: u64,
}

pub fn bool_frames_to_segments(
    predictions: &[bool],
    frame_ms: f64,
    min_segment_ms: u64,
) -> Vec<Segment> {
    let mut segments = Vec::new();
    let mut start: Option<usize> = None;

    for (i, &is_speech) in predictions.iter().enumerate() {
        if is_speech && start.is_none() {
            start = Some(i);
        }

        if !is_speech {
            if let Some(start_frame) = start {
                let start_ms = (start_frame as f64 * frame_ms).round() as u64;
                let end_ms = (i as f64 * frame_ms).round() as u64;

                if end_ms.saturating_sub(start_ms) >= min_segment_ms {
                    segments.push(Segment { start_ms, end_ms });
                }

                start = None;
            }
        }
    }

    if let Some(start_frame) = start {
        let start_ms = (start_frame as f64 * frame_ms).round() as u64;
        let end_ms = (predictions.len() as f64 * frame_ms).round() as u64;

        if end_ms.saturating_sub(start_ms) >= min_segment_ms {
            segments.push(Segment { start_ms, end_ms });
        }
    }

    segments
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct LatencyStats {
    pub avg_ms: f64,
    pub avg_abs_ms: f64,
    pub count: usize,
}

/// Считает задержку начала детекции относительно ground truth.
/// Для каждого GT-сегмента ищем первый детектированный сегмент, который его пересекает.
pub fn onset_latency(annotation: &Annotation, detected: &[Segment]) -> Option<LatencyStats> {
    let mut diffs = Vec::new();

    for gt in &annotation.speech {
        let gt_start = gt.start_ms as f64;
        let gt_end = gt.end_ms as f64;

        let maybe_detected = detected.iter().find(|segment| {
            (segment.start_ms as f64) <= gt_end && (segment.end_ms as f64) >= gt_start
        });

        if let Some(segment) = maybe_detected {
            diffs.push(segment.start_ms as f64 - gt_start);
        }
    }

    if diffs.is_empty() {
        return None;
    }

    let count = diffs.len();

    let avg_ms = diffs.iter().sum::<f64>() / count as f64;
    let avg_abs_ms = diffs.iter().map(|v| v.abs()).sum::<f64>() / count as f64;

    Some(LatencyStats {
        avg_ms,
        avg_abs_ms,
        count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classification_perfect() {
        let labels = vec![false, true, true, false];
        let preds = vec![false, true, true, false];

        let metrics = compute_classification(&labels, &preds);

        assert_eq!(metrics.tp, 2);
        assert_eq!(metrics.fp, 0);
        assert_eq!(metrics.fn_count, 0);
        assert_eq!(metrics.tn, 2);
        assert!(metrics.f1_score > 0.99);
    }

    #[test]
    fn segments_min_duration() {
        let preds = vec![true, true, true, false, true];
        let segments = bool_frames_to_segments(&preds, 16.0, 32);

        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].start_ms, 0);
        assert_eq!(segments[0].end_ms, 48);
    }
}
