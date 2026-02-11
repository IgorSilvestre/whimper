use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use clap::Parser;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

#[derive(Parser)]
#[command(name = "whimper", about = "Transcribe audio/video files using Whisper")]
struct Cli {
    /// Path to the audio/video file to transcribe
    #[arg(short = 'f', long = "file")]
    file: PathBuf,

    /// Language code for transcription (e.g. en, pt, es, fr)
    #[arg(short = 'l', long = "language", default_value = "en")]
    language: String,
}

fn convert_to_wav(input: &Path) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let tmp_wav = input.with_extension("__tmp_whimper.wav");
    let status = Command::new("ffmpeg")
        .args([
            "-y",
            "-i",
            input.to_str().ok_or("invalid input path")?,
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            tmp_wav.to_str().ok_or("invalid temp path")?,
        ])
        .status()?;
    if !status.success() {
        return Err("ffmpeg conversion failed".into());
    }
    Ok(tmp_wav)
}

fn read_wav_samples(path: &Path) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let reader = hound::WavReader::open(path)?;
    let spec = reader.spec();

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max = (1 << (spec.bits_per_sample - 1)) as f32;
            reader
                .into_samples::<i32>()
                .filter_map(Result::ok)
                .map(|s| s as f32 / max)
                .collect()
        }
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .filter_map(Result::ok)
            .collect(),
    };

    // If stereo+, take every Nth sample (first channel)
    if spec.channels > 1 {
        let ch = spec.channels as usize;
        Ok(samples.into_iter().step_by(ch).collect())
    } else {
        Ok(samples)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenvy::dotenv().ok();

    let cli = Cli::parse();
    let input_path = &cli.file;

    if !input_path.exists() {
        return Err(format!("File not found: {}", input_path.display()).into());
    }

    let model_path =
        std::env::var("MODEL_FILE").map_err(|_| "MODEL_FILE not set in environment or .env")?;

    // Convert to 16 kHz mono WAV via ffmpeg
    println!("Converting to 16 kHz mono WAV...");
    let wav_path = convert_to_wav(input_path)?;

    // Read samples
    let samples = read_wav_samples(&wav_path)?;

    // Clean up temp file
    let _ = fs::remove_file(&wav_path);

    // Initialize whisper
    println!("Loading model...");
    let ctx = WhisperContext::new_with_params(&model_path, WhisperContextParameters::default())
        .map_err(|e| format!("Failed to load model: {e}"))?;

    let mut state = ctx.create_state().map_err(|e| format!("Failed to create state: {e}"))?;

    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    params.set_language(Some(&cli.language));
    params.set_print_progress(true);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);

    println!("Transcribing...");
    state
        .full(params, &samples)
        .map_err(|e| format!("Transcription failed: {e}"))?;

    let num_segments = state.full_n_segments().map_err(|e| format!("{e}"))?;
    let mut text = String::new();
    for i in 0..num_segments {
        if let Ok(segment) = state.full_get_segment_text(i) {
            text.push_str(&segment);
        }
    }
    let text = text.trim().to_string();

    // Write output
    let output_path = input_path.with_extension("txt");
    fs::write(&output_path, &text)?;

    println!("Transcription saved to: {}", output_path.display());
    Ok(())
}
