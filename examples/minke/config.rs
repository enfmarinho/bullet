use core::include_str;
use std::{fs, path::Path};

use bullet_lib::game::inputs::get_num_buckets;

pub const CHECKPOINT_PATH: &str = "";
pub const OUTDIR: &str = "checkpoints/minke42/v1";
pub const DATASET_PATH: &str = "data/selfgen/interleaved_12-40.vf";

pub const N_THREADS: usize = 4;
pub const BUFFER_SIZE_MB: usize = 2048;
pub const SAVE_RATE: usize = 25;

pub const BATCH_SIZE: usize = 16_384;
pub const BATCHES_PER_SUPERBATCH: usize = 6104;

pub const SCALE: f32 = 400.0;

// stage 1
pub const S0_SBS: usize = 100;
pub const S1_SBS: usize = 800;

pub const S1_INITIAL_LR: f32 = 1e-3;
pub const S1_FINAL_LR: f32 = 1e-6;

pub const S1_INITIAL_WDL: f32 = 0.20;
pub const S1_FINAL_WDL: f32 = 0.40;

// stage 2
pub const S2_SBS: usize = 200;

pub const S2_INITIAL_LR: f32 = 1e-5;
pub const S2_FINAL_LR: f32 = 1e-7;

pub const FINETUNE_WDL: f32 = 1.00;

// Quantization
pub const QA: i16 = 255;
pub const QB: i16 = 128;
pub const QC: i32 = 64;

pub const L1_SHIFT: usize = 8;
pub const L1_SHIFT_SCALE: f32 = QA as f32 / (1 << L1_SHIFT) as f32;
pub const I8_RANGE: f32 = i8::MAX as f32 / QB as f32;
pub const L1_RANGE: f32 = I8_RANGE * L1_SHIFT_SCALE * L1_SHIFT_SCALE;

// arch
pub const L1_SIZE: usize = 1280;
pub const L2_SIZE: usize = 16;
pub const L3_SIZE: usize = 32;

#[rustfmt::skip]
pub const BUCKET_LAYOUT: [usize; 32] = [
    0,  1,  2,  3,
    4,  5,  6,  7,
    8,  8,  9,  9,
    10, 10, 11, 11,
    12, 12, 13, 13,
    12, 12, 13, 13,
    14, 14, 15, 15,
    14, 14, 15, 15,
];
pub const NUM_INPUT_BUCKETS: usize = get_num_buckets(&BUCKET_LAYOUT);
pub const NUM_OUTPUT_BUCKETS: usize = 8;

static SOURCE_CODE: &str = include_str!("main.rs");
static CONFIG_CODE: &str = include_str!("config.rs");
pub fn save_config() {
    let save = |path, code| {
        let dest_path = Path::new(OUTDIR).join(path);
        let _ = fs::create_dir_all(OUTDIR);
        let _ = fs::write(dest_path, code);
    };

    save("main.rs", SOURCE_CODE);
    save("config.rs", CONFIG_CODE);
}
