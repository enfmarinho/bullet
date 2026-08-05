use bullet_lib::{
    game::{
        inputs::{get_num_buckets, ChessBucketsMirrored},
        outputs::MaterialCount,
    },
    nn::{
        optimiser::{AdamW, AdamWParams},
        InitSettings, Shape,
    },
    trainer::{
        save::SavedFormat,
        schedule::{lr, wdl, TrainingSchedule, TrainingSteps},
        settings::LocalSettings,
    },
    value::{loader, ValueTrainerBuilder},
};
use std::{fs, path::Path};

const CHECKPOINT_PATH: &str = "";
const OUTDIR: &str = "checkpoints/minke37/v2";
const DATASET_PATH: &str = "data/selfgen/interleaved_12-33.vf";
const N_THREADS: usize = 4;
const BUFFER_SIZE_MB: usize = 2048;

const START_SB: usize = 1;
const END_FIRST_SB: usize = 100;
const END_SECOND_SB: usize = 600;
const END_FINETUNE_SB: usize = 800;

const BATCH_SIZE: usize = 16_384;
const BATCHES_PER_SUPERBATCH: usize = 6104;

const SAVE_RATE: usize = 25;
const INITIAL_LR: f32 = 1e-3;
const FINAL_LR: f32 = 1e-6;
const FINETUNE_LR: f32 = 1e-6;

const INITIAL_WDL: f32 = 0.20;
const FINAL_WDL: f32 = 0.60;
const FINETUNE_WDL: f32 = 0.90;

const SCALE: f32 = 400.0;

// Quantization
const QA: i16 = 255;
const QB: i16 = 128;
const QC: i32 = 64;

const L1_SHIFT: usize = 8;
const L1_SHIFT_SCALE: f32 = QA as f32 / (1 << L1_SHIFT) as f32;
const I8_RANGE: f32 = i8::MAX as f32 / QB as f32;
const L1_RANGE: f32 = I8_RANGE * L1_SHIFT_SCALE * L1_SHIFT_SCALE;

// arch
const L1_SIZE: usize = 1280;
const L2_SIZE: usize = 16;
const L3_SIZE: usize = 32;

#[rustfmt::skip]
const BUCKET_LAYOUT: [usize; 32] = [
    0, 1, 2, 3,
    4, 5, 6, 7,
    8, 8, 8, 8,
    9, 9, 9, 9,
    9, 9, 9, 9,
    9, 9, 9, 9,
    9, 9, 9, 9,
    9, 9, 9, 9,
];
const NUM_INPUT_BUCKETS: usize = get_num_buckets(&BUCKET_LAYOUT);
const NUM_OUTPUT_BUCKETS: usize = 8;

static SOURCE_CODE: &str = include_str!("minke.rs");
fn save_config() {
    let dest_path = Path::new(OUTDIR).join("config.rs");
    let _ = fs::create_dir_all(OUTDIR);
    let _ = fs::write(dest_path, SOURCE_CODE);
}

fn main() {
    save_config();

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(AdamW)
        .inputs(ChessBucketsMirrored::new(BUCKET_LAYOUT))
        .output_buckets(MaterialCount::<NUM_OUTPUT_BUCKETS>)
        .save_format(&[
            SavedFormat::id("l0w")
                .transform(|store, weights| {
                    let factorizer = store.get("l0f").values.f32().repeat(NUM_INPUT_BUCKETS);
                    weights.into_iter().zip(factorizer).map(|(a, b)| a + b).collect()
                })
                .round()
                .quantise::<i16>(QA),
            SavedFormat::id("l0b").round().quantise::<i16>(QA),
            SavedFormat::id("l1w")
                .transform(|_, mut weights| {
                    for i in weights.iter_mut() {
                        *i /= L1_SHIFT_SCALE * L1_SHIFT_SCALE;
                    }
                    weights
                })
                .round()
                .quantise::<i8>(QB)
                .transpose(),
            SavedFormat::id("l1b").round().quantise::<i32>(QC * (1 << L1_SHIFT)),
            SavedFormat::id("l2w").round().quantise::<i32>(QC).transpose(),
            SavedFormat::id("l2b").round().quantise::<i32>(QC.pow(3)),
            SavedFormat::id("l3w").round().quantise::<i32>(QC).transpose(),
            SavedFormat::id("l3b").round().quantise::<i32>(QC.pow(4)),
        ])
        .build_custom(|builder, (stm_inputs, ntm_inputs, output_buckets), target| {
            // input layer factoriser
            let l0f = builder.new_weights("l0f", Shape::new(L1_SIZE, 768), InitSettings::Zeroed);
            let expanded_factoriser = l0f.repeat(NUM_INPUT_BUCKETS);

            // input layer weights
            let mut l0 = builder.new_affine("l0", 768 * NUM_INPUT_BUCKETS, L1_SIZE);
            l0.init_with_effective_input_size(32);
            l0.weights = l0.weights + expanded_factoriser;

            // layer stacks weights
            let l1 = builder.new_affine("l1", L1_SIZE, NUM_OUTPUT_BUCKETS * L2_SIZE);
            let l2 = builder.new_affine("l2", L2_SIZE * 2, NUM_OUTPUT_BUCKETS * L3_SIZE);
            let l3 = builder.new_affine("l3", L3_SIZE, NUM_OUTPUT_BUCKETS);

            // inference
            let ft_forward = |input, start, end| l0.slice(start, end).forward(input).crelu();
            let stm_hidden = ft_forward(stm_inputs, 0, L1_SIZE / 2) * ft_forward(stm_inputs, L1_SIZE / 2, L1_SIZE);
            let ntm_hidden = ft_forward(ntm_inputs, 0, L1_SIZE / 2) * ft_forward(ntm_inputs, L1_SIZE / 2, L1_SIZE);
            let l0_out = stm_hidden.concat(ntm_hidden);

            let l1_out = l1.forward(l0_out).select(output_buckets);
            let hl2 = l1_out.concat(l1_out.abs_pow(2.0)).crelu();

            let l2_out = l2.forward(hl2).select(output_buckets);
            let hl3 = l2_out.crelu();

            let l3_out = l3.forward(hl3).select(output_buckets);

            // loss
            let ones_l1_vec = builder.new_constant(Shape::new(1, L1_SIZE), &[1.0 / L1_SIZE as f32; L1_SIZE]);
            let l0_mean_activation = ones_l1_vec.matmul(l0_out);
            let eval_loss = l3_out.sigmoid().squared_error(target);
            let loss = eval_loss + 0.005 * l0_mean_activation;

            (l3_out, loss)
        });

    // need to account for factoriser weight magnitudes
    let l0_clip = AdamWParams { max_weight: 0.99, min_weight: -0.99, ..Default::default() };
    let l1_clip = AdamWParams { max_weight: L1_RANGE, min_weight: -L1_RANGE, ..Default::default() };
    trainer.optimiser.set_params_for_weight("l0w", l0_clip);
    trainer.optimiser.set_params_for_weight("l0f", l0_clip);
    trainer.optimiser.set_params_for_weight("l1w", l1_clip);

    let settings = LocalSettings { threads: N_THREADS, test_set: None, output_directory: OUTDIR, batch_queue_size: 64 };

    let data_loader = {
        let filter = loader::viribinpack::Filter {
            min_ply: 16,
            min_pieces: 4,
            max_eval: 10000,
            filter_tactical: true,
            filter_check: true,
            filter_castling: false,
            max_eval_incorrectness: u32::MAX,
            random_fen_skipping: true,
            random_fen_skip_probability: 0.50,

            wdl_filtered: false,
            wdl_model_params_a: [0.0; 4],
            wdl_model_params_b: [0.0; 4],

            material_min: 17,
            material_max: 78,
            mom_target: 58,
            wdl_heuristic_scale: 1.5,
        };

        loader::ViriBinpackLoader::new(DATASET_PATH, BUFFER_SIZE_MB, N_THREADS, filter)
    };

    if !CHECKPOINT_PATH.is_empty() {
        trainer.load_from_checkpoint(CHECKPOINT_PATH);
    }

    let schedule = TrainingSchedule {
        net_id: "minke".to_string(),
        eval_scale: SCALE,
        steps: TrainingSteps {
            batch_size: BATCH_SIZE,
            batches_per_superbatch: BATCHES_PER_SUPERBATCH,
            start_superbatch: START_SB,
            end_superbatch: END_SECOND_SB,
        },
        wdl_scheduler: wdl::Sequence {
            first: wdl::ConstantWDL { value: INITIAL_WDL },
            second: wdl::LinearWDL { start: INITIAL_WDL, end: FINAL_WDL },
            first_scheduler_final_superbatch: END_FIRST_SB,
        },
        lr_scheduler: lr::LinearDecayLR { initial_lr: INITIAL_LR, final_lr: FINAL_LR, final_superbatch: END_SECOND_SB },
        save_rate: SAVE_RATE,
    };
    trainer.run(&schedule, &settings, &data_loader);

    let finetune_schedule = TrainingSchedule {
        net_id: "minke".to_string(),
        eval_scale: SCALE,
        steps: TrainingSteps {
            batch_size: BATCH_SIZE,
            batches_per_superbatch: BATCHES_PER_SUPERBATCH,
            start_superbatch: END_SECOND_SB + 1,
            end_superbatch: END_FINETUNE_SB,
        },
        wdl_scheduler: wdl::ConstantWDL { value: FINETUNE_WDL },
        lr_scheduler: lr::ConstantLR { value: FINETUNE_LR },
        save_rate: SAVE_RATE,
    };
    trainer.run(&finetune_schedule, &settings, &data_loader);

    for fen in [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    ] {
        let eval = trainer.eval(fen);
        println!("FEN: {fen}");
        println!("EVAL: {}", SCALE * eval);
    }
}
