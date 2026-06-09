use bullet_lib::{
    game::{
        inputs::{get_num_buckets, ChessBucketsMirrored},
        // outputs::MaterialCount,
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

const OUTPUT_DIRECTORY: &str = "checkpoints/minke20/v3";
const BINPACK_PATH: &str = "data/selfgen/interleaved_7-19.vf";
const N_THREADS: usize = 4;
const BUFFER_SIZE_MB: usize = 2048;

const START_SUPERBATCH: usize = 1;
const END_FIRST_SCHEDULER: usize = 100;
const END_SECOND_SCHEDULER: usize = 600;

const BATCH_SIZE: usize = 16_384;
const BATCHES_PER_SUPERBATCH: usize = 6104;

const SAVE_RATE: usize = 100;
const INITIAL_LR: f32 = 1e-3;
const FINAL_LR: f32 = 1e-6;

const INITIAL_WDL: f32 = 0.20;
const FINAL_WDL: f32 = 0.60;

const HIDDEN_SIZE: usize = 1024;
const SCALE: f32 = 400.0;
const QA: i16 = 255;
const QB: i16 = 64;

// arch
#[rustfmt::skip]
const BUCKET_LAYOUT: [usize; 32] = [
    0, 0, 1, 1,
    2, 2, 2, 2,
    3, 3, 3, 3,
    3, 3, 3, 3,
    3, 3, 3, 3,
    3, 3, 3, 3,
    3, 3, 3, 3,
    3, 3, 3, 3,
];
const NUM_INPUT_BUCKETS: usize = get_num_buckets(&BUCKET_LAYOUT);
// const NUM_OUTPUT_BUCKETS: usize = 1;

static SOURCE_CODE: &str = include_str!("minke.rs");
fn save_config() {
    let dest_path = Path::new(OUTPUT_DIRECTORY).join("config.rs");
    let _ = fs::create_dir_all(OUTPUT_DIRECTORY);
    let _ = fs::write(dest_path, SOURCE_CODE);
}

fn main() {
    save_config();

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(AdamW)
        .inputs(ChessBucketsMirrored::new(BUCKET_LAYOUT))
        // .output_buckets(MaterialCount::<NUM_OUTPUT_BUCKETS>)
        .save_format(&[
            // merge in the factoriser weights
            SavedFormat::id("l0w")
                .transform(|store, weights| {
                    let factoriser = store.get("l0f").values.f32().repeat(NUM_INPUT_BUCKETS);
                    weights.into_iter().zip(factoriser).map(|(a, b)| a + b).collect()
                })
                .round()
                .quantise::<i16>(QA),
            SavedFormat::id("l0b").round().quantise::<i16>(QA),
            SavedFormat::id("l1w").round().quantise::<i16>(QB),
            // SavedFormat::id("l1w").round().quantise::<i16>(QB).transpose(),
            SavedFormat::id("l1b").round().quantise::<i16>(QA * QB),
        ])
        .loss_fn(|output, target| output.sigmoid().squared_error(target))
        // .build(|builder, stm_inputs, ntm_inputs, output_buckets| {
        .build(|builder, stm_inputs, ntm_inputs| {
            // input layer factoriser
            let l0f = builder.new_weights("l0f", Shape::new(HIDDEN_SIZE, 768), InitSettings::Zeroed);
            let expanded_factoriser = l0f.repeat(NUM_INPUT_BUCKETS);

            // input layer weights
            let mut l0 = builder.new_affine("l0", 768 * NUM_INPUT_BUCKETS, HIDDEN_SIZE);
            l0.weights = l0.weights + expanded_factoriser;

            // output layer weights
            let l1 = builder.new_affine("l1", 2 * HIDDEN_SIZE, 1);
            // let l1 = builder.new_affine("l1", 2 * HIDDEN_SIZE, NUM_OUTPUT_BUCKETS);

            // inference
            let stm_hidden = l0.forward(stm_inputs).screlu();
            let ntm_hidden = l0.forward(ntm_inputs).screlu();
            let hidden_layer = stm_hidden.concat(ntm_hidden);
            l1.forward(hidden_layer)
            // l1.forward(hidden_layer).select(output_buckets)
        });

    // need to account for factoriser weight magnitudes
    let stricter_clipping = AdamWParams { max_weight: 0.99, min_weight: -0.99, ..Default::default() };
    trainer.optimiser.set_params_for_weight("l0w", stricter_clipping);
    trainer.optimiser.set_params_for_weight("l0f", stricter_clipping);

    let schedule = TrainingSchedule {
        net_id: "minke".to_string(),
        eval_scale: SCALE,
        steps: TrainingSteps {
            batch_size: BATCH_SIZE,
            batches_per_superbatch: BATCHES_PER_SUPERBATCH,
            start_superbatch: START_SUPERBATCH,
            end_superbatch: END_SECOND_SCHEDULER,
        },
        wdl_scheduler: wdl::Sequence {
            first: wdl::ConstantWDL { value: INITIAL_WDL },
            second: wdl::LinearWDL { start: INITIAL_WDL, end: FINAL_WDL },
            first_scheduler_final_superbatch: END_FIRST_SCHEDULER,
        },
        lr_scheduler: lr::LinearDecayLR {
            initial_lr: INITIAL_LR,
            final_lr: FINAL_LR,
            final_superbatch: END_SECOND_SCHEDULER,
        },
        save_rate: SAVE_RATE,
    };

    let settings =
        LocalSettings { threads: N_THREADS, test_set: None, output_directory: OUTPUT_DIRECTORY, batch_queue_size: 64 };

    let data_loader = {
        let filter = loader::viribinpack::Filter {
            min_ply: 16,
            min_pieces: 4,
            max_eval: 10000,
            filter_tactical: true,
            filter_check: true,
            filter_castling: false,
            max_eval_incorrectness: u32::MAX,
            random_fen_skipping: false,
            random_fen_skip_probability: 0.50,

            wdl_filtered: false,
            wdl_model_params_a: [0.0; 4],
            wdl_model_params_b: [0.0; 4],

            material_min: 17,
            material_max: 78,
            mom_target: 58,
            wdl_heuristic_scale: 1.5,
        };

        loader::ViriBinpackLoader::new(BINPACK_PATH, BUFFER_SIZE_MB, N_THREADS, filter)
    };

    trainer.run(&schedule, &settings, &data_loader);

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
