mod config;

use bullet_lib::{
    game::{inputs::ChessBucketsMirrored, outputs::MaterialCount},
    nn::{
        optimiser::{AdamW, AdamWParams},
        InitSettings, Shape,
    },
    trainer::{
        save::SavedFormat,
        schedule::{lr, wdl, TrainingSchedule, TrainingSteps},
        settings::LocalSettings,
    },
    value::{
        loader::{self, ViriFilter},
        ValueTrainerBuilder,
    },
};
use config::*;
use rand::{rng, Rng};
use std::cell::{Cell, RefCell};
use viriformat::{
    chess::{board::Board, chessmove::Move},
    dataformat::WDL,
};

macro_rules! run_stage {
    ($trainer:expr, $settings:expr, $data_loader:expr, $stage:expr, $sbs:expr, $lr:expr, $wdl:expr) => {{
        let schedule = TrainingSchedule {
            net_id: format!("minke-stage{}", $stage),
            eval_scale: SCALE,
            steps: TrainingSteps {
                batch_size: BATCH_SIZE,
                batches_per_superbatch: BATCHES_PER_SUPERBATCH,
                start_superbatch: 1,
                end_superbatch: $sbs,
            },
            lr_scheduler: $lr,
            wdl_scheduler: $wdl,
            save_rate: SAVE_RATE,
        };

        $trainer.run(&schedule, $settings, $data_loader);
    }};
}

fn piece_count_acceptance(board: &Board) -> f64 {
    // from pawnnochio training scripts
    #[rustfmt::skip]
    const DESIRED_DISTRIBUTION: [f64; 33] = [
        0.018411966423, 0.020641545085, 0.022727271053,
        0.024669162740, 0.026467201733, 0.028121406444,
        0.029631758462, 0.030998276198, 0.032220941240,
        0.033299772000, 0.034234750067, 0.035025893853,
        0.035673184944, 0.036176641754, 0.036536245870,
        0.036752015705, 0.036823932846, 0.036752015705,
        0.036536245870, 0.036176641754, 0.035673184944,
        0.035025893853, 0.034234750067, 0.033299772000,
        0.032220941240, 0.030998276198, 0.029631758462,
        0.028121406444, 0.026467201733, 0.024669162740,
        0.022727271053, 0.020641545085, 0.018411966423,
    ];

    thread_local! {
        static PIECE_COUNT_STATS: RefCell<[u64; 33]> = const { RefCell::new([0; 33]) };
        static PIECE_COUNT_TOTAL: Cell<u64> = const { Cell::new(0) };
    }

    let pc = board.pieces.occupied().count() as usize;
    let count = PIECE_COUNT_STATS.with_borrow_mut(|stats| {
        stats[pc] += 1;
        stats[pc]
    });
    let total = PIECE_COUNT_TOTAL.with(|t| {
        let total = t.get() + 1;
        t.set(total);
        total
    });
    let frequency = count as f64 / total as f64;

    let acceptance = 0.5 * DESIRED_DISTRIBUTION[pc] / frequency;
    acceptance.clamp(0., 1.)
}

fn filter(board: &Board, mv: Move, eval: i16, wdl: f32) -> bool {
    let viri_filter = loader::viribinpack::Filter {
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
    let wdl = match wdl {
        1.0 => WDL::Win,
        0.5 => WDL::Draw,
        0.0 => WDL::Loss,
        _ => unreachable!(),
    };
    let mut rng = rng();

    !viri_filter.should_filter(mv, eval as i32, board, wdl, &mut rng) && rng.random_bool(piece_count_acceptance(board))
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

    let data_loader =
        loader::ViriBinpackLoader::new(DATASET_PATH, BUFFER_SIZE_MB, N_THREADS, ViriFilter::Custom(filter));

    if !CHECKPOINT_PATH.is_empty() {
        trainer.load_from_checkpoint(CHECKPOINT_PATH);
    }

    // stage 0
    run_stage!(
        &mut trainer,
        &settings,
        &data_loader,
        0, // stage id
        S0_SBS,
        lr::Sequence {
            first: lr::LinearDecayLR {
                initial_lr: S0_WARMUP_INITIAL_LR,
                final_lr: S0_WARMUP_FINAL_LR,
                final_superbatch: S0_WARMUP_SBS,
            },
            second: lr::LinearDecayLR {
                initial_lr: S0_COOLDOWN_INITIAL_LR,
                final_lr: S0_COOLDOWN_FINAL_LR,
                final_superbatch: S0_COOLDOWN_SBS,
            },
            first_scheduler_final_superbatch: S0_WARMUP_SBS,
        },
        wdl::ConstantWDL { value: S0_WDL }
    );

    // stage 1
    run_stage!(
        &mut trainer,
        &settings,
        &data_loader,
        1, // stage id
        S1_SBS,
        lr::LinearDecayLR { initial_lr: S1_INITIAL_LR, final_lr: S1_FINAL_LR, final_superbatch: S1_SBS },
        wdl::LinearWDL { start: S1_INITIAL_WDL, end: S1_FINAL_WDL }
    );

    // stage 2
    run_stage!(
        &mut trainer,
        &settings,
        &data_loader,
        2, // stage id
        S2_SBS,
        lr::LinearDecayLR { initial_lr: S2_INITIAL_LR, final_lr: S2_FINAL_LR, final_superbatch: S2_SBS },
        wdl::ConstantWDL { value: S2_WDL }
    );

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
