use crate::position::{squares_iterator, GroupData, Piece, Position, Role, Square as TiltakSquare};
use arrayvec::ArrayVec;
use board_game_traits::{Color, Position as PositionTrait};

cfg_select! {
    all(target_feature = "avx512f", target_feature = "avx512bw") => {
        #[path = "simd/avx512.rs"]
        pub(super) mod simd;
    }
    target_feature = "avx2" => {
        #[path = "simd/avx2.rs"]
        pub(super) mod simd;
    }
    target_feature = "sse4.1" => {
        #[path = "simd/sse41.rs"]
        pub(super) mod simd;
    }
    all(target_arch = "aarch64", target_feature = "neon") => {
        #[path = "simd/neon.rs"]
        pub(super) mod simd;
    }
    all(target_arch = "wasm32", target_feature = "simd128") => {
        #[path = "simd/wasm.rs"]
        pub(super) mod simd;
    }
    _ => {
        #[path = "simd/scalar.rs"]
        pub(super) mod simd;
    }
}

use simd::*;

const MIN_HALF_KOMI: i8 = -7;
const MAX_HALF_KOMI: i8 = 7;

const COLOR_COUNT: usize = 2;
const PIECE_TYPE_COUNT: usize = 3;
const SQUARE_COUNT: usize = 36;

const TOP_INPUT_SIZE: usize = COLOR_COUNT * PIECE_TYPE_COUNT * SQUARE_COUNT;
const STACK_INPUT_SIZE: usize = 12 * COLOR_COUNT * PIECE_TYPE_COUNT * SQUARE_COUNT * COLOR_COUNT;
const RESERVES_INPUT_SIZE: usize = 30 * COLOR_COUNT;

const L1_SIZE: usize = 2560;
const L2_SIZE: usize = 32;
const L3_SIZE: usize = 32;

const L1_PAIR_COUNT: usize = L1_SIZE / 2;
const L2_PAIR_COUNT: usize = L2_SIZE / 2;

const RESERVE_BUCKET_COUNT: usize = 8;
const KOMI_BUCKET_COUNT: usize = (MAX_HALF_KOMI - MIN_HALF_KOMI) as usize + 1;
const OUTPUT_BUCKET_COUNT: usize = KOMI_BUCKET_COUNT * RESERVE_BUCKET_COUNT;

const Q0: i32 = 255;
const Q: i32 = 64;

const Q0_BITS: i32 = 8;
const Q1_BITS: i32 = 7;
const Q_BITS: i32 = 6;

#[repr(C, align(64))]
struct Network {
    ftw_top: [[i16; L1_SIZE]; TOP_INPUT_SIZE],
    ftw_stack: [[i16; L1_SIZE]; STACK_INPUT_SIZE],
    ftw_reserves: [[i16; L1_SIZE]; RESERVES_INPUT_SIZE],
    ftb: [i16; L1_SIZE],
    l1w: [[i8; L1_SIZE / 2 * L2_SIZE]; OUTPUT_BUCKET_COUNT],
    l1b: [[i32; L2_SIZE]; OUTPUT_BUCKET_COUNT],
    l2w: [[i16; L2_SIZE * L3_SIZE]; OUTPUT_BUCKET_COUNT],
    l2b: [[i32; L3_SIZE]; OUTPUT_BUCKET_COUNT],
    l3w: [[i32; L3_SIZE]; OUTPUT_BUCKET_COUNT],
    l3b: [i32; OUTPUT_BUCKET_COUNT],
}

static NET: Network = unsafe { std::mem::transmute(*include_bytes!(env!("EVALFILE"))) };

#[repr(C, align(64))]
struct Align64<T>(T);

const PACKUS_LANE: usize = 8;
const PACKUS_WIDTH: usize = 32;

const fn packus_position(neuron: usize) -> usize {
    let chunk = neuron % (2 * PACKUS_WIDTH);
    let lane = chunk / (2 * PACKUS_LANE);
    let half = chunk % (2 * PACKUS_LANE) / PACKUS_LANE;
    neuron - chunk + half * PACKUS_WIDTH + lane * PACKUS_LANE + chunk % PACKUS_LANE
}

const EXPLICIT_MULHI_SHIFT_UP: i32 = 7;
const IMPLIED_MULHI_SHIFT_DOWN: i32 = 16;
const MULHI_SHIFT: i32 = EXPLICIT_MULHI_SHIFT_UP - IMPLIED_MULHI_SHIFT_DOWN;

const SHIFT: i32 = Q0_BITS * 2 + MULHI_SHIFT + Q1_BITS - Q_BITS;

const L2_UNROLL: usize = 8 / (L2_SIZE / CHUNK_SIZE_I32);

pub(super) fn nn_eval_value(pos: &Position<6>, group_data: &GroupData<6>) -> f32 {
    assert!((MIN_HALF_KOMI..=MAX_HALF_KOMI).contains(&pos.komi().half_komi()));

    let acc = accumulate(pos, group_data);

    let output_bucket = calc_output_bucket(pos);

    let mut activated_ft = Align64([0; L1_PAIR_COUNT]);
    {
        let lo = zero_i16();
        let hi = set1_i16(Q0 as i16);

        let values = acc.as_ptr();
        let out = activated_ft.0.as_mut_ptr();

        let activate = |offset: usize| {
            let a = unsafe { load_i16(values.add(offset)) };
            let b = unsafe { load_i16(values.add(offset + L1_PAIR_COUNT)) };

            let a = min_i16(max_i16(a, lo), hi);
            let b = min_i16(b, hi);

            mulhi_shift_i16::<EXPLICIT_MULHI_SHIFT_UP>(a, b)
        };

        for i in (0..L1_PAIR_COUNT).step_by(2 * CHUNK_SIZE_I16) {
            let offset_1 = packus_position(i);
            let offset_2 = packus_position(i + PACKUS_LANE);
            let p = packus_i16(activate(offset_1), activate(offset_2));
            unsafe { store_u8(out.add(i), p) };
        }
    }
    debug_assert!(activated_ft.0.iter().all(|&e| e <= i8::MAX as u8));

    let mut l1_intermediate = [[zero_i32(); L2_SIZE / CHUNK_SIZE_I32]; L2_UNROLL];
    {
        let w = NET.l1w[output_bucket].as_ptr();
        let ft_groups = activated_ft.0.as_ptr();

        for i_outer in (0..L1_PAIR_COUNT / 4).step_by(2 * L2_UNROLL) {
            for (i_inner, intermediate) in l1_intermediate.iter_mut().enumerate() {
                let i_0 = i_outer + 2 * i_inner;
                let i_1 = i_0 + 1;
                let ft_vec_0 = unsafe { set1_u8x4(ft_groups.add(i_0 * 4)) };
                let ft_vec_1 = unsafe { set1_u8x4(ft_groups.add(i_1 * 4)) };

                for (j, intermediate) in intermediate.iter_mut().enumerate() {
                    let weights_0 =
                        unsafe { load_i8(w.add(i_0 * L2_SIZE * 4 + j * CHUNK_SIZE_I8)) };
                    let weights_1 =
                        unsafe { load_i8(w.add(i_1 * L2_SIZE * 4 + j * CHUNK_SIZE_I8)) };
                    *intermediate =
                        dpbusdx2_i32(*intermediate, ft_vec_0, weights_0, ft_vec_1, weights_1);
                }
            }
        }
    }

    let mut l1_out = Align64([0i16; L2_SIZE]);
    let skip;
    {
        let l1_bias = NET.l1b[output_bucket].as_ptr();

        let mut biased = [zero_i32(); L2_SIZE / CHUNK_SIZE_I32];
        for (i, biased) in biased.iter_mut().enumerate() {
            *biased = unsafe { load_i32(l1_bias.add(i * CHUNK_SIZE_I32)) };
            for unrolled in &l1_intermediate {
                *biased = add_i32(*biased, unrolled[i]);
            }
        }

        skip = to_array_i32(biased[0])[0];

        let hi = set1_i16(1 << (SHIFT + Q_BITS));

        for i in 0..L2_SIZE / CHUNK_SIZE_I16 {
            let packed = packus_i32(biased[i], biased[i + L2_SIZE / CHUNK_SIZE_I16]);
            let clamped = min_u16(packed, hi);
            let screlu = mulhi_i16(clamped, clamped);

            unsafe { store_i16(l1_out.0.as_mut_ptr().add(i * CHUNK_SIZE_I16), screlu) };
        }
    }

    let mut l2_intermediate = [[zero_i32(); 2]; L3_SIZE / CHUNK_SIZE_I32];
    {
        let l2_weights = NET.l2w[output_bucket].as_ptr();
        let l1_pairs = l1_out.0.as_ptr();

        for i in (0..L2_PAIR_COUNT).step_by(2) {
            let l1_vec_0 = unsafe { set1_i16x2(l1_pairs.add(i * 2)) };
            let l1_vec_1 = unsafe { set1_i16x2(l1_pairs.add((i + 1) * 2)) };
            for (j, intermediate) in l2_intermediate.iter_mut().enumerate() {
                let l2_weight_0 =
                    unsafe { load_i16(l2_weights.add((i * L3_SIZE + j * CHUNK_SIZE_I32) * 2)) };
                let l2_weight_1 = unsafe {
                    load_i16(l2_weights.add(((i + 1) * L3_SIZE + j * CHUNK_SIZE_I32) * 2))
                };
                *intermediate =
                    dpwssdx2_i32(*intermediate, l1_vec_0, l2_weight_0, l1_vec_1, l2_weight_1);
            }
        }
    }

    let l2_intermediate = l2_intermediate.map(reduce_dpwssd_i32);

    let mut l3_sum = zero_i32();
    {
        let l2_biases = NET.l2b[output_bucket].as_ptr();
        let l3_weights = NET.l3w[output_bucket].as_ptr();

        let skip_values = l1_out.0.as_ptr();

        let lo = zero_i32();
        let hi3 = set1_i32(1 << (3 * Q_BITS));

        for (i, &intermediate) in l2_intermediate.iter().enumerate() {
            let l2_bias = unsafe { load_i32(l2_biases.add(i * CHUNK_SIZE_I32)) };
            let l3_weight = unsafe { load_i32(l3_weights.add(i * CHUNK_SIZE_I32)) };

            let skip = unsafe { load_widen_i16_i32(skip_values.add(i * CHUNK_SIZE_I32)) };

            let biased = add_i32(intermediate, l2_bias);
            let activated = min_i32(max_i32(biased, lo), hi3);
            let skipped = add_i32(shift_left_i32::<Q_BITS>(skip), activated);

            l3_sum = add_i32(l3_sum, mul_i32(skipped, l3_weight));
        }
    }

    let bias = NET.l3b[output_bucket];
    let skip = i64::from(skip) * i64::from(Q * Q * Q * Q / (Q * 256));

    let out = i64::from(hsum_i32(l3_sum) + bias) + skip;
    let out = (out as f64 / f64::from(Q * Q * Q * Q)) as f32;

    match pos.side_to_move() {
        Color::White => out,
        Color::Black => -out,
    }
}

#[must_use]
fn accumulate(pos: &Position<6>, group_data: &GroupData<6>) -> [i16; L1_SIZE] {
    let mut acc = [0; L1_SIZE];

    let perspective = pos.side_to_move();

    let caps = match pos.side_to_move() {
        Color::White => group_data.white_caps,
        Color::Black => group_data.black_caps,
    };
    let cap_sq = caps.occupied_square().map(Square::from_tiltak);

    let transform = normalizer(cap_sq);

    let mut rows = Updates::default();

    for (tiltak_sq, top) in
        squares_iterator().filter_map(|sq| pos.top_stones()[sq].map(|piece| (sq, piece)))
    {
        let stack = pos.get_stack(tiltak_sq);

        let sq = Square::from_tiltak(tiltak_sq);
        let perspective_sq = sq.transform(transform);

        rows.adds.push(
            &NET.ftw_top[top_feature_idx(perspective, top.color(), top.role(), perspective_sq)],
        );

        if stack.height == 1 {
            continue;
        }

        for (stack_index, color) in stack
            .into_iter()
            .enumerate()
            .map(|(stack_index, stack_piece)| (stack_index, stack_piece.color()))
            .take(stack.height as usize - 1)
        {
            let depth = stack.height - stack_index as u8 - 2;
            if depth < 12 {
                rows.adds.push(
                    &NET.ftw_stack
                        [stack_feature_idx(perspective, top, color, depth, perspective_sq)],
                );
            }
        }
    }

    for (&color, reserves) in [Color::White, Color::Black]
        .iter()
        .zip([pos.white_reserves_left(), pos.black_reserves_left()])
    {
        if reserves == 0 {
            continue;
        }

        rows.adds
            .push(&NET.ftw_reserves[reserves_feature_idx(perspective, color, reserves)]);
    }

    rows.apply(&mut acc, &NET.ftb);

    acc
}

#[must_use]
fn top_feature_idx(perspective: Color, side: Color, pt: Role, sq: Square) -> usize {
    (usize::from(side != perspective) * PIECE_TYPE_COUNT + (pt as usize)) * SQUARE_COUNT + sq.idx()
}

#[must_use]
fn stack_feature_idx(perspective: Color, top: Piece, side: Color, depth: u8, sq: Square) -> usize {
    debug_assert!(depth < 12);
    let top_idx = usize::from(top.color() != perspective) * PIECE_TYPE_COUNT + top.role() as usize;
    ((top_idx * COLOR_COUNT + usize::from(side != perspective)) * SQUARE_COUNT + sq.idx()) * 12
        + depth as usize
}

#[must_use]
fn reserves_feature_idx(perspective: Color, side: Color, reserves: u8) -> usize {
    usize::from(side != perspective) * 30 + (reserves as usize - 1)
}

#[must_use]
fn calc_output_bucket(pos: &Position<6>) -> usize {
    let w_reserves = pos.white_reserves_left();
    let b_reserves = pos.black_reserves_left();
    let min_reserves = w_reserves.min(b_reserves);
    let bucket = [
        0, 0, 1, 2, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7,
    ][min_reserves as usize];
    let stm_komi = i32::from(pos.komi().half_komi())
        * match pos.side_to_move() {
            Color::White => 1,
            Color::Black => -1,
        };
    let komi_bucket = (-stm_komi - i32::from(MIN_HALF_KOMI)) as usize;
    komi_bucket * RESERVE_BUCKET_COUNT + bucket
}

#[cfg(target_feature = "avx512f")]
const UPDATE_CHUNK: usize = L1_SIZE / 4;
#[cfg(not(target_feature = "avx512f"))]
const UPDATE_CHUNK: usize = 256;
const MAX_UPDATE_ROWS: usize = 128;

#[derive(Default)]
struct Updates {
    adds: ArrayVec<&'static [i16; L1_SIZE], MAX_UPDATE_ROWS>,
    subs: ArrayVec<&'static [i16; L1_SIZE], MAX_UPDATE_ROWS>,
}

impl Updates {
    fn apply(&self, dst: &mut [i16; L1_SIZE], src: &[i16; L1_SIZE]) {
        let paired = self.adds.len().min(self.subs.len());

        for offset in (0..L1_SIZE).step_by(UPDATE_CHUNK) {
            let mut chunk: [i16; UPDATE_CHUNK] =
                src[offset..offset + UPDATE_CHUNK].try_into().unwrap();

            for (&add, &sub) in self.adds.iter().zip(self.subs.iter()) {
                for ((v, add), sub) in chunk.iter_mut().zip(&add[offset..]).zip(&sub[offset..]) {
                    *v += *add - *sub;
                }
            }

            for &add in &self.adds[paired..] {
                for (v, add) in chunk.iter_mut().zip(&add[offset..]) {
                    *v += *add;
                }
            }

            for &sub in &self.subs[paired..] {
                for (v, sub) in chunk.iter_mut().zip(&sub[offset..]) {
                    *v -= *sub;
                }
            }

            dst[offset..offset + UPDATE_CHUNK].copy_from_slice(&chunk);
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Transform(u8);

impl Transform {
    #[must_use]
    const fn identity() -> Self {
        Self(0)
    }

    #[must_use]
    const fn from_raw(raw: u8) -> Option<Self> {
        if raw < 8 {
            Some(Self(raw))
        } else {
            None
        }
    }

    #[must_use]
    const fn is_flip_rank(self) -> bool {
        self.0 & 1 != 0
    }

    #[must_use]
    const fn is_flip_file(self) -> bool {
        self.0 & 2 != 0
    }

    #[must_use]
    const fn is_flip_diag(self) -> bool {
        self.0 & 4 != 0
    }

    #[must_use]
    const fn idx(self) -> usize {
        self.0 as usize
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
#[repr(u8)]
#[rustfmt::skip]
#[allow(unused)]
enum Square {
    A1, B1, C1, D1, E1, F1,
    A2, B2, C2, D2, E2, F2,
    A3, B3, C3, D3, E3, F3,
    A4, B4, C4, D4, E4, F4,
    A5, B5, C5, D5, E5, F5,
    A6, B6, C6, D6, E6, F6,
}

impl Square {
    const COUNT: usize = 36;

    #[must_use]
    const fn from_raw(raw: u8) -> Option<Self> {
        if (raw as usize) < Self::COUNT {
            // SAFETY: we just bounds checked the value
            Some(unsafe { std::mem::transmute::<u8, Square>(raw) })
        } else {
            None
        }
    }

    #[must_use]
    const fn from_file_rank(file: u32, rank: u32) -> Option<Self> {
        if file >= 6 || rank >= 6 {
            None
        } else {
            Some(Self::from_raw((rank as u8 * 6) + file as u8).unwrap())
        }
    }

    #[must_use]
    const fn from_tiltak(sq: TiltakSquare<6>) -> Self {
        let rank = sq.rank() as u32;
        let file = sq.file() as u32;
        Self::from_file_rank(file, 5 - rank).unwrap()
    }

    #[must_use]
    const fn raw(self) -> u8 {
        self as u8
    }

    #[must_use]
    const fn idx(self) -> usize {
        self as usize
    }

    #[must_use]
    const fn rank(self) -> u32 {
        self.raw() as u32 / 6
    }

    #[must_use]
    const fn file(self) -> u32 {
        self.raw() as u32 % 6
    }

    #[must_use]
    const fn flip_rank(self) -> Self {
        let file = self.file();
        let rank = self.rank();
        Self::from_file_rank(file, 5 - rank).unwrap()
    }

    #[must_use]
    const fn flip_file(self) -> Self {
        let file = self.file();
        let rank = self.rank();
        Self::from_file_rank(5 - file, rank).unwrap()
    }

    #[must_use]
    const fn flip_diag(self) -> Self {
        let file = self.file();
        let rank = self.rank();
        Self::from_file_rank(rank, file).unwrap()
    }

    #[must_use]
    const fn transform(self, transform: Transform) -> Self {
        static LUT: [[Square; 8]; Square::COUNT] = {
            let mut lut = [[Square::A1; _]; _];
            let mut t = 0;
            while t < 8 {
                let transform = Transform::from_raw(t as u8).unwrap();
                let mut s = 0;
                while s < Square::COUNT {
                    let mut square = Square::from_raw(s as u8).unwrap();
                    if transform.is_flip_rank() {
                        square = square.flip_rank();
                    }
                    if transform.is_flip_file() {
                        square = square.flip_file();
                    }
                    if transform.is_flip_diag() {
                        square = square.flip_diag();
                    }
                    lut[s][t] = square;
                    s += 1;
                }
                t += 1;
            }
            lut
        };
        LUT[self.idx()][transform.idx()]
    }

    #[must_use]
    const fn normalizer(self) -> Transform {
        static LUT: [Transform; Square::COUNT] = {
            let mut lut = [Transform::identity(); _];
            let mut s = 0;
            while s < Square::COUNT {
                let square = Square::from_raw(s as u8).unwrap();
                let mut t = 0;
                loop {
                    let transform = Transform::from_raw(t).unwrap();
                    let new = square.transform(transform);
                    if new.rank() < 3 && new.file() < 3 && new.rank() <= new.file() {
                        lut[s] = transform;
                        break;
                    }
                    t += 1;
                }
                s += 1;
            }
            lut
        };
        LUT[self.idx()]
    }
}

#[must_use]
const fn normalizer(opt_square: Option<Square>) -> Transform {
    match opt_square {
        Some(square) => square,
        None => Square::A1,
    }
    .normalizer()
}
