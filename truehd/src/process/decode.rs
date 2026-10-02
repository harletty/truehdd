use crate::log_or_err;
use crate::process::{MAX_PRESENTATIONS, PresentationMap, PresentationType};
use crate::structs::access_unit::AccessUnit;
use crate::structs::block::Block;
use crate::structs::channel::ChannelLabel;
use crate::structs::oamd::ObjectAudioMetadataPayload;
use crate::utils::dither::fill_dither_31eb;
use crate::utils::errors::DecodeError;
use anyhow::{Result, anyhow, bail};
use log::{info, trace};
use std::collections::VecDeque;

/// Call `$function::<N>` with `N` the number of terms the matrix has, so that the
/// products of a sample are straight multiply-adds rather than a loop.
macro_rules! by_term_count {
    ($terms:expr, $function:ident $args:tt) => {
        by_term_count!(@arms $terms, $function, $args, 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15)
    };
    (@arms $terms:expr, $function:ident, $args:tt, $($n:literal)*) => {
        match $terms.count {
            $($n => $function::<$n> $args,)*
            _ => $function::<16> $args,
        }
    };
}

/// Decodes access units to PCM audio samples.
///
/// Converts parsed [`AccessUnit`] structures into 24-bit PCM audio data.
#[derive(Default)]
pub struct Decoder {
    state: DecoderState,
}

impl Decoder {
    /// Decodes an access unit to PCM audio samples.
    ///
    /// Returns a [`DecodedAccessUnit`] containing 24-bit PCM samples organized
    /// as `[sample_index][channel_index]` with up to 160 samples and 16 channels.
    ///
    /// This method now internally uses the multi-presentation decode path.
    pub fn decode_presentation(
        &mut self,
        access_unit: &AccessUnit,
        presentation: usize,
    ) -> Result<DecodedAccessUnit> {
        // Create presentation array with only the requested presentation
        let mut required_presentations = [false; MAX_PRESENTATIONS];
        required_presentations[presentation] = true;

        // Not through `decode_presentations`: only the one result is built, rather
        // than an array of four boxed and the result cloned out of it.
        self.state
            .decode_access_unit(access_unit, &required_presentations)?;

        // Read the flag and reset it, ahead of the result that reports it: built
        // last, the result is built where it is returned.
        let substream_info_changed = std::mem::take(&mut self.state.substream_info_changed);

        match self.state.effective_presentations.iter().position(|&p| p) {
            Some(i) => self.state.create_decoded_result(i, substream_info_changed),
            None => Err(anyhow!("Failed to get presentation {presentation}")),
        }
    }

    /// Decodes multiple presentations from an access unit.
    ///
    /// Takes a boolean array indicating which presentations are required
    /// and returns an array of optional decoded access units.
    pub fn decode_presentations(
        &mut self,
        access_unit: &AccessUnit,
        required_presentations: &[bool; MAX_PRESENTATIONS],
    ) -> Result<Box<[Option<DecodedAccessUnit>; MAX_PRESENTATIONS]>> {
        // Use the optimized shared decode method
        let result = self
            .state
            .decode_access_unit_presentations(access_unit, required_presentations)?;

        // Reset the flag after reading it
        self.state.substream_info_changed = false;

        Ok(Box::new(result))
    }

    /// Sets the failure level for validation errors.
    ///
    /// - `log::Level::Error`: Only fail on Error level messages (default)
    /// - `log::Level::Warn`: Fail on Warning level and above (strict mode)
    pub fn set_fail_level(&mut self, level: log::Level) {
        self.state.fail_level = level;
    }

    /// Resets stream state after a fatal decode failure.
    ///
    /// After a decode error, internal DSP state (filter memory, matrix
    /// coefficients, dither state) may be partially updated; continuing would
    /// produce corrupt audio. Calling this drops all stream state while
    /// preserving the fail level, so decoding can resume at the next access
    /// unit carrying a major sync.
    ///
    /// Must be called in lockstep with
    /// [`Parser::reset_for_next_major_sync`](crate::process::parse::Parser::reset_for_next_major_sync)
    /// at the same point in the frame sequence.
    pub fn reset_for_next_major_sync(&mut self) {
        self.state = DecoderState {
            fail_level: self.state.fail_level,
            ..Default::default()
        };
    }
}

/// The result of decoding an access unit to PCM audio.
///
/// Contains 24-bit signed integer samples in sample-major ordering
/// (`pcm_data[sample_index][channel_index]`) with associated metadata.
#[derive(Debug, Clone)]
pub struct DecodedAccessUnit {
    /// Sampling frequency in Hz.
    ///
    /// This is the sampling frequency used for the audio data.
    pub sampling_frequency: u32,

    /// Number of valid samples in this access unit.
    ///
    /// This indicates how many samples in the `pcm_data` array contain
    /// valid audio data. The remaining samples should be ignored.
    pub sample_length: usize,

    /// Channel count for the audio data.
    ///
    /// This is determined by the stream configuration and indicates how many
    /// channels are present in the audio data.
    pub channel_count: usize,

    /// PCM audio samples organized as `[sample_index][channel_index]`.
    ///
    /// Contains 24-bit signed integer samples with sample-major ordering.
    /// - Array dimensions: [160 samples][16 channels]
    /// - Valid data length: Determined by `sample_length`
    /// - Channel count: Determined by stream configuration
    pub pcm_data: [[i32; 16]; 160],

    /// Channel labels for the audio data.
    ///
    /// Contains labels for each channel in the audio data, providing
    /// descriptive names for each channel.
    pub channel_labels: Vec<ChannelLabel>,

    /// Optional object audio metadata payload.
    ///
    /// Contains spatial audio metadata when present in the stream.
    pub oamd: Vec<ObjectAudioMetadataPayload>,

    /// Indicates whether this access unit is a duplicate of the previous one.
    ///
    /// This is `true` when both the output timing and the decoded audio sample
    /// checksum match the previous access unit.
    /// Downstream applications may safely discard this frame.
    pub is_duplicate: bool,

    /// Indicates whether this access unit triggered a substream info change.
    ///
    /// This is `true` when substream_info or extended_substream_info changed,
    /// indicating that channel layout may have changed requiring new output files.
    pub substream_info_changed: bool,
}

#[derive(Debug, Clone)]
#[repr(C)]
pub struct DecoderSubstreamState {
    pub restart_sync_word: u16,
    pub output_timing: u16,
    pub max_bits: u8,
    /// Every output of the access unit folded into one word, sign removed, so the widest
    /// of them can be compared with `max_bits` once per substream.
    pub output_bits: u32,
    pub min_chan: usize,
    pub max_chan: usize,
    pub max_matrix_chan: usize,
    pub dither_shift: u32,
    pub dither_seed: u32,
    pub lossless_check_i32: i32,
    pub lossless_check_i32_prev_au: i32,
    pub lossless_check_i32_accum: i32,
    pub ch_assign: [usize; 16],

    pub block_size: usize,

    pub primitive_matrices: usize,
    pub matrix_ch: [u8; 16],
    pub frac_bits: [u8; 16],
    pub cf_shift_code: [i8; 16],
    pub dither_scale: [u8; 16],
    pub delta_precision: [u8; 16],
    pub delta_cf: [[i32; 16]; 16],
    pub m_coeff: [[i32; 16]; 16],

    pub output_shift: [i8; 16],
    pub quantiser_step_size: [u32; 16],

    pub order: [[usize; 16]; 2],
    pub coeff_q: [[i32; 16]; 2],
    pub coeff: [[[i32; 8]; 16]; 2],
    pub coeff_state: [[[i32; 8]; 16]; 2],

    pub dither_table: [i32; 256],
    pub decoded_sample_len: usize,

    pub rematrix_buffer: Box<[[i32; 16]; 160]>,
    pub output_buffer: Box<[[i32; 16]; 160]>,
    pub zero_samples: usize,
    pub channel_labels: Vec<ChannelLabel>,
}

impl Default for DecoderSubstreamState {
    fn default() -> Self {
        Self {
            restart_sync_word: 0,
            output_timing: 0,
            max_bits: 0,
            output_bits: 0,
            min_chan: 0,
            max_chan: 0,
            max_matrix_chan: 0,
            dither_shift: 0,
            dither_seed: 0,
            lossless_check_i32: 0,
            lossless_check_i32_prev_au: 0,
            lossless_check_i32_accum: 0,
            ch_assign: [0; 16],

            block_size: 8,

            primitive_matrices: 0,
            matrix_ch: [0; 16],
            frac_bits: [0; 16],
            cf_shift_code: [0; 16],
            dither_scale: [0; 16],
            delta_precision: [0; 16],
            delta_cf: [[0; 16]; 16],
            m_coeff: [[0; 16]; 16],

            output_shift: [0; 16],
            quantiser_step_size: [0; 16],

            order: [[0; 16]; 2],
            coeff_q: [[0; 16]; 2],
            coeff: [[[0; 8]; 16]; 2],
            coeff_state: [[[0; 8]; 16]; 2],

            dither_table: [0; 256],
            decoded_sample_len: 0,

            rematrix_buffer: Box::new([[0; 16]; 160]),
            output_buffer: Box::new([[0; 16]; 160]),
            zero_samples: 0,
            channel_labels: vec![],
        }
    }
}

/// Decode-time checks stay fail-fast; they report on samples rather than on bitstream
/// syntax, and a decode that continues past one produces wrong audio.
impl crate::utils::diagnostic::DiagnosticSink for DecoderState {
    fn fail_level(&self) -> log::Level {
        self.fail_level
    }
}

#[derive(Debug)]
#[repr(C)]
pub struct DecoderState {
    pub fail_level: log::Level,

    pub valid: bool,
    pub counter: usize,
    pub has_valid_branch: bool,
    pub has_invalid_branch: bool,
    pub has_duplicate_timing: bool,
    pub has_duplicate_sample: bool,

    pub sampling_frequency: u32,
    pub samples_per_au: usize,

    pub presentation_map: Option<PresentationMap>,
    pub presentation: usize,
    pub effective_presentations: [bool; MAX_PRESENTATIONS],

    pub channel_labels: Vec<ChannelLabel>,

    pub substreams: usize,
    pub substream_mask: u8,
    pub substream_info: u8,
    pub extended_substream_info: u8,

    pub substream_index: usize,
    pub substream_state: [DecoderSubstreamState; MAX_PRESENTATIONS],

    pub rematrix_buffer: [[i32; 16]; 160],
    pub output_buffer: [[i32; 16]; 160],
    pub zero_samples: usize,
    pub oamd: VecDeque<ObjectAudioMetadataPayload>,
    pub substream_info_changed: bool,
}

impl Default for DecoderState {
    fn default() -> Self {
        Self {
            fail_level: log::Level::Error,
            valid: false,
            counter: 0,
            has_valid_branch: false,
            has_invalid_branch: false,
            has_duplicate_timing: false,
            has_duplicate_sample: false,
            sampling_frequency: 0,
            samples_per_au: 0,
            presentation_map: None,
            presentation: 0,
            effective_presentations: core::array::from_fn(|_| false),
            channel_labels: vec![],
            substreams: 0,
            substream_mask: 0,
            substream_info: 0,
            extended_substream_info: 0,
            substream_index: 0,
            substream_state: core::array::from_fn(|_| DecoderSubstreamState::default()),
            rematrix_buffer: [[0; 16]; 160],
            output_buffer: [[0; 16]; 160],
            zero_samples: 0,
            oamd: VecDeque::with_capacity(4),
            substream_info_changed: false,
        }
    }
}

impl DecoderState {
    pub fn substream_state_mut(&mut self) -> Result<&mut DecoderSubstreamState> {
        Ok(&mut self.substream_state[self.substream_index])
    }

    pub fn substream_state(&self) -> Result<&DecoderSubstreamState> {
        Ok(&self.substream_state[self.substream_index])
    }

    pub fn decode_access_unit_presentations(
        &mut self,
        access_unit: &AccessUnit,
        required_presentations: &[bool; MAX_PRESENTATIONS],
    ) -> Result<[Option<DecodedAccessUnit>; MAX_PRESENTATIONS]> {
        self.decode_access_unit(access_unit, required_presentations)?;

        let mut result: [Option<DecodedAccessUnit>; MAX_PRESENTATIONS] =
            core::array::from_fn(|_| None);

        for (i, decoded) in result.iter_mut().enumerate() {
            if self.effective_presentations[i] {
                *decoded = Some(self.create_decoded_result(i, self.substream_info_changed)?);
            }
        }

        Ok(result)
    }

    /// Decode an access unit into the substreams' output buffers, without building
    /// any result from them.
    fn decode_access_unit(
        &mut self,
        access_unit: &AccessUnit,
        required_presentations: &[bool; MAX_PRESENTATIONS],
    ) -> Result<()> {
        access_unit.update_decoder_state(self)?;

        if !self.valid {
            self.update_presentations(required_presentations)?;

            for (i, &required) in self.effective_presentations.iter().enumerate() {
                if required {
                    self.substream_state[i].channel_labels =
                        access_unit.get_channel_labels(i).unwrap_or_default();
                }
            }
        }

        self.has_duplicate_timing = false;
        self.has_duplicate_sample = false;
        self.oamd.clear();

        if let Some(extra_data) = &access_unit.extra_data
            && let Some(evo_frame) = &extra_data.evo_frame
        {
            for evo_payload in &evo_frame.evo_payloads {
                if evo_payload.evo_payload_id == 11 {
                    let smploffst =
                        evo_payload.evo_payload_config.smploffst.unwrap_or_default() as u64;
                    let mut oamd = ObjectAudioMetadataPayload::read(&evo_payload.evo_payload_byte)?;
                    oamd.evo_sample_offset = smploffst;
                    self.oamd.push_back(oamd);
                }
            }
        }

        for i in 0..=self.presentation {
            if (self.substream_mask >> i) & 1 == 0 {
                continue;
            }

            let substream_segment = &access_unit.substream_segment[i];

            // Handle zero samples for required presentations
            if let Some(terminator) = &substream_segment.terminator
                && terminator.zero_samples_indicated
            {
                self.substream_state[i].zero_samples = terminator.zero_samples as usize;
            }

            self.substream_index = i;
            let ss_state = &mut self.substream_state[self.substream_index];
            ss_state.decoded_sample_len = 0;

            for block in substream_segment.block.iter() {
                block.update_decoder_state(self)?;
                self.decode_multi(block)?;
            }
        }

        self.valid = true;
        self.counter += 1;

        Ok(())
    }

    fn update_presentations(
        &mut self,
        required_presentations: &[bool; MAX_PRESENTATIONS],
    ) -> Result<()> {
        let Some(presentation_map) = self.presentation_map else {
            bail!("Presentation map not initialized");
        };

        self.substream_mask =
            presentation_map.substream_mask_by_required_presentations(required_presentations);

        self.effective_presentations =
            presentation_map.effective_presentations(required_presentations)?;

        // Find the highest required presentation for fallback logic
        let highest_required = required_presentations
            .iter()
            .enumerate()
            .rev()
            .find_map(|(i, &required)| if required { Some(i) } else { None })
            .unwrap_or(3);

        // Set presentation field for compatibility (using highest required)
        match presentation_map.presentation_type_by_index(highest_required) {
            PresentationType::Invalid => {
                if !self.valid {
                    let Some(max_independent) = presentation_map.max_independent_presentation()
                    else {
                        bail!("No presentation is available");
                    };
                    info!(
                        "Presentation {highest_required} is not available, using presentation {max_independent}"
                    );
                    self.presentation = max_independent;
                }
            }
            PresentationType::CopyOf(copy_index) => {
                if !self.valid {
                    info!("Presentation {highest_required} is a copy of presentation {copy_index}")
                }
                self.presentation = copy_index;
            }
            _ => {
                self.presentation = highest_required;
            }
        };

        Ok(())
    }

    pub fn reset_decoder_substream_state(&mut self) {
        let ss_state = &mut self.substream_state[self.substream_index];
        *ss_state = DecoderSubstreamState {
            lossless_check_i32_prev_au: ss_state.lossless_check_i32_prev_au,
            // The labels are only assigned when decoding starts or the layout
            // changes; a restart header must not erase them.
            channel_labels: std::mem::take(&mut ss_state.channel_labels),
            ..Default::default()
        }
    }

    fn decode_multi(&mut self, block: &Block) -> Result<()> {
        let DecoderSubstreamState {
            restart_sync_word,
            min_chan,
            max_chan,
            max_matrix_chan,
            dither_shift,
            // TODO: max_lsbs
            ch_assign,

            block_size,

            primitive_matrices,
            matrix_ch,
            dither_scale,

            output_shift,
            quantiser_step_size,

            order,
            coeff,
            coeff_q,
            ..
        } = *self.substream_state()?;

        let samples_per_au = self.samples_per_au;
        let current_substream_index = self.substream_index;

        let (max_val, min_val) = if restart_sync_word == 0x31EC {
            (1 << 31, -(1 << 31))
        } else {
            (1 << 23, -(1 << 23))
        };

        // recorrelation
        {
            let ss_state = &mut self.substream_state[current_substream_index];
            let decoded_sample_len = ss_state.decoded_sample_len;
            let coeff_state = &mut ss_state.coeff_state;
            let rematrix_buffer = &mut ss_state.rematrix_buffer[decoded_sample_len..];
            let block_data = &block.block_data;

            // Shared by the channels: each one reads only what it wrote itself or
            // loaded from its coefficient state, so it needs no clearing in between.
            let mut state_buffer = [[0; 168]; 2];

            #[allow(clippy::needless_range_loop)]
            for chi in min_chan..=max_chan {
                state_buffer[0][160..].copy_from_slice(&coeff_state[0][chi]);
                state_buffer[1][160..].copy_from_slice(&coeff_state[1][chi]);

                let fir_order = order[0][chi];
                let iir_order = order[1][chi];
                let coeff_q_shift = coeff_q[0][chi];
                let quantiser_mask = !((1 << quantiser_step_size[chi]) - 1);
                let fir_coeff = &coeff[0][chi];
                let iir_coeff = &coeff[1][chi];

                for blki in 0..block_size {
                    let audio_data = block_data[blki][chi] as i64;
                    let state_base = 160 - blki;

                    let mut acc = 0i64;

                    for oi in 0..fir_order {
                        acc += (fir_coeff[oi] as i64) * (state_buffer[0][state_base + oi] as i64);
                    }

                    for oi in 0..iir_order {
                        acc += (iir_coeff[oi] as i64) * (state_buffer[1][state_base + oi] as i64);
                    }

                    let pred = acc >> coeff_q_shift;
                    let fir_state = audio_data + (pred & quantiser_mask);
                    let iir_state = fir_state - pred;

                    if fir_state >= max_val {
                        bail!(DecodeError::RecorrelatorPositiveSaturation(fir_state));
                    } else if fir_state < min_val {
                        bail!(DecodeError::RecorrelatorNegativeSaturation(fir_state));
                    }

                    if !(min_val..max_val).contains(&iir_state) {
                        if restart_sync_word == 0x31EC {
                            bail!(DecodeError::FilterBInputTooWide32(iir_state));
                        } else {
                            bail!(DecodeError::FilterBInputTooWide24(iir_state));
                        }
                    }

                    state_buffer[0][159 - blki] = fir_state as i32;
                    state_buffer[1][159 - blki] = iir_state as i32;

                    rematrix_buffer[blki][chi] = fir_state as i32;
                }

                coeff_state[0][chi][..].copy_from_slice(&state_buffer[0][160 - block_size..][..8]);
                coeff_state[1][chi][..].copy_from_slice(&state_buffer[1][160 - block_size..][..8]);
            }
        }

        // lossless matrix
        if self.effective_presentations[current_substream_index] {
            let decoded_sample_len =
                self.substream_state[current_substream_index].decoded_sample_len;
            let block_rows = decoded_sample_len..decoded_sample_len + block_size;

            if let Some(presentation_map) = self.presentation_map {
                let substream_mask =
                    presentation_map.substream_mask_by_index(current_substream_index);

                for i in 0..=current_substream_index {
                    if (substream_mask >> i) & 1 == 0 {
                        continue;
                    }

                    let from = &self.substream_state[i];
                    let from_rows = &from.rematrix_buffer[block_rows.clone()];
                    let to_rows = &mut self.rematrix_buffer[block_rows.clone()];

                    // A channel at a time: a substream brings a few channels of the
                    // row, and a copy of that run is a call for a handful of bytes.
                    for chi in from.min_chan..=from.max_chan {
                        for (to, from) in to_rows.iter_mut().zip(from_rows) {
                            to[chi] = from[chi];
                        }
                    }
                }
            }

            let ss_state = &mut self.substream_state[current_substream_index];
            let rows = &mut self.rematrix_buffer[block_rows.clone()];
            let bypassed_lsb = &block.bypassed_lsb[..block_size];
            let matrix_terms = if restart_sync_word == 0x31EA {
                max_matrix_chan + 3
            } else {
                max_matrix_chan + 1
            };

            // A matrix at a time over the whole block, not a sample at a time through
            // every matrix: the samples are independent of each other, so one matrix's
            // products overlap from one sample to the next, where the matrices of one
            // sample each wait on the channel the one before wrote. It also settles
            // once per block which products a matrix has at all.
            match restart_sync_word {
                0x31EA => {
                    let dither_seed = &mut ss_state.dither_seed;

                    for row in rows.iter_mut() {
                        let dither_seed_shr7 = *dither_seed >> 7;

                        row[max_matrix_chan + 1] =
                            (((*dither_seed >> 15) as i8) << dither_shift) as i32;
                        row[max_matrix_chan + 2] =
                            ((dither_seed_shr7 as i8) << dither_shift) as i32;

                        *dither_seed =
                            (dither_seed_shr7 ^ (dither_seed_shr7 << 5) ^ (*dither_seed << 16))
                                & 0x7FFFFF;
                    }

                    #[allow(clippy::needless_range_loop)]
                    for pmi in 0..primitive_matrices {
                        let dest = matrix_ch[pmi] as usize;
                        let mask = !((1 << quantiser_step_size[dest]) - 1);
                        let terms = MatrixTerms::new(&ss_state.m_coeff[pmi][..matrix_terms], &[]);

                        by_term_count!(
                            terms,
                            apply_matrix(&terms, rows, bypassed_lsb, pmi, dest, mask, |_, _| 0)
                        );
                    }
                }
                0x31EB => {
                    let dither_index_mask = samples_per_au.next_power_of_two() - 1;

                    if decoded_sample_len == 0 {
                        fill_dither_31eb(
                            &mut ss_state.dither_table[..=dither_index_mask],
                            &mut ss_state.dither_seed,
                        );
                    }

                    let dither_table = &ss_state.dither_table;

                    for pmi in 0..primitive_matrices {
                        let dest = matrix_ch[pmi] as usize;
                        let mask = !((1 << quantiser_step_size[dest]) - 1);
                        let terms = MatrixTerms::new(&ss_state.m_coeff[pmi][..matrix_terms], &[]);
                        let dither_scale = dither_scale[pmi] as i64;

                        if dither_scale != 0 {
                            let step = primitive_matrices - pmi;
                            by_term_count!(
                                terms,
                                apply_matrix(
                                    &terms,
                                    rows,
                                    bypassed_lsb,
                                    pmi,
                                    dest,
                                    mask,
                                    |blki, _| {
                                        let blki_abs = blki + decoded_sample_len;
                                        let dither_index = step * (2 * blki_abs + 1) + blki_abs;

                                        (dither_table[dither_index & dither_index_mask] as i64)
                                            << (11 + dither_scale)
                                    }
                                )
                            );
                        } else {
                            by_term_count!(
                                terms,
                                apply_matrix(&terms, rows, bypassed_lsb, pmi, dest, mask, |_, _| 0)
                            );
                        }
                    }
                }
                0x31EC => {
                    let dither_index_mask = samples_per_au.next_power_of_two() - 1;

                    if decoded_sample_len == 0 {
                        fill_dither_31eb(
                            &mut ss_state.dither_table[..=dither_index_mask],
                            &mut ss_state.dither_seed,
                        );
                    }

                    let dither_table = &ss_state.dither_table;
                    let samples_per_au_recip = (1 << 16) / samples_per_au as i64;

                    for pmi in 0..primitive_matrices {
                        let dest = matrix_ch[pmi] as usize;
                        let mask = !((1 << quantiser_step_size[dest]) - 1);
                        let terms = MatrixTerms::new(
                            &ss_state.m_coeff[pmi][..matrix_terms],
                            &ss_state.delta_cf[pmi][..matrix_terms],
                        );
                        let dither_scale = dither_scale[pmi] as u64;
                        let step = primitive_matrices - pmi;

                        by_term_count!(
                            terms,
                            apply_matrix(
                                &terms,
                                rows,
                                bypassed_lsb,
                                pmi,
                                dest,
                                mask,
                                |blki, acc_delta| {
                                    let blki_abs = blki + decoded_sample_len;
                                    let mut acc = 0;

                                    if dither_scale != 0 {
                                        let dither_index = step * (2 * blki_abs + 1) + blki_abs;

                                        acc += (dither_table[dither_index & dither_index_mask]
                                            as i64)
                                            << (11 + dither_scale);
                                    }

                                    acc + (acc_delta >> 18)
                                        * (blki_abs as i64)
                                        * (samples_per_au_recip << 2)
                                }
                            )
                        );
                    }

                    if decoded_sample_len + block_size == samples_per_au {
                        for pmi in 0..primitive_matrices {
                            let m_coeff = &mut ss_state.m_coeff[pmi];
                            let delta_cf = &ss_state.delta_cf[pmi];
                            for chi in 0..=max_matrix_chan {
                                m_coeff[chi] += delta_cf[chi];
                            }
                        }
                    }
                }
                _ => {}
            }

            // remap
            {
                let output_rows = &mut ss_state.output_buffer[block_rows];

                if decoded_sample_len == 0 {
                    ss_state.lossless_check_i32 = 0;
                    ss_state.output_bits = 0;
                }

                let mut lossless_check_data = 0;
                let mut output_bits = 0u32;

                // A channel at a time, so that its shift and its place are settled
                // once. The rows need no clearing: the channels written are the same
                // from one restart header to the next, which starts from zeroed rows.
                for chi in 0..max_matrix_chan + 1 {
                    let ch_assign = ch_assign[chi];
                    let output_shift = output_shift[chi];

                    for (row, output_row) in rows.iter().zip(output_rows.iter_mut()) {
                        let mut output = row[chi];

                        if output_shift < 0 {
                            output >>= -output_shift;
                        } else {
                            output <<= output_shift;
                        }

                        output_row[ch_assign] = output;

                        lossless_check_data ^= (output & 0xFFFFFF) << (chi & 7);
                        output_bits |= (output ^ (output >> 31)) as u32;
                    }
                }

                ss_state.output_bits |= output_bits;
                ss_state.lossless_check_i32 ^= lossless_check_data;
                ss_state.lossless_check_i32_accum ^= lossless_check_data;

                if decoded_sample_len + block_size == samples_per_au {
                    trace!(
                        "AU {}: lossless_check_i32: {:08X}, lossless_check_i32_prev_au {:08X}",
                        self.counter,
                        ss_state.lossless_check_i32,
                        ss_state.lossless_check_i32_prev_au
                    );

                    if self.has_duplicate_timing
                        && ss_state.lossless_check_i32 == ss_state.lossless_check_i32_prev_au
                    {
                        self.has_duplicate_sample = true;
                        info!(
                            "AU {}: duplicate samples at branch, should be discarded",
                            self.counter
                        );
                    }

                    ss_state.lossless_check_i32_prev_au = ss_state.lossless_check_i32;
                }
            }
        }

        // The restart header states how many bits the substream's outputs use, sign apart,
        // and the widest of them is only known once the access unit is complete. It is a
        // property of the decoded samples, so it cannot be checked while parsing, and a
        // per-block check would report the same violation up to four times over.
        let ss_state = self.substream_state()?;

        if self.effective_presentations[current_substream_index]
            && ss_state.decoded_sample_len + block_size == samples_per_au
            && ss_state.output_bits >> ss_state.max_bits != 0
        {
            let max_bits = ss_state.max_bits;

            log_or_err!(
                self,
                log::Level::Warn,
                anyhow!(DecodeError::OutputsExceedMaxBits {
                    substream: current_substream_index,
                    max_bits,
                })
            );
        }

        self.substream_state_mut()?.decoded_sample_len += block_size;

        Ok(())
    }

    fn create_decoded_result(
        &self,
        presentation_idx: usize,
        substream_info_changed: bool,
    ) -> Result<DecodedAccessUnit> {
        let decoded = DecodedAccessUnit {
            channel_labels: self.substream_state[presentation_idx]
                .channel_labels
                .clone(),
            sampling_frequency: self.sampling_frequency,
            sample_length: self.samples_per_au
                - self.substream_state[presentation_idx].zero_samples,
            channel_count: self.substream_state[presentation_idx].max_matrix_chan + 1,
            pcm_data: *self.substream_state[presentation_idx].output_buffer,
            oamd: if presentation_idx == 3 && self.effective_presentations[3] {
                self.oamd.iter().cloned().collect::<Vec<_>>()
            } else {
                vec![]
            },
            is_duplicate: self.has_duplicate_timing && self.has_duplicate_sample,
            substream_info_changed,
        };

        Ok(decoded)
    }
}

/// The products of one primitive matrix that are not zero: which channels of the
/// sample row it reads, and with which coefficients. A matrix of a 7.1 substream
/// often reads half of the eight channels, and one that only rescales a channel
/// reads one.
struct MatrixTerms {
    count: usize,
    channel: [u8; 16],
    coeff: [i32; 16],
    /// The coefficient deltas of the same channels, for the substream that
    /// interpolates its matrices.
    delta: [i32; 16],
}

impl MatrixTerms {
    /// From a matrix's coefficients and, where it has them, its deltas; a channel
    /// is kept when either is not zero.
    fn new(coeff: &[i32], delta: &[i32]) -> Self {
        let mut terms = Self {
            count: 0,
            channel: [0; 16],
            coeff: [0; 16],
            delta: [0; 16],
        };

        for (chi, &coeff) in coeff.iter().enumerate() {
            let delta = delta.get(chi).copied().unwrap_or(0);

            if coeff != 0 || delta != 0 {
                terms.channel[terms.count] = chi as u8;
                terms.coeff[terms.count] = coeff;
                terms.delta[terms.count] = delta;
                terms.count += 1;
            }
        }

        terms
    }
}

/// Apply one primitive matrix to every sample row of a block: channel `dest`
/// becomes the sum of the matrix's `N` products and of `extra`, which is given
/// the row's index in the block and the sum of its products with the deltas.
#[inline(always)]
fn apply_matrix<const N: usize>(
    terms: &MatrixTerms,
    rows: &mut [[i32; 16]],
    bypassed_lsb: &[[i32; 16]],
    pmi: usize,
    dest: usize,
    mask: i32,
    extra: impl Fn(usize, i64) -> i64,
) {
    let channel: &[u8; N] = terms.channel[..N].try_into().unwrap();
    let coeff: &[i32; N] = terms.coeff[..N].try_into().unwrap();
    let delta: &[i32; N] = terms.delta[..N].try_into().unwrap();

    for (blki, (row, bypassed_lsb)) in rows.iter_mut().zip(bypassed_lsb).enumerate() {
        let mut acc = 0i64;
        let mut acc_delta = 0i64;

        for i in 0..N {
            let sample = row[(channel[i] & 15) as usize] as i64;

            acc += coeff[i] as i64 * sample;
            acc_delta += delta[i] as i64 * sample;
        }

        acc += extra(blki, acc_delta);

        row[dest] = (((acc >> 18) as i32) & mask) + bypassed_lsb[pmi];
    }
}
