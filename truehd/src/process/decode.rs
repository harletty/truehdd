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

    /// Decodes an access unit to PCM audio samples, into a result that is
    /// already there.
    ///
    /// As [`decode_presentation`](Self::decode_presentation), which builds a new
    /// [`DecodedAccessUnit`] for every access unit: 10 KiB of samples, most of
    /// them past the end of the access unit, and two allocations. Here the
    /// samples of the access unit are written over those `decoded` holds and
    /// its vectors are reused, so one `DecodedAccessUnit`, which
    /// [`Default`] gives, serves a whole stream.
    ///
    /// Only the rows the access unit decodes are written. Whatever `decoded`
    /// held past them stays, where `decode_presentation` returns zeros.
    pub fn decode_presentation_into(
        &mut self,
        access_unit: &AccessUnit,
        presentation: usize,
        decoded: &mut DecodedAccessUnit,
    ) -> Result<()> {
        let mut required_presentations = [false; MAX_PRESENTATIONS];
        required_presentations[presentation] = true;

        self.state
            .decode_access_unit(access_unit, &required_presentations)?;

        let substream_info_changed = std::mem::take(&mut self.state.substream_info_changed);

        match self.state.effective_presentations.iter().position(|&p| p) {
            Some(i) => {
                self.state
                    .fill_decoded_result(i, substream_info_changed, decoded);
                Ok(())
            }
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

impl Default for DecodedAccessUnit {
    /// No samples of no channels, as a result to decode into.
    fn default() -> Self {
        Self {
            sampling_frequency: 0,
            sample_length: 0,
            channel_count: 0,
            pcm_data: [[0; 16]; 160],
            channel_labels: vec![],
            oamd: vec![],
            is_duplicate: false,
            substream_info_changed: false,
        }
    }
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

    /// Where the filters of a block keep their states while it is decoded. Nothing
    /// in it outlives the block.
    recorrelator_states: Box<[LaneStates; STATE_BASE + 8]>,
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
            recorrelator_states: Box::new([[[0; LANES]; 2]; STATE_BASE + 8]),
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
            coeff_q,
            ..
        } = *self.substream_state()?;

        let samples_per_au = self.samples_per_au;
        let current_substream_index = self.substream_index;

        // A block holds the rows its own header announced. One read under another
        // block size than the decoder is at reads as zeros past its last row.
        let mut padded_data = None;
        let block_data = match block.block_data.get(..block_size) {
            Some(rows) => rows,
            None => &padded_data.insert(padded_rows(&block.block_data))[..block_size],
        };
        let mut padded_lsb = None;
        let bypassed_lsb = match block.bypassed_lsb.get(..block_size) {
            Some(rows) => rows,
            None => &padded_lsb.insert(padded_rows(&block.bypassed_lsb))[..block_size],
        };

        // recorrelation
        {
            let ss_state = &mut self.substream_state[current_substream_index];
            let decoded_sample_len = ss_state.decoded_sample_len;
            let coeff = &ss_state.coeff;
            let coeff_state = &mut ss_state.coeff_state;
            let rows = &mut ss_state.rematrix_buffer[decoded_sample_len..][..block_size];
            let wide = restart_sync_word == 0x31EC;

            let filter = |chi: usize| {
                // Zero past each filter's order (an order of zero leaves stale
                // coefficients behind), so both run at their full fixed length.
                let mut fir = [0i32; 8];
                for (tap, fir) in fir.iter_mut().enumerate() {
                    if tap < order[0][chi] {
                        *fir = coeff[0][chi][tap];
                    }
                }
                let mut iir = [0i32; 4];
                for (tap, iir) in iir.iter_mut().enumerate() {
                    if tap < order[1][chi] {
                        *iir = coeff[1][chi][tap];
                    }
                }

                Recorrelator {
                    fir,
                    iir,
                    shift: coeff_q[0][chi],
                    mask: !((1 << quantiser_step_size[chi]) - 1),
                }
            };

            let lanes = lanes_available();
            let mut chi = min_chan;

            while chi <= max_chan {
                if lanes {
                    let channels = (max_chan + 1 - chi).min(LANES);
                    let mut filters = [Recorrelator::default(); LANES];
                    for (lane, lane_filter) in filters[..channels].iter_mut().enumerate() {
                        *lane_filter = filter(chi + lane);
                    }

                    if recorrelate_lanes(
                        &filters[..channels],
                        coeff_state,
                        &mut self.recorrelator_states,
                        block_data,
                        rows,
                        chi,
                        wide,
                    ) {
                        chi += channels;
                        continue;
                    }
                }

                // Where the lanes are not to be had, or to find which channel of
                // them left its range, and at which sample.
                let [fir, iir] = &mut coeff_state.each_mut().map(|state| &mut state[chi]);
                recorrelate(&filter(chi), [fir, iir], block_data, rows, chi, wide)?;
                chi += 1;
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

    /// [`create_decoded_result`](Self::create_decoded_result) over a result that
    /// is already there, writing only the rows the access unit decoded.
    fn fill_decoded_result(
        &self,
        presentation_idx: usize,
        substream_info_changed: bool,
        decoded: &mut DecodedAccessUnit,
    ) {
        let ss_state = &self.substream_state[presentation_idx];
        let rows = self.samples_per_au.min(decoded.pcm_data.len());

        decoded.channel_labels.clone_from(&ss_state.channel_labels);
        decoded.sampling_frequency = self.sampling_frequency;
        decoded.sample_length = self.samples_per_au - ss_state.zero_samples;
        decoded.channel_count = ss_state.max_matrix_chan + 1;
        decoded.pcm_data[..rows].copy_from_slice(&ss_state.output_buffer[..rows]);
        decoded.oamd.clear();
        if presentation_idx == 3 && self.effective_presentations[3] {
            decoded.oamd.extend(self.oamd.iter().cloned());
        }
        decoded.is_duplicate = self.has_duplicate_timing && self.has_duplicate_sample;
        decoded.substream_info_changed = substream_info_changed;
    }
}

/// One channel's prediction filters, as a block runs them.
#[derive(Clone, Copy, Default)]
struct Recorrelator {
    fir: [i32; 8],
    iir: [i32; 4],
    shift: i32,
    mask: i64,
}

/// The longest block, which is how far back a filter's states are kept while one
/// is decoded: newest first, the block's own below the eight it started with.
const STATE_BASE: usize = 160;

/// Run one channel's filters over a block, from the residuals in `block_data` to
/// the samples in `rows`. `history` holds the last eight states of each filter,
/// newest first, and is brought up to date.
fn recorrelate(
    filter: &Recorrelator,
    history: [&mut [i32; 8]; 2],
    block_data: &[[i32; 16]],
    rows: &mut [[i32; 16]],
    chi: usize,
    wide: bool,
) -> Result<()> {
    let (max_val, min_val) = if wide {
        (1i64 << 31, -(1i64 << 31))
    } else {
        (1i64 << 23, -(1i64 << 23))
    };
    let block_size = block_data.len().min(STATE_BASE);
    let mut fir_states = [0; STATE_BASE + 8];
    let mut iir_states = [0; STATE_BASE + 8];

    fir_states[STATE_BASE..].copy_from_slice(history[0]);
    iir_states[STATE_BASE..].copy_from_slice(history[1]);

    for blki in 0..block_size {
        let audio_data = block_data[blki][chi] as i64;
        let state_base = STATE_BASE - blki;

        let acc = dot_fixed(&filter.fir, &fir_states[state_base..])
            + dot_fixed(&filter.iir, &iir_states[state_base..]);

        let pred = acc >> filter.shift;
        let fir_state = audio_data + (pred & filter.mask);
        let iir_state = fir_state - pred;

        if fir_state >= max_val {
            bail!(DecodeError::RecorrelatorPositiveSaturation(fir_state));
        } else if fir_state < min_val {
            bail!(DecodeError::RecorrelatorNegativeSaturation(fir_state));
        }

        if !(min_val..max_val).contains(&iir_state) {
            if wide {
                bail!(DecodeError::FilterBInputTooWide32(iir_state));
            } else {
                bail!(DecodeError::FilterBInputTooWide24(iir_state));
            }
        }

        fir_states[state_base - 1] = fir_state as i32;
        iir_states[state_base - 1] = iir_state as i32;

        rows[blki][chi] = fir_state as i32;
    }

    history[0].copy_from_slice(&fir_states[STATE_BASE - block_size..][..8]);
    history[1].copy_from_slice(&iir_states[STATE_BASE - block_size..][..8]);

    Ok(())
}

/// How many channels [`recorrelate_lanes`] runs side by side.
const LANES: usize = 4;

/// The states of both filters of every lane at one sample.
type LaneStates = [[i64; LANES]; 2];

/// Whether [`recorrelate_lanes`] can run on this processor.
fn lanes_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// [`recorrelate`] for the channels `chi..chi + filters.len()`, at most [`LANES`]
/// of them, side by side. `false` where the processor cannot, and where a state
/// of any of them left its range: the rows are then not to be used, and the
/// caller runs the channels one at a time, which reports the first.
///
/// One channel alone is held to twelve multiplications a sample, one after the
/// other. Side by side, each multiplication is one instruction for all the lanes.
#[allow(unused_variables)]
fn recorrelate_lanes(
    filters: &[Recorrelator],
    coeff_state: &mut [[[i32; 8]; 16]; 2],
    states: &mut [LaneStates; STATE_BASE + 8],
    block_data: &[[i32; 16]],
    rows: &mut [[i32; 16]],
    chi: usize,
    wide: bool,
) -> bool {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: the processor has AVX2, which is all the function requires.
        return unsafe {
            avx2::recorrelate_lanes(filters, coeff_state, states, block_data, rows, chi, wide)
        };
    }

    false
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::{LANES, LaneStates, Recorrelator, STATE_BASE};
    use std::arch::x86_64::*;

    #[inline(always)]
    fn load(lanes: &[i64; LANES]) -> __m256i {
        // SAFETY: an array of four `i64` is 32 readable bytes.
        unsafe { _mm256_loadu_si256(lanes.as_ptr().cast()) }
    }

    #[inline(always)]
    fn store(lanes: &mut [i64; LANES], value: __m256i) {
        // SAFETY: an array of four `i64` is 32 writable bytes.
        unsafe { _mm256_storeu_si256(lanes.as_mut_ptr().cast(), value) }
    }

    #[inline(always)]
    fn load_half(lanes: &[i32; LANES]) -> __m128i {
        // SAFETY: an array of four `i32` is 16 readable bytes.
        unsafe { _mm_loadu_si128(lanes.as_ptr().cast()) }
    }

    #[inline(always)]
    fn store_half(lanes: &mut [i32; LANES], value: __m128i) {
        // SAFETY: an array of four `i32` is 16 writable bytes.
        unsafe { _mm_storeu_si128(lanes.as_mut_ptr().cast(), value) }
    }

    /// See [`super::recorrelate_lanes`]. Lane `l` is channel `first + l`, where
    /// `first` is `chi` unless that would run past the sixteenth channel. A lane
    /// outside `chi..chi + filters.len()` has zero coefficients and takes zero
    /// residuals, so its states stay zero, and it leaves its channel alone.
    #[target_feature(enable = "avx2")]
    pub(super) fn recorrelate_lanes(
        filters: &[Recorrelator],
        coeff_state: &mut [[[i32; 8]; 16]; 2],
        states: &mut [LaneStates; STATE_BASE + 8],
        block_data: &[[i32; 16]],
        rows: &mut [[i32; 16]],
        chi: usize,
        wide: bool,
    ) -> bool {
        let block_size = block_data.len().min(rows.len()).min(STATE_BASE);
        let first = chi.min(16 - LANES);
        let lane_of = |channel: usize| channel - first;

        if chi + filters.len() > first + LANES {
            return false;
        }

        let mut fir = [[0i64; LANES]; 8];
        let mut iir = [[0i64; LANES]; 4];
        let mut shift = [0i64; LANES];
        let mut mask = [0i64; LANES];
        let mut live = [0i32; LANES];

        for (channel, filter) in (chi..).zip(filters) {
            let lane = lane_of(channel);

            for (tap, coeff) in filter.fir.iter().enumerate() {
                fir[tap][lane] = *coeff as i64;
            }
            for (tap, coeff) in filter.iir.iter().enumerate() {
                iir[tap][lane] = *coeff as i64;
            }
            // As a scalar shift takes its count.
            shift[lane] = (filter.shift & 63) as i64;
            mask[lane] = filter.mask;
            live[lane] = -1;
        }

        for (tap, states) in states[STATE_BASE..].iter_mut().enumerate() {
            *states = [[0; LANES]; 2];

            for channel in chi..chi + filters.len() {
                states[0][lane_of(channel)] = coeff_state[0][channel][tap] as i64;
                states[1][lane_of(channel)] = coeff_state[1][channel][tap] as i64;
            }
        }

        let mut fir_lanes = [_mm256_setzero_si256(); 8];
        for (lanes, coeff) in fir_lanes.iter_mut().zip(&fir) {
            *lanes = load(coeff);
        }
        let mut iir_lanes = [_mm256_setzero_si256(); 4];
        for (lanes, coeff) in iir_lanes.iter_mut().zip(&iir) {
            *lanes = load(coeff);
        }
        let (fir, iir) = (fir_lanes, iir_lanes);
        // The count is below 64, which the shift has no way to know.
        let shift = _mm256_and_si256(load(&shift), _mm256_set1_epi64x(63));
        let mask = load(&mask);
        let live_half = load_half(&live);
        let live = _mm256_cvtepi32_epi64(live_half);
        // An arithmetic shift is a logical one of the value moved up by 2^63,
        // moved back down by 2^63 shifted as much. The sum of the older
        // products, which is ready early, carries the move up.
        let sign = _mm256_set1_epi64x(i64::MIN);
        let shifted_sign = _mm256_srlv_epi64(sign, shift);
        // A state in range, moved up by half the range, has no bit at or above
        // the width of the range.
        let (bias, bits) = if wide {
            (1i64 << 31, 32)
        } else {
            (1i64 << 23, 24)
        };
        let bias = _mm256_set1_epi64x(bias);
        let bits = _mm_cvtsi32_si128(bits);
        // The low halves of the four lanes, side by side.
        let low_halves = _mm256_setr_epi32(0, 2, 4, 6, 0, 2, 4, 6);

        // The products of every state but the newest, oldest first: each sample
        // has them ready for the next one, so that all that waits on a sample
        // is its own two products.
        let older_products = |window: &[LaneStates]| -> __m256i {
            let window: &[LaneStates; 8] = window[..8].try_into().expect("eight states");
            let mut acc = sign;

            for tap in (1..8).rev() {
                acc = _mm256_add_epi64(acc, _mm256_mul_epi32(fir[tap], load(&window[tap][0])));
            }
            for tap in (1..4).rev() {
                acc = _mm256_add_epi64(acc, _mm256_mul_epi32(iir[tap], load(&window[tap][1])));
            }

            acc
        };

        let mut older = older_products(&states[STATE_BASE..]);
        let mut newest_fir = load(&states[STATE_BASE][0]);
        let mut newest_iir = load(&states[STATE_BASE][1]);
        let mut out_of_range = _mm256_setzero_si256();

        for blki in 0..block_size {
            let state_base = STATE_BASE - blki;
            let data: &[i32; LANES] = block_data[blki][first..first + LANES]
                .try_into()
                .expect("four channels");
            let row: &mut [i32; LANES] = (&mut rows[blki][first..first + LANES])
                .try_into()
                .expect("four channels");

            let acc = _mm256_add_epi64(
                _mm256_add_epi64(older, _mm256_mul_epi32(fir[0], newest_fir)),
                _mm256_mul_epi32(iir[0], newest_iir),
            );
            let pred = _mm256_sub_epi64(_mm256_srlv_epi64(acc, shift), shifted_sign);
            let residual = _mm256_and_si256(_mm256_cvtepi32_epi64(load_half(data)), live);
            let fir_state = _mm256_add_epi64(residual, _mm256_and_si256(pred, mask));
            let iir_state = _mm256_sub_epi64(fir_state, pred);

            out_of_range = _mm256_or_si256(
                out_of_range,
                _mm256_srl_epi64(
                    _mm256_or_si256(
                        _mm256_add_epi64(fir_state, bias),
                        _mm256_add_epi64(iir_state, bias),
                    ),
                    bits,
                ),
            );

            store(&mut states[state_base - 1][0], fir_state);
            store(&mut states[state_base - 1][1], iir_state);

            let samples =
                _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(fir_state, low_halves));
            store_half(row, _mm_blendv_epi8(load_half(row), samples, live_half));

            older = older_products(&states[state_base - 1..]);
            newest_fir = fir_state;
            newest_iir = iir_state;
        }

        if _mm256_testz_si256(out_of_range, out_of_range) == 0 {
            return false;
        }

        for (tap, states) in states[STATE_BASE - block_size..][..8].iter().enumerate() {
            for channel in chi..chi + filters.len() {
                coeff_state[0][channel][tap] = states[0][lane_of(channel)] as i32;
                coeff_state[1][channel][tap] = states[1][lane_of(channel)] as i32;
            }
        }

        true
    }
}

/// Sum of the products of `a` with the first `N` values of `b`, in 64 bits. A
/// length known at compile time unrolls into straight multiply-adds, where a
/// run-time one pays a loop, or a vector loop's set-up, for a handful of terms.
#[inline(always)]
fn dot_fixed<const N: usize>(a: &[i32; N], b: &[i32]) -> i64 {
    let b: &[i32; N] = b[..N].try_into().unwrap();
    let mut acc = 0i64;
    for i in 0..N {
        acc += a[i] as i64 * b[i] as i64;
    }
    acc
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
    bypassed_lsb: &[[u8; 16]],
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

        row[dest] = (((acc >> 18) as i32) & mask) + bypassed_lsb[pmi] as i32;
    }
}

/// The rows of a block followed by zeros, up to the longest block there is.
#[cold]
fn padded_rows<T: Copy + Default>(rows: &[[T; 16]]) -> Box<[[T; 16]; 160]> {
    let mut padded = Box::new([[T::default(); 16]; 160]);
    let len = rows.len().min(160);

    padded[..len].copy_from_slice(&rows[..len]);
    padded
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The lanes leave the rows and the filter states that one channel at a time
    /// leaves, and refuse the blocks in which a state leaves its range.
    #[test]
    fn lanes_match_one_channel_at_a_time() {
        if !lanes_available() {
            return;
        }

        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut random = |bits: u32| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed as i64 >> (64 - bits)) as i32
        };
        let mut states = Box::new([[[0; LANES]; 2]; STATE_BASE + 8]);
        let (mut in_range, mut out_of_range) = (0, 0);

        for round in 0..400usize {
            let wide = round % 5 == 0;
            let channels = 1 + round % LANES;
            let chi = [0, 3, 12, 16 - channels][round / LANES % 4];
            let block_size = 8 + (random(16) as usize & 0xFFFF) % 153;
            // Mostly residuals a stable filter keeps in range, sometimes not.
            let residual_bits = if round % 7 == 0 { 24 } else { 16 };

            let filters: Vec<Recorrelator> = (0..channels)
                .map(|_| Recorrelator {
                    fir: core::array::from_fn(|_| random(11)),
                    iir: core::array::from_fn(|_| random(10)),
                    shift: 8 + (random(4) & 7),
                    mask: !((1 << (random(3) & 3)) - 1),
                })
                .collect();
            let block_data: Vec<[i32; 16]> = (0..block_size)
                .map(|_| core::array::from_fn(|_| random(residual_bits)))
                .collect();
            let coeff_state: [[[i32; 8]; 16]; 2] = core::array::from_fn(|_| {
                core::array::from_fn(|_| core::array::from_fn(|_| random(20)))
            });

            let mut one_state = coeff_state;
            let mut one_rows = vec![[0x5a5a5a5a; 16]; block_size];
            let one_at_a_time = filters.iter().zip(chi..).all(|(filter, chi)| {
                let [fir, iir] = &mut one_state.each_mut().map(|state| &mut state[chi]);

                recorrelate(filter, [fir, iir], &block_data, &mut one_rows, chi, wide).is_ok()
            });

            let mut lane_state = coeff_state;
            let mut lane_rows = vec![[0x5a5a5a5a; 16]; block_size];
            let side_by_side = recorrelate_lanes(
                &filters,
                &mut lane_state,
                &mut states,
                &block_data,
                &mut lane_rows,
                chi,
                wide,
            );

            assert_eq!(side_by_side, one_at_a_time, "round {round}");
            if side_by_side {
                assert_eq!(lane_rows, one_rows, "round {round}");
                assert_eq!(lane_state, one_state, "round {round}");
                in_range += 1;
            } else {
                out_of_range += 1;
            }
        }

        assert!(
            in_range > 100 && out_of_range > 10,
            "{in_range} {out_of_range}"
        );
    }
}
