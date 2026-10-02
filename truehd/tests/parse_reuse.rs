//! `Parser::parse_into` over one access unit kept for a whole stream, and from
//! one stream to the next, against `Parser::parse`.

use truehd::process::{decode::Decoder, extract::Extractor, parse::Parser};
use truehd::structs::access_unit::AccessUnit;
use truehd::utils::errors::ExtractError;

fn assert_same(kept: &AccessUnit, fresh: &AccessUnit) {
    assert_eq!(kept.check_nibble, fresh.check_nibble);
    assert_eq!(kept.access_unit_length, fresh.access_unit_length);
    assert_eq!(kept.input_timing, fresh.input_timing);
    assert_eq!(kept.has_valid_branch, fresh.has_valid_branch);
    assert_eq!(kept.has_invalid_branch, fresh.has_invalid_branch);
    assert_eq!(
        format!("{:?}", kept.major_sync_info),
        format!("{:?}", fresh.major_sync_info)
    );
    assert_eq!(
        format!("{:?}", kept.substream_directory),
        format!("{:?}", fresh.substream_directory)
    );
    // Blocks, sample rows and all.
    assert_eq!(
        format!("{:?}", kept.substream_segment),
        format!("{:?}", fresh.substream_segment)
    );
    assert_eq!(
        format!("{:?}", kept.extra_data),
        format!("{:?}", fresh.extra_data)
    );
}

/// One access unit, kept across streams of other channel counts, sampling
/// frequencies and block sizes and across the substreams asked for, is what a
/// new one would be, and decodes to the same.
#[test]
fn kept_access_unit_is_the_one_parse_builds() {
    let mut kept = AccessUnit::default();

    for data in [
        include_bytes!("assets/fba_atmos_cbi.mlp").as_slice(),
        include_bytes!("assets/fba_192k_8ch.mlp").as_slice(),
        include_bytes!("assets/fba_atmos_dimtrim.mlp").as_slice(),
        include_bytes!("assets/fba_spliced.mlp").as_slice(),
        include_bytes!("assets/fbb_6ch.mlp").as_slice(),
        include_bytes!("assets/fba_2ch.mlp").as_slice(),
    ] {
        // From the widest presentation down, so that the access unit holds the
        // segments of substreams the next pass skips; then each alone; then the
        // widest again, which leaves the next stream every segment this one has.
        let passes = (0..4)
            .rev()
            .map(|p| (p, false))
            .chain((0..4).map(|p| (p, true)))
            .chain([(3, false)]);

        for (presentation, alone) in passes {
            let mut required = [false; 4];
            if alone {
                required[presentation] = true;
            } else {
                required[..=presentation].fill(true);
            }

            let mut parser = Parser::default();
            let mut keeping_parser = Parser::default();
            parser.set_required_presentations(&required);
            keeping_parser.set_required_presentations(&required);
            let mut decoder = Decoder::default();
            let mut keeping_decoder = Decoder::default();

            let mut extractor = Extractor::default();
            extractor.push_bytes(data);
            let mut count = 0;

            for frame in extractor {
                let frame = match frame {
                    Ok(frame) => frame,
                    Err(ExtractError::InsufficientData) => break,
                    Err(error) => panic!("{error}"),
                };
                let fresh = parser.parse(&frame).unwrap();

                keeping_parser.parse_into(&frame, &mut kept).unwrap();
                assert_same(&kept, &fresh);

                let decoded = decoder.decode_presentation(&fresh, presentation).unwrap();
                let from_kept = keeping_decoder
                    .decode_presentation(&kept, presentation)
                    .unwrap();
                assert_eq!(format!("{from_kept:?}"), format!("{decoded:?}"));

                count += 1;
            }

            assert!(count > 50, "{count} access units");
        }
    }
}

/// A frame that does not parse leaves nothing of itself or of the frame before
/// in the access unit, which then takes the next one.
#[test]
fn failed_parse_leaves_a_default_access_unit() {
    let mut extractor = Extractor::default();
    extractor.push_bytes(include_bytes!("assets/fba_atmos_cbi.mlp"));
    let major = extractor.next().unwrap().unwrap();
    let mut minor = extractor.next().unwrap().unwrap();
    let continuation = extractor.next().unwrap().unwrap();

    let mut parser = Parser::default();
    let mut kept = parser.parse(&major).unwrap();

    // Its header, without the samples.
    minor.data = minor.data[..8].into();
    assert!(parser.parse_into(&minor, &mut kept).is_err());
    assert_same(&kept, &AccessUnit::default());

    // Refused before anything is read, for want of a major sync.
    parser.reset_for_next_major_sync();
    kept = Parser::default().parse(&major).unwrap();
    assert!(parser.parse_into(&continuation, &mut kept).is_err());
    assert_same(&kept, &AccessUnit::default());

    parser.parse_into(&major, &mut kept).unwrap();
    assert_same(&kept, &Parser::default().parse(&major).unwrap());
}
