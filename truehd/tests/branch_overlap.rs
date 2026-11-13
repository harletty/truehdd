//! `DecodedAccessUnit::overlaps_previous`: an access unit whose restart header
//! dates it at the output time of the access unit before it.
//!
//! Blu-ray titles that branch between clips end each clip's stream one access
//! unit past the splice, and the next clip's first access unit restates that
//! same output time. The fixtures here are no such title, so the overlap is
//! built by sending a major-sync access unit twice: the copy restates the
//! output time of the access unit just decoded, which is the condition the
//! flag reports, with samples that match bit for bit (so `is_duplicate` holds
//! too). What the flag adds over `is_duplicate`, an overlap whose samples
//! differ, is not built here: it takes two encodes of the same time.

use truehd::process::decode::{DecodedAccessUnit, Decoder};
use truehd::process::extract::Extractor;
use truehd::process::parse::Parser;

/// Access units 384..536 of an FBA encode with one two-channel substream; it
/// starts at a major sync and holds a second one.
const FBA_2CH_SLICE: &[u8] = include_bytes!("assets/fba_2ch.mlp");

/// FBA encodes spliced end to end, starting over at access units 44 and 344:
/// the stream after each splice restarts its output timing at zero.
const FBA_SPLICED: &[u8] = include_bytes!("assets/fba_spliced.mlp");

const MAJOR_SYNC_FBA: [u8; 4] = [0xF8, 0x72, 0x6F, 0xBA];

/// The access units of a stream that starts with one, by their length field.
fn access_units(data: &[u8]) -> Vec<&[u8]> {
    assert_eq!(
        data[4..8],
        MAJOR_SYNC_FBA,
        "the fixture starts at a major sync"
    );
    let mut units = Vec::new();
    let mut at = 0;
    while at + 4 <= data.len() {
        let length = (u16::from_be_bytes([data[at], data[at + 1]]) & 0x0FFF) as usize * 2;
        assert!(
            length >= 4 && at + length <= data.len(),
            "a whole access unit at {at}"
        );
        units.push(&data[at..at + length]);
        at += length;
    }
    units
}

fn is_major_sync(unit: &[u8]) -> bool {
    unit[4..8] == MAJOR_SYNC_FBA
}

/// Each access unit's flags, decoded through the kept-result path.
fn decode(data: &[u8], presentation: usize) -> Vec<(bool, bool)> {
    let mut extractor = Extractor::default();
    extractor.push_bytes(data);
    let mut parser = Parser::default();
    let mut decoder = Decoder::default();
    let mut decoded = DecodedAccessUnit::default();
    let mut flags = Vec::new();

    // The extractor answers the end of its input with an error, as the stream
    // might go on.
    for result in &mut extractor {
        let Ok(frame) = result else { break };
        let access_unit = parser.parse(&frame).expect("the stream parses");
        decoder
            .decode_presentation_into(&access_unit, presentation, &mut decoded)
            .expect("the stream decodes");
        flags.push((decoded.overlaps_previous, decoded.is_duplicate));
    }
    flags
}

#[test]
fn a_major_sync_sent_twice_overlaps_the_access_unit_before_it() {
    let units = access_units(FBA_2CH_SLICE);
    let second_sync = (1..units.len())
        .find(|&i| is_major_sync(units[i]))
        .expect("the slice holds a second major sync");

    // ..., unit[second_sync], unit[second_sync] again, unit[second_sync + 1], ...
    let mut stream = Vec::new();
    for unit in &units[..=second_sync] {
        stream.extend_from_slice(unit);
    }
    for unit in &units[second_sync..] {
        stream.extend_from_slice(unit);
    }

    let flags = decode(&stream, 1);
    let copy = second_sync + 1;
    assert!(flags.len() > copy + 1, "the decode runs past the copy");
    for (index, &(overlaps, duplicate)) in flags.iter().enumerate() {
        if index == copy {
            assert!(overlaps, "the copy restates the output time before it");
            assert!(duplicate, "and decodes to the same samples");
        } else {
            assert!(!overlaps, "access unit {index} follows on in time");
            assert!(!duplicate, "access unit {index} is no duplicate");
        }
    }
}

#[test]
fn a_stream_that_follows_on_overlaps_nothing() {
    for (index, (overlaps, duplicate)) in decode(FBA_2CH_SLICE, 1).into_iter().enumerate() {
        assert!(!overlaps && !duplicate, "access unit {index}");
    }
}

/// A splice whose next stream starts its output timing over somewhere else is
/// a jump in time, not an overlap of one access unit.
#[test]
fn a_splice_that_restarts_its_timing_overlaps_nothing() {
    let mut extractor = Extractor::default();
    extractor.push_bytes(FBA_SPLICED);
    let mut parser = Parser::default();
    let mut decoder = Decoder::default();
    let mut decoded = DecodedAccessUnit::default();
    let mut units = 0;

    for (index, result) in (&mut extractor).enumerate() {
        let Ok(frame) = result else { break };
        let Ok(access_unit) = parser.parse(&frame) else {
            break;
        };
        if decoder
            .decode_presentation_into(&access_unit, 0, &mut decoded)
            .is_err()
        {
            break;
        }
        assert!(!decoded.overlaps_previous, "access unit {index}");
        units += 1;
    }
    assert!(units > 344, "the decode runs past both splices");
}
