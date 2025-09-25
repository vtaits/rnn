use chrono::{NaiveDate, NaiveDateTime, Timelike};
use rnn_core::{Partition, RegressResult};
use serde_derive::Deserialize;

use crate::{bits_to_number, number_to_single_bit, ComplexTimelineValue, Timeline};

const DEFAULT_FORMAT: &str = "%Y-%m-%d %H:%M:%S";
const MINUTES_IN_DAY: usize = 24 * 60;

#[derive(Clone, Deserialize)]
pub struct TimeTimelineConfig {
    pub format: Option<String>,
    pub capacity: Option<u8>,
}

pub struct TimeTimeline {
    config: TimeTimelineConfig,
    capacity: u8,
}

impl TimeTimeline {
    pub fn new(config: TimeTimelineConfig) -> Self {
        let capacity = config.capacity.unwrap_or(12);

        TimeTimeline { config, capacity }
    }
}

impl TimeTimeline {
    fn get_date_format(&self) -> &str {
        match &self.config.format {
            Some(f) => f,
            _ => DEFAULT_FORMAT,
        }
    }

    pub fn from_config(config: &TimeTimelineConfig) -> Self {
        TimeTimeline::new(config.clone())
    }
}

impl Timeline for TimeTimeline {
    fn is_target(&self) -> bool {
        false
    }

    fn get_bits(&self, timeline_value: &ComplexTimelineValue) -> Vec<bool> {
        let capacity = self.get_capacity();

        if let ComplexTimelineValue::Datetime(date_str) = timeline_value {
            let format = self.get_date_format();

            let date_result = NaiveDateTime::parse_from_str(date_str, format);

            return match date_result {
                Ok(date) => {
                    let current_minute =
                        date.hour() as usize * 60 as usize + date.minute() as usize;

                    number_to_single_bit(current_minute, *capacity as usize, MINUTES_IN_DAY, true)
                }
                _ => vec![false; *capacity as usize],
            };
        }

        panic!("Invalid value of time timeline");
    }

    fn reverse(&self, bits: &[bool]) -> ComplexTimelineValue {
        return ComplexTimelineValue::Datetime(String::from(""));

        /*
        let format = self.get_date_format();

        let year = 2025;
        let month = 9;
        let day = 23;

        let normalized_value = single_bit_to_number(bits);

        let hour = std::cmp::min(bits_to_number(&bits[17..22]), 23);
        let minute = std::cmp::min(bits_to_number(&bits[22..24]) * 15, 59);

        let date_opt = NaiveDate::from_ymd_opt(year as i32, month as u32, day as u32);

        if date_opt.is_none() {
            return ComplexTimelineValue::Datetime(String::from(""));
        }

        let date_opt = date_opt.unwrap().and_hms_opt(hour as u32, minute as u32, 0);

        if date_opt.is_none() {
            return ComplexTimelineValue::Datetime(String::from(""));
        }

        let datetime = date_opt.unwrap();

        ComplexTimelineValue::Datetime(format!("{}", datetime.format(format))) */
    }

    fn get_capacity(&self) -> &u8 {
        &self.capacity
    }

    fn normalize_prediction(&self, bits: &[bool]) -> Vec<bool> {
        bits.to_vec()
    }

    fn get_partition(&self) -> Partition {
        Partition {
            size: *self.get_capacity() as usize,
            accept_all: false,
        }
    }

    fn regress(&self, _bits: &[bool]) -> f32 {
        panic!("Regression is not implemented for time timeline");
    }

    fn get_regress_difference(
        &self,
        _original: &ComplexTimelineValue,
        _computed: &ComplexTimelineValue,
    ) -> RegressResult {
        panic!("Regression is not implemented for time timeline");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn datetime_encode_and_decode() {
        let timeline = TimeTimeline::new(TimeTimelineConfig {
            format: None,
            capacity: None,
        });

        let cases: Vec<(&str, Vec<bool>)> = vec![
            (
                "2025-01-23 00:01:00",
                vec![
                    true, false, false, false, false, false, false, false, false, false, false,
                    false,
                ],
            ),
            (
                "2024-01-23 02:04:00",
                vec![
                    false, true, false, false, false, false, false, false, false, false, false,
                    false,
                ],
            ),
            (
                "1950-08-04 12:30:00",
                vec![
                    false, false, false, false, false, false, true, false, false, false, false,
                    false,
                ],
            ),
        ];

        for case in cases {
            let result = timeline.get_bits(&ComplexTimelineValue::Datetime(String::from(case.0)));

            assert_eq!(result, case.1);
        }
    }
}
