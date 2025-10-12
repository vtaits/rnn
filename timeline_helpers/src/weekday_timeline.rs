use chrono::{Datelike, NaiveDate, NaiveDateTime};
use rnn_core::{Partition, RegressResult};
use serde_derive::Deserialize;

use crate::{ComplexTimelineValue, Timeline};

const DEFAULT_FORMAT: &str = "%Y-%m-%d %H:%M:%S";

fn _first_date_of_month(year: i32, month: u32, weekday: chrono::Weekday) -> NaiveDate {
    let mut day = 1;
    loop {
        let date = NaiveDate::from_ymd_opt(year, month, day).unwrap();
        if date.weekday() == weekday {
            return date;
        }
        day += 1;
    }
}

#[derive(Clone, Deserialize)]
pub struct WeekdayTimelineConfig {
    pub format: Option<String>,
    pub correlate_only: Option<Vec<usize>>,
}

pub struct WeekdayTimeline {
    config: WeekdayTimelineConfig,
}

impl WeekdayTimeline {
    pub fn new(config: WeekdayTimelineConfig) -> Self {
        WeekdayTimeline { config }
    }
}

impl WeekdayTimeline {
    fn get_date_format(&self) -> &str {
        match &self.config.format {
            Some(f) => f,
            _ => DEFAULT_FORMAT,
        }
    }

    pub fn from_config(config: &WeekdayTimelineConfig) -> Self {
        WeekdayTimeline::new(config.clone())
    }
}

impl Timeline for WeekdayTimeline {
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
                    let mut res = vec![false; 7];

                    let bit_index = match date.weekday() {
                        chrono::Weekday::Mon => 0,
                        chrono::Weekday::Tue => 1,
                        chrono::Weekday::Wed => 2,
                        chrono::Weekday::Thu => 3,
                        chrono::Weekday::Fri => 4,
                        chrono::Weekday::Sat => 5,
                        chrono::Weekday::Sun => 6,
                    };

                    res[bit_index] = true;

                    res
                }
                _ => vec![false; *capacity as usize],
            };
        }

        panic!("Invalid value of weekday timeline");
    }

    fn reverse(&self, _bits: &[bool]) -> ComplexTimelineValue {
        return ComplexTimelineValue::Datetime(String::from(""));

        /*
        let format = self.get_date_format();

        let target_day = if bits[0] { chrono::Weekday::Wed } else { chrono::Weekday::Sun };

        let datetime = first_date_of_month(2025, 1, target_day);

        ComplexTimelineValue::Datetime(format!("{}", datetime.format(format))) */
    }

    fn get_capacity(&self) -> &u8 {
        &7u8
    }

    fn normalize_prediction(&self, bits: &[bool]) -> Vec<bool> {
        bits.to_vec()
    }

    fn get_partition(&self) -> Partition {
        Partition {
            size: *self.get_capacity() as usize,
            accept_all: false,
            correlate_only: self.config.correlate_only.clone(),
            no_correlate: None,
        }
    }

    fn regress(&self, _bits: &[bool]) -> f32 {
        panic!("Regression is not implemented for weekday timeline");
    }

    fn get_regress_difference(
        &self,
        _original: &ComplexTimelineValue,
        _computed: &ComplexTimelineValue,
    ) -> RegressResult {
        panic!("Regression is not implemented for weekday timeline");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn datetime_encode_and_decode() {
        let timeline = WeekdayTimeline::new(WeekdayTimelineConfig {
            format: None,
            correlate_only: None,
        });

        let cases: Vec<(&str, Vec<bool>)> = vec![
            (
                "2025-08-07 00:01:00",
                vec![false, false, false, true, false, false, false],
            ),
            (
                "2025-08-08 00:01:00",
                vec![false, false, false, false, true, false, false],
            ),
            (
                "2025-08-09 00:01:00",
                vec![false, false, false, false, false, true, false],
            ),
            (
                "2025-08-10 00:01:00",
                vec![false, false, false, false, false, false, true],
            ),
            (
                "2025-08-11 02:04:00",
                vec![true, false, false, false, false, false, false],
            ),
            (
                "2025-08-12 12:30:00",
                vec![false, true, false, false, false, false, false],
            ),
            (
                "2025-08-13 12:30:00",
                vec![false, false, true, false, false, false, false],
            ),
        ];

        for case in cases {
            let result = timeline.get_bits(&ComplexTimelineValue::Datetime(String::from(case.0)));

            assert_eq!(result, case.1);
        }
    }
}
