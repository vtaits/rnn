use chrono::{Datelike, NaiveDate, NaiveDateTime};
use serde_derive::Deserialize;

use crate::{bits_to_number, ComplexTimelineValue, Timeline};

const DEFAULT_FORMAT: &str = "%Y-%m-%d %H:%M:%S";

#[derive(Clone, Deserialize)]
pub struct WeekendTimelineConfig {
    pub format: Option<String>,
}

pub struct WeekendTimeline {
    config: WeekendTimelineConfig,
}

impl WeekendTimeline {
    pub fn new(config: WeekendTimelineConfig) -> Self {
        WeekendTimeline { config }
    }
}

impl WeekendTimeline {
    fn get_date_format(&self) -> &str {
        match &self.config.format {
            Some(f) => f,
            _ => DEFAULT_FORMAT,
        }
    }

    pub fn from_config(config: &WeekendTimelineConfig) -> Self {
        WeekendTimeline::new(config.clone())
    }
}

impl Timeline for WeekendTimeline {
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
                    let mut res = vec![false; 2];

                    let bit_index = match date.weekday() {
                        chrono::Weekday::Sat => 1,
                        chrono::Weekday::Sun => 1,
                        _ => 0,
                    };

                    res[bit_index] = true;

                    res
                }
                _ => vec![false; *capacity as usize],
            };
        }

        panic!("Invalid value of weekend timeline");
    }

    fn reverse(&self, bits: &[bool]) -> ComplexTimelineValue {
        let format = self.get_date_format();

        let year = bits_to_number(&bits[0..8]) + 1900;
        let month = std::cmp::max(std::cmp::min(bits_to_number(&bits[8..12]), 12), 1);
        let day = std::cmp::max(std::cmp::min(bits_to_number(&bits[12..17]), 31), 1);
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

        ComplexTimelineValue::Datetime(format!("{}", datetime.format(format)))
    }

    fn get_capacity(&self) -> &u8 {
        &2u8
    }

    fn normalize_prediction(&self, bits: &[bool]) -> Vec<bool> {
        bits.to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn datetime_encode_and_decode() {
        let timeline = WeekendTimeline::new(WeekendTimelineConfig { format: None });

        let cases: Vec<(&str, Vec<bool>)> = vec![
            ("2025-08-07 00:01:00", vec![true, false]),
            ("2025-08-08 00:01:00", vec![true, false]),
            ("2025-08-09 00:01:00", vec![false, true]),
            ("2025-08-10 00:01:00", vec![false, true]),
            ("2025-08-11 02:04:00", vec![true, false]),
            ("2025-08-12 12:30:00", vec![true, false]),
            ("2025-08-13 12:30:00", vec![true, false]),
        ];

        for case in cases {
            let result = timeline.get_bits(&ComplexTimelineValue::Datetime(String::from(case.0)));

            assert_eq!(result, case.1);
        }
    }
}
