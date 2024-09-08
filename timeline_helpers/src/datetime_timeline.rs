use chrono::{Datelike, NaiveDate, NaiveDateTime, Timelike};
use serde_derive::Deserialize;

use crate::{bits_to_number, number_to_bits, ComplexTimelineValue, Timeline};

const DEFAULT_FORMAT: &str = "%Y-%m-%d %H:%M:%S";
// 8 (year) + 4 (month) + 5 (day) + 5 (hour) + 2 (minute) 
const CAPACITY: usize = 24;

#[derive(Clone, Deserialize)]
pub struct DatetimeTimelineConfig {
    pub format: Option<String>,
}

pub struct DatetimeTimeline {
  config: DatetimeTimelineConfig,
}

impl DatetimeTimeline {
    pub fn new(config: DatetimeTimelineConfig) -> Self {
        DatetimeTimeline {
          config,
        }
    }
}

impl DatetimeTimeline {
  fn get_date_format(&self) -> &str {
    match &self.config.format {
      Some(f) => f,
      _ => DEFAULT_FORMAT,
    }
  }

    pub fn from_config(config: &DatetimeTimelineConfig) -> Self {
        DatetimeTimeline::new(config.clone())
    }
}

impl Timeline for DatetimeTimeline {
    fn get_bits(&self, timeline_value: &ComplexTimelineValue) -> Vec<bool> {
        if let ComplexTimelineValue::Datetime(date_str) = timeline_value {
          let format = self.get_date_format();

            let date_result = NaiveDateTime::parse_from_str(date_str, format);

            return match date_result {
              Ok(date) => {
                let mut res = vec![];

                for value in number_to_bits(date.year() as usize - 1900, 8, 2155) {
                  res.push(value);
                }

                for value in number_to_bits(date.month() as usize, 4, 12) {
                  res.push(value);
                }

                for value in number_to_bits(date.day() as usize, 5, 31) {
                  res.push(value);
                }

                for value in number_to_bits(date.hour() as usize, 5, 24) {
                  res.push(value);
                }

                for value in number_to_bits(date.minute() as usize / 4, 2, 16) {
                  res.push(value);
                }

                res
              },
              _ => vec![false; CAPACITY],
            };
        }

        panic!("Invalid value of datetime timeline");
    }

    fn reverse(&self, bits: &[bool]) -> ComplexTimelineValue {
        let format = self.get_date_format();

        let year = bits_to_number(&bits[0..8]);
        let month = std::cmp::min(bits_to_number(&bits[8..12]), 12);
        let day = std::cmp::min(bits_to_number(&bits[12..17]), 31);
        let hour = std::cmp::min(bits_to_number(&bits[17..22]), 23);
        let minute = std::cmp::min(bits_to_number(&bits[22..24]) * 4, 59);

        let date_opt = NaiveDate::from_ymd_opt(year as i32, month as u32, day as u32);

        if date_opt.is_none() {
          return  ComplexTimelineValue::Datetime(String::from(""));
        }

        let date_opt = date_opt.unwrap().and_hms_opt(hour as u32, minute as u32, 0);

        if date_opt.is_none() {
          return ComplexTimelineValue::Datetime(String::from(""));
        }

        let datetime = date_opt.unwrap();
        
        ComplexTimelineValue::Datetime(format!("{}", datetime.format(format)))
    }

    fn get_capacity(&self) -> &u8 {
        &(CAPACITY as u8)
    }
}
