use chrono::NaiveDateTime;
use serde_derive::Deserialize;
use timeline_helpers::ComplexTimelineValue;

use super::{CsvStreamConfig, CsvDateTimeStreamConfig};

#[derive(Deserialize)]
#[serde(tag = "type")]

pub enum TrainingStreamConfig {
    CsvDateTime(CsvDateTimeStreamConfig),
    Csv(CsvStreamConfig),
}

pub trait TrainingStream {
    fn get_value(&self) -> ComplexTimelineValue;

    fn get_date(&self) -> &Option<NaiveDateTime>;

    fn get_next_date(&self) -> &Option<NaiveDateTime>;

    fn set_date(&mut self, date: NaiveDateTime);

    fn is_finish(&self) -> bool;
}
