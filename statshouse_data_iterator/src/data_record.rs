use chrono::NaiveDateTime;
use data_multiple_timelines::TimelineValue;

pub trait DataRecord {
    fn get_time(&self) -> &NaiveDateTime;

    fn get_value(&self) -> Box<dyn TimelineValue>;
}
