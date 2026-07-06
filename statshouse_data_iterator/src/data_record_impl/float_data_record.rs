use chrono::NaiveDateTime;
use data_multiple_timelines::{FloatTimelineValue, TimelineValue};

use crate::data_record::DataRecord;

pub struct FloatDataRecord {
    time: NaiveDateTime,
    value: f32,
    capacity: usize,
    min_value: f32,
    max_value: f32,
}

impl FloatDataRecord {
    pub fn new(
        time: NaiveDateTime,
        value: f32,
        capacity: usize,
        min_value: f32,
        max_value: f32,
    ) -> Self {
        Self {
            time,
            value,
            capacity,
            min_value,
            max_value,
        }
    }
}

impl DataRecord for FloatDataRecord {
    fn get_time(&self) -> &NaiveDateTime {
        &self.time
    }

    fn get_value(&self) -> Box<dyn TimelineValue> {
        Box::new(FloatTimelineValue::parse(
            self.value,
            self.capacity,
            self.min_value,
            self.max_value,
        ))
    }
}
