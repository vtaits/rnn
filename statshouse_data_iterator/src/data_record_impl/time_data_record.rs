use chrono::{NaiveDateTime, Timelike};
use data_multiple_timelines::{TimeTimelineValue, TimelineValue};

use crate::data_record::DataRecord;

pub struct TimeDataRecord {
    time: NaiveDateTime,
    capacity: usize,
}

impl TimeDataRecord {
    pub fn new(time: NaiveDateTime, capacity: usize) -> Self {
        Self { time, capacity }
    }
}

impl DataRecord for TimeDataRecord {
    fn get_time(&self) -> &NaiveDateTime {
        &self.time
    }

    fn get_value(&self) -> Box<dyn TimelineValue> {
        Box::new(TimeTimelineValue::parse(
            self.time.hour() as u8,
            self.time.minute() as u8,
            self.capacity,
        ))
    }
}
