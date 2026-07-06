use chrono::{Datelike, NaiveDateTime};
use data_multiple_timelines::{TimelineValue, WeekdayTimelineValue};

use crate::data_record::DataRecord;

pub struct WeekdayDataRecord {
    time: NaiveDateTime,
}

impl WeekdayDataRecord {
    pub fn new(time: NaiveDateTime) -> Self {
        Self { time }
    }
}

impl DataRecord for WeekdayDataRecord {
    fn get_time(&self) -> &NaiveDateTime {
        &self.time
    }

    fn get_value(&self) -> Box<dyn TimelineValue> {
        let value = match self.time.weekday() {
            chrono::Weekday::Mon => 0,
            chrono::Weekday::Tue => 1,
            chrono::Weekday::Wed => 2,
            chrono::Weekday::Thu => 3,
            chrono::Weekday::Fri => 4,
            chrono::Weekday::Sat => 5,
            chrono::Weekday::Sun => 6,
        };
        Box::new(WeekdayTimelineValue::parse(value))
    }
}
