use chrono::NaiveDateTime;
use data_multiple_timelines::TimelineValue;

use crate::statshouse_stream::StatsHouseStream;

pub struct CsvValueStream {
    next_time: Option<NaiveDateTime>,
}

impl StatsHouseStream for CsvValueStream {
    fn retrieve(&self) -> Box<dyn TimelineValue> {}

    fn get_next_time(&self) -> &Option<NaiveDateTime> {
        &self.next_time
    }

    fn shift(&mut self) {}
}
