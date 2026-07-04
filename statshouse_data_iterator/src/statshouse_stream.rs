use chrono::NaiveDateTime;
use data_multiple_timelines::TimelineValue;

pub trait StatsHouseStream {
    fn get_next_time(&self) -> &Option<NaiveDateTime>;
    fn shift(&mut self);
    fn retrieve(&self) -> Box<dyn TimelineValue>;
}
