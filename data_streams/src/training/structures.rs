use chrono::NaiveDateTime;
use timeline_helpers::ComplexTimelineValue;

pub trait TrainingStream {
    fn get_value(&self) -> ComplexTimelineValue;

    fn get_date(&self) -> &Option<NaiveDateTime>;

    fn get_next_date(&self) -> &Option<NaiveDateTime>;

    fn set_date(&mut self, date: NaiveDateTime);

    fn is_finish(&self) -> bool;
}
