use crate::timeline_value::TimelineValue;

pub trait TimelineValueRetriever {
    fn get_capacity(&self) -> usize;

    fn retrieve(&self, signal: &[bool]) -> Box<dyn TimelineValue>;
}
