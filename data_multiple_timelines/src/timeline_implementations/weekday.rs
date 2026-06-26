use crate::{timeline_value::TimelineValue, timeline_value_retriever::TimelineValueRetriever};

pub struct WeekdayTimelineValue {
    value: u8,
    bit_index: usize,
}

impl WeekdayTimelineValue {
    pub fn new(value: u8, bit_index: usize) -> Self {
        Self { value, bit_index }
    }

    pub fn parse(value: u8) -> Self {
        Self {
            value,
            bit_index: if value < 7 { value as usize } else { 6 },
        }
    }
}

impl TimelineValue for WeekdayTimelineValue {
    fn get_primitive_value(&self) -> crate::TimelinePrimitiveValue {
        crate::TimelinePrimitiveValue::Weekday(self.value)
    }

    fn to_binary(&self) -> Vec<bool> {
        let mut res = vec![false; 7];

        res[self.bit_index] = true;

        res
    }
}

pub struct WeekdayRetriever();

impl WeekdayRetriever {
    pub fn new() -> Self {
        Self()
    }
}

impl TimelineValueRetriever for WeekdayRetriever {
    fn get_capacity(&self) -> usize {
        7
    }

    fn retrieve(&self, signal: &[bool]) -> Box<dyn TimelineValue> {
        let truthy_index = signal.iter().position(|&x| x).unwrap_or(0);

        Box::new(WeekdayTimelineValue::new(truthy_index as u8, truthy_index))
    }
}
