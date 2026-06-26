use crate::{timeline_value::TimelineValue, timeline_value_retriever::TimelineValueRetriever};

pub struct TimeTimelineValue {
    hours: u8,
    minutes: u8,
    capacity: usize,
    bit_index: usize,
}

const MINUTES_IN_DAY: usize = 24 * 60;

impl TimeTimelineValue {
    pub fn new(hours: u8, minutes: u8, capacity: usize, bit_index: usize) -> Self {
        Self {
            hours,
            minutes,
            capacity,
            bit_index,
        }
    }

    pub fn parse(hours: u8, minutes: u8, capacity: usize) -> Self {
        let current_minute = (hours as usize) * 60 + minutes as usize;
        let step_minutes = MINUTES_IN_DAY / ((capacity - 1) as usize);

        Self {
            hours,
            minutes,
            capacity,
            bit_index: {
                let res = current_minute / step_minutes;

                if res >= capacity {
                    capacity - 1
                } else {
                    res
                }
            },
        }
    }
}

impl TimelineValue for TimeTimelineValue {
    fn get_primitive_value(&self) -> crate::TimelinePrimitiveValue {
        crate::TimelinePrimitiveValue::Time(self.hours, self.minutes)
    }

    fn to_binary(&self) -> Vec<bool> {
        let mut res = vec![false; self.capacity];

        res[self.bit_index] = true;

        res
    }
}

pub struct TimeRetriever {
    capacity: usize,
    step_minutes: usize,
}

impl TimeRetriever {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            step_minutes: MINUTES_IN_DAY / ((capacity - 1) as usize),
        }
    }
}

impl TimelineValueRetriever for TimeRetriever {
    fn get_capacity(&self) -> usize {
        self.capacity
    }

    fn retrieve(&self, signal: &[bool]) -> Box<dyn TimelineValue> {
        let truthy_index = signal.iter().position(|&x| x).unwrap_or(0);

        let minutes = self.step_minutes * truthy_index;

        Box::new(TimeTimelineValue::new(
            (minutes / 60) as u8,
            (minutes % 60) as u8,
            self.capacity,
            truthy_index,
        ))
    }
}
