use crate::{timeline_value::TimelineValue, timeline_value_retriever::TimelineValueRetriever};

pub struct FloatTimelineValue {
    value: f32,
    capacity: usize,
    bit_index: usize,
}

impl FloatTimelineValue {
    pub fn new(value: f32, capacity: usize, bit_index: usize) -> Self {
        Self {
            value,
            capacity,
            bit_index,
        }
    }

    pub fn parse(value: f32, capacity: usize, min_value: f32, max_value: f32) -> Self {
        let step = (max_value - min_value) / ((capacity - 1) as f32);

        Self {
            value,
            capacity,
            bit_index: if value < min_value {
                0
            } else {
                let res = ((value - min_value) / step).round() as usize;

                if res >= capacity {
                    capacity - 1
                } else {
                    res
                }
            },
        }
    }
}

impl TimelineValue for FloatTimelineValue {
    fn get_primitive_value(&self) -> crate::TimelinePrimitiveValue {
        crate::TimelinePrimitiveValue::Float(self.value)
    }

    fn to_binary(&self) -> Vec<bool> {
        let mut res = vec![false; self.capacity];

        res[self.bit_index] = true;

        res
    }
}

pub struct FloatRetriever {
    capacity: usize,
    min_value: f32,
    step: f32,
}

impl FloatRetriever {
    pub fn new(capacity: usize, min_value: f32, max_value: f32) -> Self {
        Self {
            capacity,
            min_value,
            step: (max_value - min_value) / ((capacity - 1) as f32),
        }
    }
}

impl TimelineValueRetriever for FloatRetriever {
    fn get_capacity(&self) -> usize {
        self.capacity
    }

    fn retrieve(&self, signal: &[bool]) -> Box<dyn TimelineValue> {
        let max_truthy_index = signal.iter().rposition(|&x| x).unwrap_or(0);

        Box::new(FloatTimelineValue::new(
            self.min_value + self.step * max_truthy_index as f32,
            self.capacity,
            max_truthy_index,
        ))
    }
}
