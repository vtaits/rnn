use rnn_core::{Partition, RegressResult};

use crate::{ComplexTimelineValue, Timeline};

pub struct ComplexTimeline {
    items: Vec<Box<dyn Timeline>>,
}

impl ComplexTimeline {
    pub fn new(items: Vec<Box<dyn Timeline>>) -> Self {
        ComplexTimeline { items }
    }

    /**
     * Returns Vec<bool> where truthy elements are targets and falsy elements sorce data to calculate targets
     */
    pub fn get_target_mask(&self) -> Vec<bool> {
        let mut result = vec![];

        for timeline_item in self.items.iter() {
            let bits = vec![timeline_item.is_target(); *timeline_item.get_capacity() as usize];

            for bit in bits {
                result.push(bit);
            }
        }

        result
    }

    pub fn get_bits(&self, value: &[ComplexTimelineValue]) -> Result<Vec<bool>, ()> {
        let mut result = vec![];

        for (index, item) in value.iter().enumerate() {
            let timeline_item = &self.items[index];

            let bits = timeline_item.get_bits(item);

            for bit in bits {
                result.push(bit);
            }
        }

        Ok(result)
    }

    pub fn reverse(&self, bits: &[bool]) -> Vec<ComplexTimelineValue> {
        let mut res = vec![];
        let mut offset = 0;

        for timeline_item in &self.items {
            let capacity = timeline_item.get_capacity();

            let next_offset = offset + *capacity as usize;

            let timeline_bits = &bits[offset..next_offset];

            let res_item = timeline_item.reverse(timeline_bits);

            res.push(res_item);

            offset = next_offset;
        }

        res
    }

    pub fn normalize_prediction(&self, bits: &[bool]) -> Vec<bool> {
        let mut result = vec![];
        let mut last_index = 0usize;

        for timeline_item in self.items.iter() {
            let capacity = *timeline_item.get_capacity() as usize;

            let timeline_bits = &bits[last_index..last_index + capacity];

            let normalized_bits = timeline_item.normalize_prediction(timeline_bits);

            for bit in normalized_bits {
                result.push(bit);
            }

            last_index += capacity;
        }

        result
    }

    pub fn get_partitions(&self) -> Vec<Partition> {
        self.items
            .iter()
            .map(|timeline_item| timeline_item.get_partition())
            .collect()
    }

    pub fn regress(&self, bits: &[bool]) -> f32 {
        let mut result = None;

        let mut last_index = 0usize;

        for timeline_item in self.items.iter() {
            let capacity = *timeline_item.get_capacity() as usize;

            if timeline_item.is_target() {
                let timeline_bits = &bits[last_index..last_index + capacity];

                let regressed_value = timeline_item.regress(timeline_bits);

                result = Some(regressed_value);
            } else if result.is_some() {
                panic!("There are many regression timelines");
            }

            last_index += capacity;
        }

        result.expect("There should be one target timeline")
    }

    pub fn get_regress_difference(
        &self,
        original: &[ComplexTimelineValue],
        computed: &[ComplexTimelineValue],
    ) -> RegressResult {
        for (index, timeline_item) in self.items.iter().enumerate() {
            if timeline_item.is_target() {
                return timeline_item.get_regress_difference(&original[index], &computed[index]);
            }
        }

        panic!("Target timeline is not defined")
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        EnumTimeline, EnumTimelineParams, FloatTimeline, FloatTimelineParams, IntegerTimeline,
        IntegerTimelineParams,
    };

    use super::*;

    #[test]
    fn get_value_bits() {
        let timeline = ComplexTimeline::new(vec![
            Box::new(FloatTimeline::new(FloatTimelineParams {
                capacity: 5,
                min_value: 10.0,
                max_value: 110.0,
                get_multiplier: None,
                get_reverse_multiplier: None,
                is_target: false,
                is_single_bit: false,
            })),
            Box::new(IntegerTimeline::new(IntegerTimelineParams {
                capacity: 5,
                min_value: 10,
                max_value: 110,
                get_multiplier: None,
                get_reverse_multiplier: None,
                is_target: None,
            })),
            Box::new(EnumTimeline::<String>::new(EnumTimelineParams {
                capacity: 3,
                to_number: Box::new(|value| match &value[..] {
                    "one" => 1,
                    "two" => 2,
                    "three" => 3,
                    "four" => 4,
                    "five" => 5,
                    _ => 0,
                }),
                to_option: Box::new(|value| {
                    String::from(match value {
                        1 => "one",
                        2 => "two",
                        3 => "three",
                        4 => "four",
                        5 => "five",
                        _ => "zero",
                    })
                }),
                is_target: None,
            })),
        ]);

        assert_eq!(
            timeline
                .get_bits(&[
                    ComplexTimelineValue::Float(39.0),
                    ComplexTimelineValue::Integer(106),
                    ComplexTimelineValue::Enum(String::from("three")),
                ])
                .unwrap(),
            vec![false, true, false, false, true, true, true, true, true, false, false, true, true],
        );
    }

    #[test]
    fn reverse() {
        let timeline = ComplexTimeline::new(vec![
            Box::new(FloatTimeline::new(FloatTimelineParams {
                capacity: 5,
                min_value: 10.0,
                max_value: 110.0,
                get_multiplier: None,
                get_reverse_multiplier: None,
                is_target: false,
                is_single_bit: false,
            })),
            Box::new(IntegerTimeline::new(IntegerTimelineParams {
                capacity: 5,
                min_value: 10,
                max_value: 110,
                get_multiplier: None,
                get_reverse_multiplier: None,
                is_target: None,
            })),
            Box::new(EnumTimeline::<String>::new(EnumTimelineParams {
                capacity: 3,
                to_number: Box::new(|value| match &value[..] {
                    "one" => 1,
                    "two" => 2,
                    "three" => 3,
                    "four" => 4,
                    "five" => 5,
                    _ => 0,
                }),
                to_option: Box::new(|value| {
                    String::from(match value {
                        1 => "one",
                        2 => "two",
                        3 => "three",
                        4 => "four",
                        5 => "five",
                        _ => "zero",
                    })
                }),
                is_target: None,
            })),
        ]);

        let result = timeline.reverse(&[
            false, true, false, false, true, true, true, true, true, false, false, true, true,
        ]);

        assert_eq!(result[1], ComplexTimelineValue::Integer(107));
        assert_eq!(result[2], ComplexTimelineValue::Enum(String::from("three")));

        if let ComplexTimelineValue::Float(float_value) = result[0] {
            assert!((float_value - 39.0).abs() < 0.1)
        } else {
            panic!("Wrong item type");
        }
    }
}
