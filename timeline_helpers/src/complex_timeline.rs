use crate::{ComplexTimelineValue, Timeline};

pub struct ComplexTimeline {
    items: Vec<Box<dyn Timeline>>,
}

impl ComplexTimeline {
    pub fn new(items: Vec<Box<dyn Timeline>>) -> Self {
        ComplexTimeline { items }
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
            })),
            Box::new(IntegerTimeline::new(IntegerTimelineParams {
                capacity: 5,
                min_value: 10,
                max_value: 110,
                get_multiplier: None,
                get_reverse_multiplier: None,
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
            })),
            Box::new(IntegerTimeline::new(IntegerTimelineParams {
                capacity: 5,
                min_value: 10,
                max_value: 110,
                get_multiplier: None,
                get_reverse_multiplier: None,
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
