use crate::{EnumTimeline, FloatTimeline, IntegerTimeline};

pub enum ComplexTimelineItem {
    Float(FloatTimeline),
    Integer(IntegerTimeline),
    Enum(EnumTimeline<String>),
}

#[derive(PartialEq, Debug)]
pub enum ComplexTimelineValue {
    Float(f32),
    Integer(i32),
    Enum(String),
}

pub struct ComplexTimeline {
    items: Vec<ComplexTimelineItem>,
}

impl ComplexTimeline {
    pub fn new(items: Vec<ComplexTimelineItem>) -> Self {
        ComplexTimeline { items }
    }

    pub fn get_bits(&self, value: &[ComplexTimelineValue]) -> Result<Vec<bool>, ()> {
        let mut result = vec![];

        for (index, item) in value.iter().enumerate() {
            let timeline_item = &self.items[index];

            let bits = match item {
                ComplexTimelineValue::Float(float_value) => match timeline_item {
                    ComplexTimelineItem::Float(float_timeline) => {
                        float_timeline.get_bits(*float_value)
                    }
                    _ => {
                        return Result::Err(());
                    }
                },
                ComplexTimelineValue::Integer(integer_value) => match timeline_item {
                    ComplexTimelineItem::Integer(integer_timeline) => {
                        integer_timeline.get_bits(*integer_value)
                    }
                    _ => {
                        return Result::Err(());
                    }
                },
                ComplexTimelineValue::Enum(enum_value) => match timeline_item {
                    ComplexTimelineItem::Enum(integer_timeline) => {
                        integer_timeline.get_bits(enum_value.clone())
                    }
                    _ => {
                        return Result::Err(());
                    }
                },
            };

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
            let capacity = match timeline_item {
                ComplexTimelineItem::Float(float_timeline) => float_timeline.get_capacity(),
                ComplexTimelineItem::Integer(integer_timeline) => integer_timeline.get_capacity(),
                ComplexTimelineItem::Enum(enum_timeline) => enum_timeline.get_capacity(),
            };

            let next_offset = offset + *capacity as usize;

            let timeline_bits = &bits[offset..next_offset];

            let res_item = match timeline_item {
                ComplexTimelineItem::Float(float_timeline) => {
                    ComplexTimelineValue::Float(float_timeline.reverse(timeline_bits))
                }
                ComplexTimelineItem::Integer(integer_timeline) => {
                    ComplexTimelineValue::Integer(integer_timeline.reverse(timeline_bits))
                }
                ComplexTimelineItem::Enum(enum_timeline) => {
                    ComplexTimelineValue::Enum(enum_timeline.reverse(timeline_bits))
                }
            };

            res.push(res_item);

            offset = next_offset;
        }

        res
    }
}

#[cfg(test)]
mod tests {
    use crate::{EnumTimelineParams, FloatTimelineParams, IntegerTimelineParams};

    use super::*;

    #[test]
    fn get_value_bits() {
        let timeline = ComplexTimeline::new(vec![
            ComplexTimelineItem::Float(FloatTimeline::new(FloatTimelineParams {
                capacity: 5,
                min_value: 10.0,
                max_value: 110.0,
                get_multiplier: None,
                get_reverse_multiplier: None,
            })),
            ComplexTimelineItem::Integer(IntegerTimeline::new(IntegerTimelineParams {
                capacity: 5,
                min_value: 10,
                max_value: 110,
                get_multiplier: None,
                get_reverse_multiplier: None,
            })),
            ComplexTimelineItem::Enum(EnumTimeline::new(EnumTimelineParams {
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
            ComplexTimelineItem::Float(FloatTimeline::new(FloatTimelineParams {
                capacity: 5,
                min_value: 10.0,
                max_value: 110.0,
                get_multiplier: None,
                get_reverse_multiplier: None,
            })),
            ComplexTimelineItem::Integer(IntegerTimeline::new(IntegerTimelineParams {
                capacity: 5,
                min_value: 10,
                max_value: 110,
                get_multiplier: None,
                get_reverse_multiplier: None,
            })),
            ComplexTimelineItem::Enum(EnumTimeline::new(EnumTimelineParams {
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
