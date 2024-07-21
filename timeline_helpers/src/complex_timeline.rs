use crate::{EnumTimeline, FloatTimeline, IntegerTimeline};

pub enum ComplexTimelineItem {
    Float(FloatTimeline),
    Integer(IntegerTimeline),
    Enum(EnumTimeline<String>),
}

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

    fn reverse(&self, bits: &[bool]) -> Vec<ComplexTimelineValue> {
        let mut res = vec![];
        let mut offset = 0;

        for timeline_item in &self.items {
            let capacity = match timeline_item {
                ComplexTimelineItem::Float(float_timeline) => float_timeline.get_capacity(),
                ComplexTimelineItem::Integer(integer_timeline) => integer_timeline.get_capacity(),
                ComplexTimelineItem::Enum(enum_timeline) => enum_timeline.get_capacity(),
            };

            let next_offset = offset + capacity.clone() as usize;

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
