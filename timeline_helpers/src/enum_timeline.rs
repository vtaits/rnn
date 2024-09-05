use std::collections::HashMap;

use serde_derive::Deserialize;

use crate::{bits_to_number, number_to_bits, ComplexTimelineValue, Timeline};

#[derive(Deserialize)]
pub struct EnumTimelineConfig {
    pub capacity: u8,
    pub options: Vec<String>,
}

pub struct EnumTimelineParams<T> {
    pub capacity: u8,
    pub to_number: Box<dyn Fn(T) -> usize + Send + Sync>,
    pub to_option: Box<dyn Fn(usize) -> T + Send + Sync>,
}

pub struct EnumTimeline<T> {
    max_normalize_value: usize,
    params: EnumTimelineParams<T>,
}

impl<T> EnumTimeline<T> {
    pub fn new(params: EnumTimelineParams<T>) -> Self {
        let max_normalize_value = 2usize.pow(params.capacity as u32) - 1;

        EnumTimeline {
            max_normalize_value,
            params,
        }
    }
}

impl EnumTimeline<String> {
    pub fn from_config(config: &EnumTimelineConfig) -> Self {
        let EnumTimelineConfig { capacity, options } = config;
        let mut option_to_index: HashMap<String, usize> = HashMap::new();

        for (index, option) in options.iter().enumerate() {
            option_to_index.insert(String::from(option), index);
        }

        EnumTimeline::new(EnumTimelineParams {
            capacity: *capacity,
            to_number: Box::new(move |option| {
                let index_option = option_to_index.get(&option);

                match index_option {
                    Some(index) => *index,
                    _ => 0,
                }
            }),
            to_option: {
                let options = options.clone();

                Box::new(move |index| {
                    if options.len() <= index {
                        return options[index].clone();
                    }

                    options[0].clone()
                })
            },
        })
    }
}

impl Timeline for EnumTimeline<String> {
    fn get_bits(&self, timeline_value: &ComplexTimelineValue) -> Vec<bool> {
        if let ComplexTimelineValue::Enum(value) = timeline_value {
            let number = (self.params.to_number)(value.clone());

            return number_to_bits(number, self.params.capacity, self.max_normalize_value);
        }

        panic!("Invalid value of float timeline");
    }

    fn reverse(&self, bits: &[bool]) -> ComplexTimelineValue {
        let number = bits_to_number(bits);

        ComplexTimelineValue::Enum((self.params.to_option)(number))
    }

    fn get_capacity(&self) -> &u8 {
        &self.params.capacity
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_value_bits() {
        let timeline = EnumTimeline::new(EnumTimelineParams {
            capacity: 3,
            to_number: Box::new(|value: String| match &value[..] {
                "one" => 1,
                "two" => 2,
                "three" => 3,
                "four" => 4,
                "five" => 5,
                _ => 0,
            }),
            to_option: Box::new(|value| match value {
                1 => String::from("one"),
                2 => String::from("two"),
                3 => String::from("three"),
                4 => String::from("four"),
                5 => String::from("five"),
                _ => String::from("zero"),
            }),
        });

        assert_eq!(
            timeline.get_bits(&ComplexTimelineValue::Enum(String::from("one"))),
            vec![false, false, true],
        );

        assert_eq!(
            timeline.get_bits(&ComplexTimelineValue::Enum(String::from("two"))),
            vec![false, true, false],
        );

        assert_eq!(
            timeline.get_bits(&ComplexTimelineValue::Enum(String::from("three"))),
            vec![false, true, true],
        );

        assert_eq!(
            timeline.get_bits(&ComplexTimelineValue::Enum(String::from("four"))),
            vec![true, false, false],
        );

        assert_eq!(
            timeline.get_bits(&ComplexTimelineValue::Enum(String::from("five"))),
            vec![true, false, true],
        );

        assert_eq!(
            timeline.get_bits(&ComplexTimelineValue::Enum(String::from("zero"))),
            vec![false, false, false],
        );
    }

    #[test]
    fn reverse() {
        let timeline = EnumTimeline::new(EnumTimelineParams {
            capacity: 3,
            to_number: Box::new(|value: String| match &value[..] {
                "one" => 1,
                "two" => 2,
                "three" => 3,
                "four" => 4,
                "five" => 5,
                _ => 0,
            }),
            to_option: Box::new(|value| match value {
                1 => String::from("one"),
                2 => String::from("two"),
                3 => String::from("three"),
                4 => String::from("four"),
                5 => String::from("five"),
                _ => String::from("zero"),
            }),
        });

        if let ComplexTimelineValue::Enum(result) = timeline.reverse(&[false, false, true]) {
            assert_eq!(result, "one",);
        } else {
            panic!("Wrong result type");
        }

        if let ComplexTimelineValue::Enum(result) = timeline.reverse(&[false, true, false]) {
            assert_eq!(result, "two",);
        } else {
            panic!("Wrong result type");
        }

        if let ComplexTimelineValue::Enum(result) = timeline.reverse(&[false, true, true]) {
            assert_eq!(result, "three",);
        } else {
            panic!("Wrong result type");
        }

        if let ComplexTimelineValue::Enum(result) = timeline.reverse(&[true, false, false]) {
            assert_eq!(result, "four",);
        } else {
            panic!("Wrong result type");
        }

        if let ComplexTimelineValue::Enum(result) = timeline.reverse(&[true, false, true]) {
            assert_eq!(result, "five",);
        } else {
            panic!("Wrong result type");
        }

        if let ComplexTimelineValue::Enum(result) = timeline.reverse(&[false, false, false]) {
            assert_eq!(result, "zero",);
        } else {
            panic!("Wrong result type");
        }
    }
}
