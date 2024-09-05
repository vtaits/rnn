use serde_derive::Deserialize;

use crate::{bits_to_number, number_to_bits, ComplexTimelineValue, Timeline};

#[derive(Deserialize)]
pub struct IntegerTimelineConfig {
    pub min_value: i32,
    pub max_value: i32,
    pub capacity: u8,
}

pub struct IntegerTimelineParams {
    pub min_value: i32,
    pub max_value: i32,
    pub capacity: u8,
    pub get_multiplier: Option<Box<dyn Fn(f32) -> f32 + Send + Sync>>,
    pub get_reverse_multiplier: Option<Box<dyn Fn(f32) -> f32 + Send + Sync>>,
}

pub struct IntegerTimeline {
    range: i32,
    max_normalize_value: usize,
    params: IntegerTimelineParams,
}

impl IntegerTimeline {
    pub fn new(params: IntegerTimelineParams) -> Self {
        let max_normalize_value = 2usize.pow(params.capacity as u32) - 1;
        let range = params.max_value - params.min_value;

        IntegerTimeline {
            max_normalize_value,
            params,
            range,
        }
    }

    pub fn from_config(config: &IntegerTimelineConfig) -> Self {
        let IntegerTimelineConfig {
            min_value,
            max_value,
            capacity,
        } = config;

        let params = IntegerTimelineParams {
            min_value: *min_value,
            max_value: *max_value,
            capacity: *capacity,
            get_multiplier: None,
            get_reverse_multiplier: None,
        };

        IntegerTimeline::new(params)
    }

    fn get_multiplier(&self, default_multiplier: f32) -> f32 {
        if let Some(get_multiplier) = &self.params.get_multiplier {
            return (get_multiplier)(default_multiplier);
        }

        default_multiplier
    }

    fn normalize_value(&self, value: i32) -> usize {
        let multiplier =
            self.get_multiplier((value - self.params.min_value) as f32 / self.range as f32);

        (self.max_normalize_value as f32 * multiplier).round() as usize
    }

    fn get_reverse_multiplier(&self, multiplier: f32) -> f32 {
        if let Some(get_reverse_multiplier) = &self.params.get_reverse_multiplier {
            return (get_reverse_multiplier)(multiplier);
        }

        multiplier
    }
}

impl Timeline for IntegerTimeline {
    fn reverse(&self, bits: &[bool]) -> ComplexTimelineValue {
        let normalized_value = bits_to_number(bits);

        let multiplier = normalized_value as f32 / self.max_normalize_value as f32;

        let reverse_multiplier = self.get_reverse_multiplier(multiplier);

        let result =
            self.params.min_value + (self.range as f32 * reverse_multiplier).round() as i32;

        ComplexTimelineValue::Integer(result)
    }

    fn get_capacity(&self) -> &u8 {
        &self.params.capacity
    }

    fn get_bits(&self, timeline_value: &ComplexTimelineValue) -> Vec<bool> {
        if let ComplexTimelineValue::Integer(value) = timeline_value {
            if *value > self.params.max_value {
                return vec![true; self.params.capacity as usize];
            }

            if *value < self.params.min_value {
                return vec![false; self.params.capacity as usize];
            }

            let normalized_value = self.normalize_value(*value);

            return number_to_bits(
                normalized_value,
                self.params.capacity,
                self.max_normalize_value,
            );
        }

        panic!("Invalid value of integer timeline");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normaize_linear_value() {
        let timeline = IntegerTimeline::new(IntegerTimelineParams {
            capacity: 5,
            min_value: 10,
            max_value: 110,
            get_multiplier: None,
            get_reverse_multiplier: None,
        });

        assert_eq!(timeline.normalize_value(16), 2);
        assert_eq!(timeline.normalize_value(39), 9);
        assert_eq!(timeline.normalize_value(106), 30);
    }

    #[test]
    fn get_linear_value_bits() {
        let timelines = vec![
            IntegerTimeline::new(IntegerTimelineParams {
                capacity: 5,
                min_value: 10,
                max_value: 110,
                get_multiplier: None,
                get_reverse_multiplier: None,
            }),
            IntegerTimeline::from_config(&IntegerTimelineConfig {
                capacity: 5,
                min_value: 10,
                max_value: 110,
            }),
        ];

        for timeline in timelines {
            assert_eq!(
                timeline.get_bits(&ComplexTimelineValue::Integer(5)),
                vec![false, false, false, false, false],
                "too small value"
            );
            assert_eq!(
                timeline.get_bits(&ComplexTimelineValue::Integer(115)),
                vec![true, true, true, true, true],
                "too big value"
            );

            assert_eq!(
                timeline.get_bits(&ComplexTimelineValue::Integer(16)),
                vec![false, false, false, true, false],
                "2"
            );
            assert_eq!(
                timeline.get_bits(&ComplexTimelineValue::Integer(39)),
                vec![false, true, false, false, true],
                "9"
            );
            assert_eq!(
                timeline.get_bits(&ComplexTimelineValue::Integer(106)),
                vec![true, true, true, true, false],
                "30"
            );
        }
    }

    #[test]
    fn get_parabolic_value_bits() {
        let timeline = IntegerTimeline::new(IntegerTimelineParams {
            capacity: 5,
            min_value: 10,
            max_value: 110,
            get_multiplier: Some(Box::new(|value| value * value)),
            get_reverse_multiplier: None,
        });

        assert_eq!(
            timeline.get_bits(&ComplexTimelineValue::Integer(5)),
            vec![false, false, false, false, false],
            "too small value"
        );
        assert_eq!(
            timeline.get_bits(&ComplexTimelineValue::Integer(115)),
            vec![true, true, true, true, true],
            "too big value"
        );

        assert_eq!(
            timeline.get_bits(&ComplexTimelineValue::Integer(100)),
            vec![true, true, false, false, true],
            "25"
        );
        assert_eq!(
            timeline.get_bits(&ComplexTimelineValue::Integer(39)),
            vec![false, false, false, true, true],
            "9"
        );
        assert_eq!(
            timeline.get_bits(&ComplexTimelineValue::Integer(77)),
            vec![false, true, true, true, false],
            "14"
        );
    }

    #[test]
    fn reverse_linear() {
        let timeline = IntegerTimeline::new(IntegerTimelineParams {
            capacity: 5,
            min_value: 10,
            max_value: 110,
            get_multiplier: None,
            get_reverse_multiplier: None,
        });

        assert_eq!(
            timeline.reverse(&[false, false, false, true, false]),
            ComplexTimelineValue::Integer(16)
        );
        assert_eq!(
            timeline.reverse(&[false, true, false, false, true]),
            ComplexTimelineValue::Integer(39)
        );
        assert_eq!(
            timeline.reverse(&[true, true, true, true, false]),
            ComplexTimelineValue::Integer(107)
        );
    }

    #[test]
    fn reverse_parabolic() {
        let timeline = IntegerTimeline::new(IntegerTimelineParams {
            capacity: 5,
            min_value: 10,
            max_value: 110,
            get_multiplier: None,
            get_reverse_multiplier: Some(Box::new(|value| value.sqrt())),
        });

        assert_eq!(
            timeline.reverse(&[true, true, false, false, true]),
            ComplexTimelineValue::Integer(100)
        );
        assert_eq!(
            timeline.reverse(&[false, false, false, true, true]),
            ComplexTimelineValue::Integer(41)
        );
        assert_eq!(
            timeline.reverse(&[false, true, true, true, false]),
            ComplexTimelineValue::Integer(77)
        );
    }
}
