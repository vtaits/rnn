use serde_derive::Deserialize;

use crate::{
    bits_to_number, number_to_bits, number_to_single_bit, single_bit_to_number,
    ComplexTimelineValue, Timeline,
};

#[derive(Deserialize)]
pub struct FloatTimelineConfig {
    pub min_value: f32,
    pub max_value: f32,
    pub capacity: u8,
    pub is_target: Option<bool>,
    pub is_single_bit: Option<bool>,
}

pub struct FloatTimelineParams {
    pub min_value: f32,
    pub max_value: f32,
    pub capacity: u8,
    pub get_multiplier: Option<Box<dyn Fn(f32) -> f32 + Send + Sync>>,
    pub get_reverse_multiplier: Option<Box<dyn Fn(f32) -> f32 + Send + Sync>>,
    pub is_target: bool,
    pub is_single_bit: bool,
}

pub struct FloatTimeline {
    range: f32,
    max_normalize_value: usize,
    params: FloatTimelineParams,
    is_target: bool,
    is_single_bit: bool,
}

impl FloatTimeline {
    pub fn new(params: FloatTimelineParams) -> Self {
        let max_normalize_value = if params.is_single_bit {
            params.capacity as usize
        } else {
            2usize.pow(params.capacity as u32) - 1
        };

        let range = params.max_value - params.min_value;
        let is_target = params.is_target;
        let is_single_bit = params.is_single_bit;

        FloatTimeline {
            max_normalize_value,
            params,
            range,
            is_target,
            is_single_bit,
        }
    }

    pub fn from_config(config: &FloatTimelineConfig) -> Self {
        let FloatTimelineConfig {
            min_value,
            max_value,
            capacity,
            is_target,
            is_single_bit,
        } = config;

        let params = FloatTimelineParams {
            min_value: *min_value,
            max_value: *max_value,
            capacity: *capacity,
            get_multiplier: None,
            get_reverse_multiplier: None,
            is_target: is_target.unwrap_or_default(),
            is_single_bit: is_single_bit.unwrap_or_default(),
        };

        FloatTimeline::new(params)
    }

    fn get_multiplier(&self, default_multiplier: f32) -> f32 {
        if let Some(get_multiplier) = &self.params.get_multiplier {
            return (get_multiplier)(default_multiplier);
        }

        default_multiplier
    }

    fn normalize_value(&self, value: f32) -> usize {
        let multiplier = self.get_multiplier((value - self.params.min_value) / self.range);

        let result = self.max_normalize_value as f32 * multiplier;

        if self.is_single_bit {
            return result.floor() as usize;
        }

        result.round() as usize
    }

    fn get_reverse_multiplier(&self, multiplier: f32) -> f32 {
        if let Some(get_reverse_multiplier) = &self.params.get_reverse_multiplier {
            return (get_reverse_multiplier)(multiplier);
        }

        multiplier
    }
}

impl Timeline for FloatTimeline {
    fn is_target(&self) -> bool {
        self.is_target
    }

    fn get_bits(&self, timeline_value: &ComplexTimelineValue) -> Vec<bool> {
        if let ComplexTimelineValue::Float(value) = timeline_value {
            if self.is_single_bit {
                let normalized_value = self.normalize_value(*value);

                return number_to_single_bit(
                    normalized_value,
                    self.params.capacity as usize,
                    self.max_normalize_value,
                    false,
                );
            }

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

        panic!("Invalid value of float timeline");
    }

    fn get_capacity(&self) -> &u8 {
        &self.params.capacity
    }

    fn reverse(&self, bits: &[bool]) -> ComplexTimelineValue {
        let normalized_value = if self.is_single_bit {
            single_bit_to_number(bits)
        } else {
            bits_to_number(bits)
        };

        let multiplier = normalized_value as f32 / self.max_normalize_value as f32;

        let reverse_multiplier = self.get_reverse_multiplier(multiplier);

        let result = self.params.min_value + self.range * reverse_multiplier;

        ComplexTimelineValue::Float(result)
    }

    fn normalize_prediction(&self, bits: &[bool]) -> Vec<bool> {
        if !self.is_single_bit {
            return bits.to_vec();
        }

        let mut result = vec![false; bits.len()];

        let last_positive_bit_index = bits
            .iter()
            .rev()
            .position(|x| *x)
            .map(|pos| bits.len() - 1 - pos);

        if let Some(last_positive_bit_index) = last_positive_bit_index {
            result[last_positive_bit_index] = true
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normaize_linear_value() {
        let timelines = vec![
            FloatTimeline::from_config(&FloatTimelineConfig {
                capacity: 5,
                min_value: 10.0,
                max_value: 110.0,
                is_target: None,
                is_single_bit: None,
            }),
            FloatTimeline::new(FloatTimelineParams {
                capacity: 5,
                min_value: 10.0,
                max_value: 110.0,
                is_target: false,
                is_single_bit: false,
                get_multiplier: None,
                get_reverse_multiplier: None,
            }),
            FloatTimeline::from_config(&FloatTimelineConfig {
                capacity: 32,
                min_value: 10.0,
                max_value: 110.0,
                is_target: None,
                is_single_bit: Some(true),
            }),
        ];

        for timeline in timelines {
            assert_eq!(timeline.normalize_value(16.4), 2);
            assert_eq!(timeline.normalize_value(39.0), 9);
            assert_eq!(timeline.normalize_value(106.7), 30);
        }
    }

    #[test]
    fn get_linear_value_bits() {
        let timelines = vec![
            FloatTimeline::from_config(&FloatTimelineConfig {
                capacity: 5,
                min_value: 10.0,
                max_value: 110.0,
                is_target: None,
                is_single_bit: None,
            }),
            FloatTimeline::new(FloatTimelineParams {
                capacity: 5,
                min_value: 10.0,
                max_value: 110.0,
                get_multiplier: None,
                get_reverse_multiplier: None,
                is_target: false,
                is_single_bit: false,
            }),
        ];

        for timeline in timelines {
            assert_eq!(
                timeline.get_bits(&ComplexTimelineValue::Float(5.0)),
                vec![false, false, false, false, false],
                "too small value"
            );
            assert_eq!(
                timeline.get_bits(&ComplexTimelineValue::Float(115.0)),
                vec![true, true, true, true, true],
                "too big value"
            );

            assert_eq!(
                timeline.get_bits(&ComplexTimelineValue::Float(16.4)),
                vec![false, false, false, true, false],
                "2"
            );
            assert_eq!(
                timeline.get_bits(&ComplexTimelineValue::Float(39.0)),
                vec![false, true, false, false, true],
                "9"
            );
            assert_eq!(
                timeline.get_bits(&ComplexTimelineValue::Float(106.7)),
                vec![true, true, true, true, false],
                "30"
            );
        }
    }

    #[test]
    fn get_linear_value_single_bits() {
        let timelines = vec![
            FloatTimeline::from_config(&FloatTimelineConfig {
                capacity: 5,
                min_value: 10.0,
                max_value: 110.0,
                is_target: None,
                is_single_bit: Some(true),
            }),
            FloatTimeline::new(FloatTimelineParams {
                capacity: 5,
                min_value: 10.0,
                max_value: 110.0,
                get_multiplier: None,
                get_reverse_multiplier: None,
                is_target: false,
                is_single_bit: true,
            }),
        ];

        for timeline in timelines {
            assert_eq!(
                timeline.get_bits(&ComplexTimelineValue::Float(5.0)),
                vec![false, false, false, false, false],
                "too small value"
            );
            assert_eq!(
                timeline.get_bits(&ComplexTimelineValue::Float(115.0)),
                vec![false, false, false, false, true],
                "too big value"
            );

            assert_eq!(
                timeline.get_bits(&ComplexTimelineValue::Float(16.4)),
                vec![false, false, false, false, false]
            );
            assert_eq!(
                timeline.get_bits(&ComplexTimelineValue::Float(39.0)),
                vec![true, false, false, false, false]
            );
            assert_eq!(
                timeline.get_bits(&ComplexTimelineValue::Float(106.7)),
                vec![false, false, false, true, false]
            );
        }
    }

    #[test]
    fn get_parabolic_value_bits() {
        let timeline = FloatTimeline::new(FloatTimelineParams {
            capacity: 5,
            min_value: 10.0,
            max_value: 110.0,
            get_multiplier: Some(Box::new(|value| value * value)),
            get_reverse_multiplier: None,
            is_target: false,
            is_single_bit: false,
        });

        assert_eq!(
            timeline.get_bits(&ComplexTimelineValue::Float(5.0)),
            vec![false, false, false, false, false],
            "too small value"
        );
        assert_eq!(
            timeline.get_bits(&ComplexTimelineValue::Float(115.0)),
            vec![true, true, true, true, true],
            "too big value"
        );

        assert_eq!(
            timeline.get_bits(&ComplexTimelineValue::Float(100.3)),
            vec![true, true, false, false, true],
            "25"
        );
        assert_eq!(
            timeline.get_bits(&ComplexTimelineValue::Float(39.0)),
            vec![false, false, false, true, true],
            "9"
        );
        assert_eq!(
            timeline.get_bits(&ComplexTimelineValue::Float(77.0)),
            vec![false, true, true, true, false],
            "14"
        );
    }

    #[test]
    fn reverse_linear() {
        let timeline = FloatTimeline::new(FloatTimelineParams {
            capacity: 5,
            min_value: 10.0,
            max_value: 110.0,
            get_multiplier: None,
            get_reverse_multiplier: None,
            is_target: false,
            is_single_bit: false,
        });

        if let ComplexTimelineValue::Float(result) =
            timeline.reverse(&[false, false, false, true, false])
        {
            assert!((result - 16.4).abs() < 0.1);
        } else {
            panic!("Wrong result type");
        }

        if let ComplexTimelineValue::Float(result) =
            timeline.reverse(&[false, true, false, false, true])
        {
            assert!((result - 39.0).abs() < 0.1);
        } else {
            panic!("Wrong result type");
        }

        if let ComplexTimelineValue::Float(result) =
            timeline.reverse(&[true, true, true, true, false])
        {
            assert!((result - 106.7).abs() < 0.1);
        } else {
            panic!("Wrong result type");
        }
    }

    #[test]
    fn reverse_parabolic() {
        let timeline = FloatTimeline::new(FloatTimelineParams {
            capacity: 5,
            min_value: 10.0,
            max_value: 110.0,
            get_multiplier: None,
            get_reverse_multiplier: Some(Box::new(|value| value.sqrt())),
            is_target: false,
            is_single_bit: false,
        });

        if let ComplexTimelineValue::Float(result) =
            timeline.reverse(&[true, true, false, false, true])
        {
            assert!((result - 99.8).abs() < 0.1);
        } else {
            panic!("Wrong result type");
        }

        if let ComplexTimelineValue::Float(result) =
            timeline.reverse(&[false, false, false, true, true])
        {
            assert!((result - 41.1).abs() < 0.1);
        } else {
            panic!("Wrong result type");
        }

        if let ComplexTimelineValue::Float(result) =
            timeline.reverse(&[false, true, true, true, false])
        {
            assert!((result - 77.2).abs() < 0.1);
        } else {
            panic!("Wrong result type");
        }
    }
}
