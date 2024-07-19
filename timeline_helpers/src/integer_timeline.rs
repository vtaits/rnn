use crate::number_to_bits;

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
        });

        assert_eq!(timeline.normalize_value(16), 2);
        assert_eq!(timeline.normalize_value(39), 9);
        assert_eq!(timeline.normalize_value(106), 30);
    }

    #[test]
    fn get_linear_value_bits() {
        let timeline = IntegerTimeline::new(IntegerTimelineParams {
            capacity: 5,
            min_value: 10,
            max_value: 110,
            get_multiplier: None,
        });

        assert_eq!(
            timeline.get_bits(5),
            vec![false, false, false, false, false],
            "too small value"
        );
        assert_eq!(
            timeline.get_bits(115),
            vec![true, true, true, true, true],
            "too big value"
        );

        assert_eq!(
            timeline.get_bits(16),
            vec![false, false, false, true, false],
            "2"
        );
        assert_eq!(
            timeline.get_bits(39),
            vec![false, true, false, false, true],
            "9"
        );
        assert_eq!(
            timeline.get_bits(106),
            vec![true, true, true, true, false],
            "30"
        );
    }

    #[test]
    fn get_parabolic_value_bits() {
      let timeline = IntegerTimeline::new(IntegerTimelineParams {
          capacity: 5,
          min_value: 10,
          max_value: 110,
          get_multiplier: Some(Box::new(|value| {
            let normalized = (value as f32 - 10.0) / 100.0;

            normalized * normalized
          })),
      });

      assert_eq!(
          timeline.get_bits(5),
          vec![false, false, false, false, false],
          "too small value"
      );
      assert_eq!(
          timeline.get_bits(115),
          vec![true, true, true, true, true],
          "too big value"
      );

      assert_eq!(
          timeline.get_bits(100),
          vec![true, true, false, false, true],
          "25"
      );
      assert_eq!(
          timeline.get_bits(39),
          vec![false, false, false, true, true],
          "9"
      );
      assert_eq!(
          timeline.get_bits(77),
          vec![false, true, true, true, false],
          "14"
      );
  }
}

pub struct IntegerTimelineParams {
    pub min_value: i32,
    pub max_value: i32,
    pub capacity: u8,
    pub get_multiplier: Option<Box<dyn Fn(i32) -> f32>>,
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

        return IntegerTimeline {
            max_normalize_value,
            params,
            range,
        };
    }

    fn get_multiplier(&self, value: i32) -> f32 {
        if let Some(get_multiplier) = &self.params.get_multiplier {
            return (get_multiplier)(value);
        }

        (value - self.params.min_value) as f32 / self.range as f32
    }

    fn normalize_value(&self, value: i32) -> usize {
      let multiplier = self.get_multiplier(value);

      (self.max_normalize_value as f32 * multiplier).round() as usize
  }

    pub fn get_bits(&self, value: i32) -> Vec<bool> {
        if value > self.params.max_value {
            return vec![true; self.params.capacity as usize];
        }

        if value < self.params.min_value {
            return vec![false; self.params.capacity as usize];
        }

        let normalized_value = self.normalize_value(value);

        number_to_bits(
            normalized_value,
            self.params.capacity,
            self.max_normalize_value,
        )
    }
}
