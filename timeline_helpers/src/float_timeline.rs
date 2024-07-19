use crate::number_to_bits;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normaize_linear_value() {
        let timeline = FloatTimeline::new(FloatTimelineParams {
            capacity: 5,
            min_value: 10.0,
            max_value: 110.0,
            get_multiplier: None,
        });

        assert_eq!(timeline.normalize_value(16.4), 2);
        assert_eq!(timeline.normalize_value(39.0), 9);
        assert_eq!(timeline.normalize_value(106.7), 30);
    }

    #[test]
    fn get_linear_value_bits() {
        let timeline = FloatTimeline::new(FloatTimelineParams {
            capacity: 5,
            min_value: 10.0,
            max_value: 110.0,
            get_multiplier: None,
        });

        assert_eq!(
            timeline.get_bits(5.0),
            vec![false, false, false, false, false],
            "too small value"
        );
        assert_eq!(
            timeline.get_bits(115.0),
            vec![true, true, true, true, true],
            "too big value"
        );

        assert_eq!(
            timeline.get_bits(16.4),
            vec![false, false, false, true, false],
            "2"
        );
        assert_eq!(
            timeline.get_bits(39.0),
            vec![false, true, false, false, true],
            "9"
        );
        assert_eq!(
            timeline.get_bits(106.7),
            vec![true, true, true, true, false],
            "30"
        );
    }

    #[test]
    fn get_parabolic_value_bits() {
      let timeline = FloatTimeline::new(FloatTimelineParams {
          capacity: 5,
          min_value: 10.0,
          max_value: 110.0,
          get_multiplier: Some(Box::new(|value| {
            let normalized = (value - 10.0) / 100.0;

            normalized * normalized
          })),
      });

      assert_eq!(
          timeline.get_bits(5.0),
          vec![false, false, false, false, false],
          "too small value"
      );
      assert_eq!(
          timeline.get_bits(115.0),
          vec![true, true, true, true, true],
          "too big value"
      );

      assert_eq!(
          timeline.get_bits(100.3),
          vec![true, true, false, false, true],
          "25"
      );
      assert_eq!(
          timeline.get_bits(39.0),
          vec![false, false, false, true, true],
          "9"
      );
      assert_eq!(
          timeline.get_bits(77.0),
          vec![false, true, true, true, false],
          "14"
      );
  }
}

pub struct FloatTimelineParams {
    pub min_value: f32,
    pub max_value: f32,
    pub capacity: u8,
    pub get_multiplier: Option<Box<dyn Fn(f32) -> f32>>,
}


pub struct FloatTimeline {
    range: f32,
    max_normalize_value: usize,
    params: FloatTimelineParams,
}

impl FloatTimeline {
    pub fn new(params: FloatTimelineParams) -> Self {
        let max_normalize_value = 2usize.pow(params.capacity as u32) - 1;
        let range = params.max_value - params.min_value;

        return FloatTimeline {
            max_normalize_value,
            params,
            range,
        };
    }

    fn get_multiplier(&self, value: f32) -> f32 {
        if let Some(get_multiplier) = &self.params.get_multiplier {
            return (get_multiplier)(value);
        }

        (value - self.params.min_value) / self.range
    }

    fn normalize_value(&self, value: f32) -> usize {
      let multiplier = self.get_multiplier(value);

      (self.max_normalize_value as f32 * multiplier).round() as usize
  }

    pub fn get_bits(&self, value: f32) -> Vec<bool> {
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
