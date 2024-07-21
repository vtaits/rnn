use crate::{bits_to_number, number_to_bits};

pub struct EnumTimelineParams<T> {
    pub capacity: u8,
    pub to_number: Box<dyn Fn(T) -> usize>,
    pub to_option: Box<dyn Fn(usize) -> T>,
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

    pub fn get_bits(&self, value: T) -> Vec<bool> {
        let number = (self.params.to_number)(value);

        number_to_bits(number, self.params.capacity, self.max_normalize_value)
    }

    pub fn reverse(&self, bits: &[bool]) -> T {
        let number = bits_to_number(bits);

        let option = (self.params.to_option)(number);

        option
    }

    pub fn get_capacity(&self) -> &u8 {
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
            to_number: Box::new(|value: &str| match value {
                "one" => 1,
                "two" => 2,
                "three" => 3,
                "four" => 4,
                "five" => 5,
                _ => 0,
            }),
            to_option: Box::new(|value| match value {
                1 => "one",
                2 => "two",
                3 => "three",
                4 => "four",
                5 => "five",
                _ => "zero",
            }),
        });

        assert_eq!(timeline.get_bits("one"), vec![false, false, true],);

        assert_eq!(timeline.get_bits("two"), vec![false, true, false],);

        assert_eq!(timeline.get_bits("three"), vec![false, true, true],);

        assert_eq!(timeline.get_bits("four"), vec![true, false, false],);

        assert_eq!(timeline.get_bits("five"), vec![true, false, true],);

        assert_eq!(timeline.get_bits("zero"), vec![false, false, false],);
    }

    #[test]
    fn reverse() {
        let timeline = EnumTimeline::new(EnumTimelineParams {
            capacity: 3,
            to_number: Box::new(|value: &str| match value {
                "one" => 1,
                "two" => 2,
                "three" => 3,
                "four" => 4,
                "five" => 5,
                _ => 0,
            }),
            to_option: Box::new(|value| match value {
                1 => "one",
                2 => "two",
                3 => "three",
                4 => "four",
                5 => "five",
                _ => "zero",
            }),
        });

        assert_eq!(timeline.reverse(&[false, false, true]), "one",);

        assert_eq!(timeline.reverse(&[false, true, false]), "two",);

        assert_eq!(timeline.reverse(&[false, true, true]), "three",);

        assert_eq!(timeline.reverse(&[true, false, false]), "four",);

        assert_eq!(timeline.reverse(&[true, false, true]), "five",);

        assert_eq!(timeline.reverse(&[false, false, false]), "zero",);
    }
}
