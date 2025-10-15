pub fn number_to_single_bit(
    number: usize,
    capacity: usize,
    max: usize,
    always_filled: bool,
) -> Vec<bool> {
    let mut result = vec![false; capacity as usize];

    let step = max / capacity;

    let interval_index = number / step;

    if interval_index == 0 && !always_filled {
        return result;
    }

    let bit_index = if interval_index >= capacity {
        capacity - 1
    } else {
        if always_filled {
            interval_index
        } else {
            interval_index - 1
        }
    };

    result[bit_index] = true;

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case(0, 5, 50, vec![false, false, false, false, false])]
    #[case(5, 5, 50, vec![false, false, false, false, false])]
    #[case(10, 5, 50, vec![true, false, false, false, false])]
    #[case(15, 5, 50, vec![true, false, false, false, false])]
    #[case(20, 5, 50, vec![false, true, false, false, false])]
    #[case(25, 5, 50, vec![false, true, false, false, false])]
    #[case(30, 5, 50, vec![false, false, true, false, false])]
    #[case(35, 5, 50, vec![false, false, true, false, false])]
    #[case(40, 5, 50, vec![false, false, false, true, false])]
    #[case(45, 5, 50, vec![false, false, false, true, false])]
    #[case(50, 5, 50, vec![false, false, false, false, true])]
    #[case(55, 5, 50, vec![false, false, false, false, true])]
    fn convert_correctly(
        #[case] number: usize,
        #[case] capacity: usize,
        #[case] max: usize,
        #[case] result: Vec<bool>,
    ) {
        assert_eq!(number_to_single_bit(number, capacity, max, false), result);
    }

    #[rstest]
    #[case(0, 5, 50, vec![true, false, false, false, false])]
    #[case(5, 5, 50, vec![true, false, false, false, false])]
    #[case(10, 5, 50, vec![false, true, false, false, false])]
    #[case(15, 5, 50, vec![false, true, false, false, false])]
    #[case(20, 5, 50, vec![false, false, true, false, false])]
    #[case(25, 5, 50, vec![false, false, true, false, false])]
    #[case(30, 5, 50, vec![false, false, false, true, false])]
    #[case(35, 5, 50, vec![false, false, false, true, false])]
    #[case(40, 5, 50, vec![false, false, false, false, true])]
    #[case(45, 5, 50, vec![false, false, false, false, true])]
    #[case(50, 5, 50, vec![false, false, false, false, true])]
    #[case(55, 5, 50, vec![false, false, false, false, true])]
    fn convert_correctly_always_filled(
        #[case] number: usize,
        #[case] capacity: usize,
        #[case] max: usize,
        #[case] result: Vec<bool>,
    ) {
        assert_eq!(number_to_single_bit(number, capacity, max, true), result);
    }
}
