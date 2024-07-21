pub fn number_to_bits(number: usize, capacity: u8, max: usize) -> Vec<bool> {
    let mut result = Vec::new();

    if number > max {
        return vec![true; capacity as usize];
    }

    let mut temp_number = number;

    for _ in 0..capacity {
        result.push(temp_number % 2 == 1);

        temp_number /= 2;
    }

    result.reverse();

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fits() {
        assert_eq!(
            number_to_bits(0, 5, 31),
            vec![false, false, false, false, false]
        );
        assert_eq!(
            number_to_bits(2, 5, 31),
            vec![false, false, false, true, false]
        );
        assert_eq!(
            number_to_bits(9, 5, 31),
            vec![false, true, false, false, true]
        );
        assert_eq!(
            number_to_bits(30, 5, 31),
            vec![true, true, true, true, false]
        );
        assert_eq!(
            number_to_bits(31, 5, 31),
            vec![true, true, true, true, true]
        );
    }

    #[test]
    fn overflows() {
        assert_eq!(
            number_to_bits(32, 5, 31),
            vec![true, true, true, true, true]
        );
        assert_eq!(
            number_to_bits(100, 5, 31),
            vec![true, true, true, true, true]
        );
    }
}
