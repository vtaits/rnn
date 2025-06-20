#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn convert() {
        assert_eq!(
            single_bit_to_number(&[false, false, false, false, false]),
            0
        );
        assert_eq!(single_bit_to_number(&[true, false, false, false, false]), 1);
        assert_eq!(single_bit_to_number(&[false, true, false, false, false]), 2);
        assert_eq!(single_bit_to_number(&[false, false, true, false, false]), 3);
        assert_eq!(single_bit_to_number(&[false, false, false, true, false]), 4);
        assert_eq!(single_bit_to_number(&[false, false, false, false, true]), 5);
    }
}

pub fn single_bit_to_number(bits: &[bool]) -> usize {
    for (index, bit) in bits.iter().enumerate() {
        if *bit {
            return index + 1;
        }
    }

    0
}
