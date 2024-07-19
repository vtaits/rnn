#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn convert() {
        assert_eq!(bits_to_number(&[false, false, false, false, false]), 0);
        assert_eq!(bits_to_number(&[false, false, false, true, false]), 2);
        assert_eq!(bits_to_number(&[false, true, false, false, true]), 9);
        assert_eq!(bits_to_number(&[true, true, true, true, false]), 30);
        assert_eq!(bits_to_number(&[true, true, true, true, true]), 31);
    }
}

pub fn bits_to_number(bits: &[bool]) -> usize {
    let mut result: usize = 0;

    for bit in bits {
        result *= 2;

        if *bit {
            result += 1;
        }
    }

    result
}
