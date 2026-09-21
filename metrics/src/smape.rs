pub fn smape(actual: &[f32], received: &[f32]) -> f32 {
    if received.is_empty() {
        return 0.0;
    }

    let mut sum_numerator = 0.0;
    let mut sum_denominator = 0.0;

    for (index, received_value) in received.iter().enumerate() {
        let actual_value = actual[index];
        let denom = (actual_value.abs() + received_value.abs() + 1e-8) / 2.0;
        sum_numerator += (actual_value - received_value).abs();
        sum_denominator += denom;
    }

    100.0 * (sum_numerator / sum_denominator)
}
