pub fn wape(actual: &[f32], received: &[f32]) -> f32 {
    let abs_error_sum: f32 = received
        .iter()
        .enumerate()
        .map(|(index, received_value)| (actual[index] - received_value).abs())
        .sum();
    let actual_sum: f32 = actual.iter().map(|actual_value| actual_value.abs()).sum();

    if actual_sum == 0.0 {
        panic!("There are no results");
    }

    abs_error_sum / actual_sum * 100.0
}
