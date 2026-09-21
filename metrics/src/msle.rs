pub fn msle(actual: &[f32], received: &[f32]) -> f32 {
    if received.is_empty() {
        return 0.0;
    }

    let mut sum_squared_log = 0.0;
    let n = received.len() as f32;

    for (index, received_value) in received.iter().enumerate() {
        let log_actual = (actual[index] + 1.0).ln();
        let log_received = (received_value + 1.0).ln();
        let diff_log = log_actual - log_received;
        sum_squared_log += diff_log * diff_log;
    }

    sum_squared_log / n
}
