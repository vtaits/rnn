pub fn mase(actual: &[f32], received: &[f32]) -> f32 {
    if received.is_empty() {
        return 0.0;
    }

    // MAE модели
    let mae_model = received
        .iter()
        .enumerate()
        .map(|(index, received_value)| (actual[index] - received_value).abs())
        .sum::<f32>()
        / received.len() as f32;

    // MAE наивного прогноза (train_diffs = |y[t]-y[t-1]|)

    let diff_summ = {
        let mut prev = None;
        let mut res = 0.0;

        for actual_value in actual {
            if let Some(prev) = prev {
                let diff: f32 = prev - actual_value;
                res += diff.abs();
            }

            prev = Some(actual_value);
        }

        res
    };
    let mae_naive = diff_summ / (received.len() - 1) as f32;

    if mae_naive.abs() < 1e-8 {
        0.0
    } else {
        mae_model / mae_naive
    }
}
