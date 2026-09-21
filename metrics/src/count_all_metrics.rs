use std::println;

use crate::{mape, mase, msle, smape, wape};

pub fn count_all_metrics(actual: &[f32], received: &[f32]) {
    let mape_result = mape(actual, received);
    let wape_result = wape(actual, received);
    let smape_result = smape(actual, received);
    let msle_result = msle(actual, received);
    let mase_result = mase(actual, received);

    println!(
        "mape: {} , wape: {} , smape: {}, msle: {}, mase: {}",
        mape_result, wape_result, smape_result, msle_result, mase_result,
    )
}
