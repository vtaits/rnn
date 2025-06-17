use tokio;

use console_ui::run_console_app;
use rnn_instance::{init_data_layer_by_env, InitDataLayerParams};

#[tokio::main]
async fn main() -> Result<(), ()> {
    let (data_layer, _) = init_data_layer_by_env(&InitDataLayerParams { train: true, end_measurement_index: None, start_measurement_index: None });

    let _ = run_console_app(data_layer.get_network()).await;

    Ok(())
}
