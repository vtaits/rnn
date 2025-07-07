use tokio;

use console_ui::run_console_app;
use rnn_instance::{init_data_layer_by_env, InitDataLayerParams};

#[tokio::main]
async fn main() -> Result<(), ()> {
    let (data_layer, _) = init_data_layer_by_env(&InitDataLayerParams {
        train: false,
        start_index: None,
        end_index: None,
        start_measurement_index: None,
        end_measurement_index: None,
        redefine_params: None,
    });

    let _ = run_console_app(data_layer.get_network()).await;

    Ok(())
}
