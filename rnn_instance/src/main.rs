use tokio;

use console_ui::run_console_app;
use rnn_instance::init_data_layer_by_env;

#[tokio::main]
async fn main() -> Result<(), ()> {
    let data_layer = init_data_layer_by_env(true);

    let _ = run_console_app(data_layer.get_network()).await;

    Ok(())
}
