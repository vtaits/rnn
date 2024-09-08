use tokio;

use console_ui::run_console_app;
use rnn_instance::init_by_toml;

#[tokio::main]
async fn main() -> Result<(), ()> {
    let data_layer = init_by_toml("../media/network.toml");

    let _ = run_console_app(data_layer.get_network()).await;

    Ok(())
}
