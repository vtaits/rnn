use tokio;

use console_ui::run_console_app;
use rnn_instance::init_by_toml;

#[tokio::main]
async fn main() -> Result<(), ()> {
    let config_path = std::env::var("CONFIG_PATH").expect("CONFIG_PATH should be defined");
    let data_layer = init_by_toml(config_path);

    let _ = run_console_app(data_layer.get_network()).await;

    Ok(())
}
