use tokio;

use console_ui::run_console_app;
use rnn_instance::init_by_toml;
use std::env;

#[tokio::main]
async fn main() -> Result<(), ()> {
    let config_dir = env::var("CONFIG_DIR").ok();
    let config_path = env::var("CONFIG_PATH").expect("CONFIG_PATH should be defined");
    let data_layer = init_by_toml(config_path, &config_dir);

    let _ = run_console_app(data_layer.get_network()).await;

    Ok(())
}
