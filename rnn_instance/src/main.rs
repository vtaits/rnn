use console_ui::run_console_app;
use rnn_instance::init_by_toml;

fn main() {
    let network = init_by_toml("../media/network.toml");

    let _ = run_console_app(Box::new(network));
}
