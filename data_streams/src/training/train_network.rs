use rnn_core::Network;
use timeline_helpers::ComplexTimelineValue;

use super::ComplexStream;

pub fn train_network(network: &mut Network<Vec<ComplexTimelineValue>>, steam: &ComplexStream) {
    while !steam.is_finish() {
        let data = steam.get_value();

        network.push_data(data);
    }
}
