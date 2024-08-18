use rnn_core::Network;
use timeline_helpers::ComplexTimelineValue;

use super::ComplexStream;

pub fn train_network(
    network: &mut Network<Vec<ComplexTimelineValue>>,
    complex_stream: &mut ComplexStream,
) {
    while !complex_stream.is_finish() {
        let data = complex_stream.get_value();

        network.push_data(data);

        complex_stream.step();
    }
}
