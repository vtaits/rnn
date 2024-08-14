use std::rc::Rc;

use data_streams::{train_network, ComplexStream, TrainingStream};
use rnn_core::{DataAdapter, Network, NetworkParams, SynapseParams};
use timeline_helpers::{ComplexTimeline, ComplexTimelineItem, ComplexTimelineValue};

pub fn init_network(
    params: NetworkParams,
    synapse_params: SynapseParams,
    timelines: Vec<ComplexTimelineItem>,
    streams: Vec<Box<dyn TrainingStream>>,
) -> Network<Vec<ComplexTimelineValue>> {
    let complex_stream = ComplexStream::new(streams);

    let complex_timeline = Rc::new(ComplexTimeline::new(timelines));

    let data_adapter = DataAdapter {
        data_to_binary: {
            let complex_timeline = Rc::clone(&complex_timeline);

            Box::new(move |data: Vec<ComplexTimelineValue>| complex_timeline.get_bits(&data))
        },
        binary_to_data: {
            let complex_timeline = Rc::clone(&complex_timeline);

            Box::new(move |binary| Ok(complex_timeline.reverse(&binary)))
        },
    };

    let mut network = Network::new(params, synapse_params, data_adapter);

    train_network(&mut network, &complex_stream);

    network
}
