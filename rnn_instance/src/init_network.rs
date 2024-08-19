use std::rc::Rc;

use data_streams::{train_network, ComplexStream, TrainingStream};
use rnn_core::{DataAdapter, LayerParams, Network, SynapseParams};
use timeline_helpers::{ComplexTimeline, ComplexTimelineValue, Timeline};

pub fn init_network(
    layer_params: LayerParams,
    synapse_params: SynapseParams,
    timelines: Vec<Box<dyn Timeline>>,
    training_streams: Vec<Box<dyn TrainingStream>>,
) -> Network<Vec<ComplexTimelineValue>> {
    let mut complex_stream = ComplexStream::new(training_streams);

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

    let mut network = Network::new(layer_params, synapse_params, data_adapter);

    train_network(&mut network, &mut complex_stream);

    network
}
