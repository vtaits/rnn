use std::path::Path;
use std::{sync::Arc, sync::RwLock};
use std::fs::File;
use std::io::Write;

use data_streams::{train_network, ComplexStream, TrainingStream};
use rnn_core::{DataLayer, DataLayerParams, LayerParams, Logger, LoggerEvent, Network, SynapseParams};
use timeline_helpers::{ComplexTimeline, ComplexTimelineValue, Timeline};

pub struct FileLogger {
    file: File,
}

impl FileLogger {
    pub fn new<P: AsRef<Path>>(path: P) -> Self {
        let file = File::create(path).unwrap();

        FileLogger {
            file
        }
    }
}

impl Logger for FileLogger {
    fn log_event(&mut self, logger_event: LoggerEvent) {
        match logger_event {
            LoggerEvent::ChangeLayerWeights(_, inc, dec) => {
                let _ = writeln!(
                    self.file,
                    "{} {}",
                    inc,
                    dec
                );
            },
            _ => {}
        };
    }
}

pub fn init_data_layer(
    layer_params: LayerParams,
    synapse_params: SynapseParams,
    timelines: Vec<Box<dyn Timeline>>,
    training_streams: Vec<Box<dyn TrainingStream>>,
) -> DataLayer<Vec<ComplexTimelineValue>> {
    let mut complex_stream = ComplexStream::new(training_streams);

    let complex_timeline = Arc::new(ComplexTimeline::new(timelines));

    let network = Network::new(layer_params, synapse_params, Some(Box::new(FileLogger::new("data.txt"))));

    let mut data_layer = DataLayer::new(
        DataLayerParams {
            data_to_binary: {
                let complex_timeline = Arc::clone(&complex_timeline);

                Box::new(move |data: Vec<ComplexTimelineValue>| complex_timeline.get_bits(&data))
            },
            binary_to_data: {
                let complex_timeline = Arc::clone(&complex_timeline);

                Box::new(move |binary| Ok(complex_timeline.reverse(&binary)))
            },
        },
        Arc::new(RwLock::new(network)),
    );

    train_network(&mut data_layer, &mut complex_stream);

    data_layer
}
