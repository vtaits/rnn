mod training;

pub use training::{
    init_training_stream_by_config, train_network, ComplexStream, CsvStream, TrainingStream,
    TrainingStreamConfig,
};
