use super::{structures::TrainingStreamConfig, CsvDateTimeStream, CsvStream, TrainingStream};

pub fn init_training_stream_by_config(config: &TrainingStreamConfig) -> Box<dyn TrainingStream> {
    match config {
        TrainingStreamConfig::Csv(config) => Box::new(CsvStream::from_config(config)),
        TrainingStreamConfig::CsvDateTime(config) => {
            Box::new(CsvDateTimeStream::from_config(config))
        }
    }
}
