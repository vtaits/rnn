use rnn_architecture::{ResultReader, ResultReaderFactory};

use crate::simple_result_reader::SimpleResultReader;

pub struct SimpleResultReaderFactory();

impl SimpleResultReaderFactory {
    pub fn new() -> Self {
        Self {}
    }
}

impl ResultReaderFactory for SimpleResultReaderFactory {
    fn create(&self) -> Box<dyn ResultReader> {
        Box::new(SimpleResultReader::new())
    }
}
