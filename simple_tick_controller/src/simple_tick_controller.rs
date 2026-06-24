use rnn_architecture::{Processor, ResultReader, TickController};

pub struct SimpleTickController {
    processor: Box<dyn Processor>,
    result_reader: Option<Box<dyn ResultReader>>,
}

impl SimpleTickController {
    pub fn new(processor: Box<dyn Processor>) -> Self {
        Self {
            processor,
            result_reader: None,
        }
    }

    fn get_transfer_fields(
        &mut self,
    ) -> (&mut Box<dyn Processor>, &mut Option<Box<dyn ResultReader>>) {
        (&mut self.processor, &mut self.result_reader)
    }
}

impl TickController for SimpleTickController {
    fn write_single_signal(&mut self, signal: Vec<bool>) {
        self.processor.input_signal(signal);
    }

    fn transfer_signal(&mut self) {
        let (processor, result_reader) = self.get_transfer_fields();

        processor.transfer_1_to_2();

        if let Some(result_reader) = result_reader {
            result_reader.write_single_signal(processor.read_signal());
        }

        processor.transfer_2_to_1();
    }

    fn attach_reader(&mut self, result_reader: Box<dyn ResultReader>) {
        self.result_reader = Some(result_reader);
    }

    fn detach_reader(&mut self) -> Box<dyn ResultReader> {
        self.result_reader
            .take()
            .expect("Result reader is not attached")
    }
}
