use rnn_architecture::{BinaryController, ResultReaderFactory, TickController};

pub struct SyncBinaryController {
    tick_controller: Box<dyn TickController>,
    result_reader_factory: Box<dyn ResultReaderFactory>,
}

impl SyncBinaryController {
    pub fn new(
        tick_controller: Box<dyn TickController>,
        result_reader_factory: Box<dyn ResultReaderFactory>,
    ) -> Self {
        Self {
            tick_controller,
            result_reader_factory,
        }
    }
}

impl BinaryController for SyncBinaryController {
    fn push_single_signal(&mut self, signal: Vec<bool>) {
        self.tick_controller.write_single_signal(signal);
    }

    fn predict(&mut self, depth: usize) -> Result<Vec<Vec<bool>>, ()> {
        let result_reader = self.result_reader_factory.create();
        self.tick_controller.attach_reader(result_reader);

        for _ in 0..depth {
            self.tick_controller.transfer_signal();
        }

        let result_reader = self.tick_controller.detach_reader();

        Result::Ok(result_reader.read_full_signal())
    }
}
