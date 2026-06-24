use rnn_architecture::{BinaryController, ResultReader, TickController};

pub struct SyncBinaryController {
    tick_controller: Box<dyn TickController>,
    create_result_reader: Box<dyn Fn() -> Box<dyn ResultReader>>,
}

impl SyncBinaryController {
    pub fn new(
        tick_controller: Box<dyn TickController>,
        create_result_reader: Box<dyn Fn() -> Box<dyn ResultReader>>,
    ) -> Self {
        Self {
            tick_controller,
            create_result_reader,
        }
    }
}

impl BinaryController for SyncBinaryController {
    fn push_single_signal(&mut self, signal: Vec<bool>) {
        self.tick_controller.write_single_signal(signal);
    }

    fn predict(&mut self, depth: usize) -> Result<Vec<Vec<bool>>, ()> {
        let result_reader = (self.create_result_reader)();
        self.tick_controller.attach_reader(result_reader);

        for _ in 0..depth {
            self.tick_controller.transfer_signal();
        }

        let result_reader = self.tick_controller.detach_reader();

        Result::Ok(result_reader.read_full_signal())
    }
}
