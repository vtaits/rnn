use rnn_architecture::ResultReader;

pub struct SimpleResultReader {
    result: Vec<Vec<bool>>,
}

impl SimpleResultReader {
    pub fn new() -> Self {
        Self { result: vec![] }
    }
}

impl ResultReader for SimpleResultReader {
    fn write_single_signal(&mut self, signal: Vec<bool>) {
        self.result.push(signal);
    }

    fn read_full_signal(&self) -> Vec<Vec<bool>> {
        self.result.clone()
    }
}
