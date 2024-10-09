use std::io::Write;
use std::{fs::File, path::Path};

use super::{Logger, LoggerEvent};

pub struct FileLogger {
    file: File,
}

impl FileLogger {
    pub fn new<P: AsRef<Path>>(path: P) -> Self {
        let file = File::create(path).unwrap();

        FileLogger { file }
    }
}

impl Logger for FileLogger {
    fn log_event(&mut self, logger_event: LoggerEvent) {
        match logger_event {
            LoggerEvent::ChangeLayerWeights(_, inc, dec) => {
                let _ = writeln!(self.file, "{} {}", inc, dec);
            }
            _ => {}
        };
    }
}
