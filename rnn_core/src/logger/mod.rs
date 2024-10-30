mod logger;
mod multiple_file_logger;
mod structures;

pub use logger::Logger;
pub use multiple_file_logger::{MultipleFileLogger, MultipleFileLoggerParams};
pub use structures::LoggerEvent;
