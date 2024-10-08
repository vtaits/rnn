use super::structures::LoggerEvent;

pub trait Logger {
    fn log_event(&mut self, event: LoggerEvent);
}
