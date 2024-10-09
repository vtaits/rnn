use super::structures::LoggerEvent;

pub trait Logger: Send + Sync {
    fn log_event(&mut self, event: LoggerEvent);
}
