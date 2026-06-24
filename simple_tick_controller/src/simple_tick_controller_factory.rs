use rnn_architecture::{ProcessorFactory, TickController, TickControllerFactory};

use crate::SimpleTickController;

pub struct SimpleTickControllerFactory {
    processor_factory: Box<dyn ProcessorFactory>,
}

impl SimpleTickControllerFactory {
    pub fn new(processor_factory: Box<dyn ProcessorFactory>) -> Self {
        Self { processor_factory }
    }
}

impl TickControllerFactory for SimpleTickControllerFactory {
    fn create_tick_controller(&self) -> Box<dyn TickController> {
        Box::new(SimpleTickController::new(
            self.processor_factory.create_processor(),
        ))
    }
}
