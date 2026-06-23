use rnn_architecture::{
    BinaryControllerFactory, ForwardTransformer, HighLevelController, HighLevelControllerFactory,
    InverseTransformer, ResultReader, ResultReaderFactory,
};

use crate::simple_high_level_controller::SimpleHighLevelController;

pub struct SimpleHighLevelControllerFactory<DataType> {
    forward_transformer: Box<dyn ForwardTransformer<DataType>>,
    inverse_transformer: Box<dyn InverseTransformer<DataType>>,
    binary_controller_factory: Box<dyn BinaryControllerFactory>,
}

impl<DataType> SimpleHighLevelControllerFactory<DataType> {
    pub fn new(
        forward_transformer: Box<dyn ForwardTransformer<DataType>>,
        inverse_transformer: Box<dyn InverseTransformer<DataType>>,
        binary_controller_factory: Box<dyn BinaryControllerFactory>,
    ) -> Self {
        Self {
            forward_transformer,
            inverse_transformer,
            binary_controller_factory,
        }
    }
}

impl<DataType> HighLevelControllerFactory<DataType> for SimpleHighLevelControllerFactory<DataType> {
    fn create_high_level_controller(&self) -> Box<dyn HighLevelController<DataType>> {
        Box::new(SimpleHighLevelController::new(
            self.forward_transformer,
            self.inverse_transformer,
            self.binary_controller_factory.create_binary_controller(),
        ))
    }
}
