use rnn_architecture::{
    BinaryControllerFactory, ForwardTransformerFactory, HighLevelController,
    HighLevelControllerFactory, InverseTransformerFactory, ResultReaderFactory,
};

use crate::simple_high_level_controller::SimpleHighLevelController;

pub struct SimpleHighLevelControllerFactory<DataType> {
    forward_transformer_factory: Box<dyn ForwardTransformerFactory<DataType>>,
    inverse_transformer_factory: Box<dyn InverseTransformerFactory<DataType>>,
    binary_controller_factory: Box<dyn BinaryControllerFactory>,
}

impl<DataType> SimpleHighLevelControllerFactory<DataType> {
    pub fn new(
        forward_transformer_factory: Box<dyn ForwardTransformerFactory<DataType>>,
        inverse_transformer_factory: Box<dyn InverseTransformerFactory<DataType>>,
        binary_controller_factory: Box<dyn BinaryControllerFactory>,
    ) -> Self {
        Self {
            forward_transformer_factory,
            inverse_transformer_factory,
            binary_controller_factory,
        }
    }
}

impl<DataType: 'static> HighLevelControllerFactory<DataType>
    for SimpleHighLevelControllerFactory<DataType>
{
    fn create_high_level_controller(
        &self,
        result_reader_factory: Box<dyn ResultReaderFactory>,
    ) -> Box<dyn HighLevelController<DataType>> {
        Box::new(SimpleHighLevelController::new(
            self.forward_transformer_factory
                .create_forward_transformer(),
            self.inverse_transformer_factory
                .create_inverse_transformer(),
            self.binary_controller_factory
                .create_binary_controller(result_reader_factory),
        ))
    }
}
