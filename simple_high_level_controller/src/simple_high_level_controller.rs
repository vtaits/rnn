use rnn_architecture::{
    BinaryController, ForwardTransformer, HighLevelController, InverseTransformer,
};

pub struct SimpleHighLevelController<DataType> {
    forward_transformer: Box<dyn ForwardTransformer<DataType>>,
    inverse_transformer: Box<dyn InverseTransformer<DataType>>,
    binary_controller: Box<dyn BinaryController>,
}

impl<DataType> SimpleHighLevelController<DataType> {
    pub fn new(
        forward_transformer: Box<dyn ForwardTransformer<DataType>>,
        inverse_transformer: Box<dyn InverseTransformer<DataType>>,
        binary_controller: Box<dyn BinaryController>,
    ) -> Self {
        Self {
            forward_transformer,
            inverse_transformer,
            binary_controller,
        }
    }
}

impl<DataType> HighLevelController<DataType> for SimpleHighLevelController<DataType> {
    fn predict(&mut self, depth: usize) -> Result<Vec<DataType>, ()> {
        self.binary_controller.predict(depth).map(|binary_result| {
            binary_result
                .into_iter()
                .map(|signal| self.inverse_transformer.transform(signal))
                .collect()
        })
    }

    fn reset_neurons(&mut self) {
        self.binary_controller.reset_neurons();
    }

    fn push(&mut self, data: DataType) {
        self.binary_controller
            .push_single_signal(self.forward_transformer.transform(data));
    }
}
