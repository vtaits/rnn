use rnn_architecture::InverseTransformer;

pub struct InverseTransformerLogger<DataType> {
    inverse_transformer: Box<dyn InverseTransformer<DataType>>,
}

impl<DataType> InverseTransformerLogger<DataType> {
    pub fn new(inverse_transformer: Box<dyn InverseTransformer<DataType>>) -> Self {
        Self {
            inverse_transformer,
        }
    }
}

impl<DataType> InverseTransformer<DataType> for InverseTransformerLogger<DataType> {
    fn transform(&self, signal: Vec<bool>) -> DataType {
        print!("I ");

        for value in signal.iter() {
            print!("{}", if *value { "+" } else { "." });
        }

        println!();

        self.inverse_transformer.transform(signal)
    }
}
