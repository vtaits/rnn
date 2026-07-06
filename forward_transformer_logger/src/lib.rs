use rnn_architecture::ForwardTransformer;

pub struct ForwardTransformerLogger<DataType> {
    forward_transformer: Box<dyn ForwardTransformer<DataType>>,
}

impl<DataType> ForwardTransformerLogger<DataType> {
    pub fn new(forward_transformer: Box<dyn ForwardTransformer<DataType>>) -> Self {
        Self {
            forward_transformer,
        }
    }
}

impl<DataType> ForwardTransformer<DataType> for ForwardTransformerLogger<DataType> {
    fn transform(&self, data: DataType) -> Vec<bool> {
        let result = self.forward_transformer.transform(data);

        print!("F ");

        for value in result.iter() {
            print!("{}", if *value { "+" } else { "." });
        }

        println!();

        result
    }
}
