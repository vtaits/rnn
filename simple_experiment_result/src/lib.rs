use rnn_architecture::ExperimentResult;

pub struct SimpleExperimentResult<DataType> {
    original_data: Vec<DataType>,
    generated_data: Vec<DataType>,
}

impl<DataType> SimpleExperimentResult<DataType> {
    pub fn new(original_data: Vec<DataType>, generated_data: Vec<DataType>) -> Self {
        Self {
            original_data,
            generated_data,
        }
    }
}

impl<DataType> ExperimentResult<DataType> for SimpleExperimentResult<DataType> {
    fn get_generated_data(&self) -> &[DataType] {
        &self.generated_data
    }

    fn get_original_data(&self) -> &[DataType] {
        &self.original_data
    }
}
