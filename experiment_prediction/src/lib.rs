use rnn_architecture::{DataProvider, ExperimentResult, HighLevelController, SingleExperiment};

pub struct ExprerimentPrediction<DataType> {
    data_provider: Box<dyn DataProvider<DataType>>,
    high_level_controller: Box<dyn HighLevelController<DataType>>,
    create_expreriment_result:
        Box<dyn Fn(Vec<DataType>, Vec<DataType>) -> Box<dyn ExperimentResult<DataType>>>,
}

impl<DataType> ExprerimentPrediction<DataType> {
    pub fn new(
        data_provider: Box<dyn DataProvider<DataType>>,
        high_level_controller: Box<dyn HighLevelController<DataType>>,
        create_expreriment_result: Box<
            dyn Fn(Vec<DataType>, Vec<DataType>) -> Box<dyn ExperimentResult<DataType>>,
        >,
    ) -> Self {
        Self {
            data_provider,
            high_level_controller,
            create_expreriment_result,
        }
    }
}

impl<DataType> SingleExperiment<DataType> for ExprerimentPrediction<DataType> {
    fn execute(&mut self) -> Box<dyn ExperimentResult<DataType>> {
        let mut original_data = vec![];

        for datum in self.data_provider.as_ref().iterate_training_data() {
            self.high_level_controller.push(datum);
        }

        self.high_level_controller.reset_neurons();

        let mut is_first_item = true;

        for datum in self.data_provider.as_ref().iterate_test_data() {
            if is_first_item {
                self.high_level_controller.push(datum);
                is_first_item = false;
            } else {
                original_data.push(datum);
            }
        }

        let generated_data = self
            .high_level_controller
            .predict(original_data.len())
            .unwrap();

        (self.create_expreriment_result)(original_data, generated_data)
    }
}
