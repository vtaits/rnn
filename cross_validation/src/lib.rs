use rnn_architecture::{ExperimentResult, FullExperiment, SingleExperiment};

pub struct CrossValidation<DataType> {
    experiment_count: usize,
    create_expreriment: Box<dyn Fn(usize) -> Box<dyn SingleExperiment<DataType>>>,
}

impl<DataType> CrossValidation<DataType> {
    pub fn new(
        experiment_count: usize,
        create_expreriment: Box<dyn Fn(usize) -> Box<dyn SingleExperiment<DataType>>>,
    ) -> Self {
        Self {
            experiment_count,
            create_expreriment,
        }
    }
}

impl<DataType> FullExperiment<DataType> for CrossValidation<DataType> {
    fn execute(&self) -> Vec<Box<dyn ExperimentResult<DataType>>> {
        let mut result = vec![];

        for experiment_index in 0..self.experiment_count {
            let mut experiment = (self.create_expreriment)(experiment_index);

            result.push(experiment.execute());
        }

        result
    }
}
