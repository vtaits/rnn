pub trait Memory {
    fn set_distance_1_to_2(
        &mut self,
        neuron_from_index: usize,
        neuron_to_index: usize,
        distance: f32,
    );
    fn set_distance_2_to_1(
        &mut self,
        neuron_from_index: usize,
        neuron_to_index: usize,
        distance: f32,
    );
    fn set_synapse_weight_1_to_2(
        &mut self,
        neuron_from_index: usize,
        neuron_to_index: usize,
        weight: f32,
    );
    fn set_synapse_weight_2_to_1(
        &mut self,
        neuron_from_index: usize,
        neuron_to_index: usize,
        weight: f32,
    );
}

pub trait NeuronCoordinatesResolver {
    fn resolve(
        &self,
        field_x: usize,
        field_y: usize,
        neuron_in_field_x: usize,
        neuron_in_field_y: usize,
    ) -> usize;
}

pub trait DistanceBetweenNeurons {
    fn get_distance(&mut self, dx: i32, dy: i32) -> f32;
}

pub trait FieldsConnector {
    fn connect_1_to_2(
        &mut self,
        field_1_x: usize,
        field_1_y: usize,
        field_2_x: usize,
        field_2_y: usize,
        shift_x: i32,
        shift_y: i32,
    );
    fn connect_2_to_1(
        &mut self,
        field_1_x: usize,
        field_1_y: usize,
        field_2_x: usize,
        field_2_y: usize,
        shift_x: i32,
        shift_y: i32,
    );
}

pub trait BlockSequence {
    fn iterate_blocks(&self) -> Box<dyn Iterator<Item = (usize, usize)>>;
}

pub trait Topology {
    fn fill(
        &self,
        block_sequence: Box<dyn BlockSequence>,
        fields_connector: Box<dyn FieldsConnector>,
    );
}

pub trait Processor {
    fn input_signal(&mut self, signal: Vec<bool>);

    fn transfer_1_to_2(&mut self);
    fn transfer_2_to_1(&mut self);

    fn read_signal(&self) -> Vec<bool>;
}

pub trait ResultReader {
    fn write_single_signal(&mut self, signal: Vec<bool>);
    fn read_full_signal(&self) -> Vec<Vec<bool>>;
}

pub trait ResultReaderFactory {
    fn create(&self) -> Box<dyn ResultReader>;
}

pub trait TickController {
    fn write_single_signal(&mut self, signal: Vec<bool>);

    fn transfer_signal(&mut self);

    fn attach_reader(&mut self, result_reader: Box<dyn ResultReader>);

    fn detach_reader(&mut self) -> Box<dyn ResultReader>;
}

pub trait BinaryController {
    fn push_single_signal(&mut self, signal: Vec<bool>);

    fn predict(&mut self, depth: usize) -> Result<Vec<Vec<bool>>, ()>;
}

pub trait BinaryControllerFactory {
    fn create_binary_controller(&self) -> Box<dyn BinaryController>;
}

pub trait ForwardTransformer<DataType> {
    fn transform(&self, data: DataType) -> Vec<bool>;
}

pub trait InverseTransformer<DataType> {
    fn transform(&self, signal: Vec<bool>) -> DataType;
}

pub trait HighLevelController<DataType> {
    fn push(&mut self, data: DataType);

    fn predict(&mut self, depth: usize) -> Result<Vec<DataType>, ()>;
}

pub trait HighLevelControllerFactory<DataType> {
    fn create_high_level_controller(&self) -> Box<dyn HighLevelController<DataType>>;
}

pub trait DataProvider<DataType> {
    fn iterate_training_data(&self) -> Box<dyn Iterator<Item = DataType>>;

    fn iterate_test_data(&self) -> Box<dyn Iterator<Item = DataType>>;
}

pub trait DataProviderFactory<DataType> {
    fn create_data_provider(&self, experiment_index: usize) -> Box<dyn DataProvider<DataType>>;
}

pub trait ExperimentResult<DataType> {
    fn original_data(&self) -> Vec<DataType>;

    fn generated_data(&self) -> Vec<DataType>;
}

pub trait SingleExperiment<DataType> {
    fn execute(&self) -> dyn ExperimentResult<DataType>;
}

pub trait FullExperiment<DataType> {
    fn execute() -> Vec<Box<dyn ExperimentResult<DataType>>>;
}
