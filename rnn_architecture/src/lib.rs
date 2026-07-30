pub trait Memory {
    fn reset_neurons(&mut self);

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
        memory: &mut dyn Memory,
        field_1_x: usize,
        field_1_y: usize,
        field_2_x: usize,
        field_2_y: usize,
        shift_x: i32,
        shift_y: i32,
    );

    fn connect_2_to_1(
        &mut self,
        memory: &mut dyn Memory,
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
        fields_connector: &mut dyn FieldsConnector,
        memory: &mut dyn Memory,
    );
}

pub trait Processor {
    fn input_signal(&mut self, signal: Vec<bool>);

    fn reset_neurons(&mut self);

    fn transfer_1_to_2(&mut self);
    fn transfer_2_to_1(&mut self);

    fn read_signal(&self) -> Vec<bool>;
}

pub trait ResultReader {
    fn write_single_signal(&mut self, signal: Vec<bool>);
    fn read_full_signal(&self) -> Vec<Vec<bool>>;
}

pub trait TickController {
    fn write_single_signal(&mut self, signal: Vec<bool>);

    fn reset_neurons(&mut self);

    fn transfer_signal(&mut self);

    fn attach_reader(&mut self, result_reader: Box<dyn ResultReader>);

    fn detach_reader(&mut self) -> Box<dyn ResultReader>;
}

pub trait BinaryController {
    fn push_single_signal(&mut self, signal: Vec<bool>);

    fn reset_neurons(&mut self);

    fn predict(&mut self, depth: usize) -> Result<Vec<Vec<bool>>, ()>;
}

pub trait ForwardTransformer<DataType> {
    fn transform(&self, data: DataType) -> Vec<bool>;
}

pub trait InverseTransformer<DataType> {
    fn transform(&self, signal: Vec<bool>) -> DataType;
}

pub trait HighLevelController<DataType> {
    fn push(&mut self, data: DataType);

    fn reset_neurons(&mut self);

    fn predict(&mut self, depth: usize) -> Result<Vec<DataType>, ()>;
}

pub trait DataProvider<DataType> {
    fn iterate_training_data(&self) -> Box<dyn Iterator<Item = DataType>>;

    fn iterate_test_data(&self) -> Box<dyn Iterator<Item = DataType>>;
}

pub trait ExperimentResult<DataType> {
    fn get_original_data(&self) -> &[DataType];

    fn get_generated_data(&self) -> &[DataType];
}

pub trait SingleExperiment<DataType> {
    fn execute(&mut self) -> Box<dyn ExperimentResult<DataType>>;
}

pub trait FullExperiment<DataType> {
    fn execute(&self) -> Vec<Box<dyn ExperimentResult<DataType>>>;
}
