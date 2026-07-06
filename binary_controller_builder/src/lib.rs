use block_sequence_spiral::BlockSequenceSpiral;
use full_fields_connector::FullFieldsConnector;
use memory_cpu_full::MemoryCpuFull;
use power_law_distance::PowerLawDistance;
use processor_cpu_full::{ProcessorCpuFull, ProcessorCpuFullMemory, ProcessorCpuFullParams};
use refract_recounter_cpu::RefractRecounterCpu;
use rnn_architecture::{
    BinaryController, BlockSequence, FieldsConnector, Processor, ResultReader, Topology,
};
use row_col_coordinates_resolver::RowColCoordinatesResolver;
use signal_transferer_full_cpu::{SignalTransfererCpuFull, SignalTransfererCpuFullParams};
use simple_result_reader::SimpleResultReader;
use simple_tick_controller::SimpleTickController;
use sync_binary_controller::SyncBinaryController;
use topology_restore_first::TopologyRestoreFirst;

pub struct BinaryControllerBuilder {
    field_width: Option<usize>,
    field_height: Option<usize>,
    layer_width: Option<usize>,
    layer_height: Option<usize>,
    alpha: f32,
    gamma_dec: f32,
    gamma_inc: f32,
    g_dec: f32,
    g_inc: f32,
    g_0: f32,
    min_g: f32,
    max_g: f32,
    h: f32,
    refract_interval: u8,
    threshold: f32,
}

impl BinaryControllerBuilder {
    pub fn new() -> Self {
        Self {
            field_width: None,
            field_height: None,
            layer_width: None,
            layer_height: None,
            alpha: 3.0,
            gamma_dec: 0.5,
            gamma_inc: 0.5,
            g_dec: 0.1,
            g_inc: 0.1,
            g_0: 1.0,
            min_g: -10.0,
            max_g: 10.0,
            h: 0.5,
            refract_interval: 3,
            threshold: 0.9,
        }
    }

    pub fn set_field_width(&mut self, field_width: usize) {
        self.field_width = Some(field_width);
    }

    pub fn set_field_height(&mut self, field_height: usize) {
        self.field_height = Some(field_height);
    }

    pub fn set_layer_width(&mut self, layer_width: usize) {
        self.layer_width = Some(layer_width);
    }

    pub fn set_layer_height(&mut self, layer_height: usize) {
        self.layer_height = Some(layer_height);
    }

    fn build_fields_connector(&self) -> Box<dyn FieldsConnector> {
        let BinaryControllerBuilder {
            field_width,
            field_height,
            layer_width,
            alpha,
            max_g,
            h,
            ..
        } = self;

        let field_width = field_width.unwrap();
        let field_height = field_height.unwrap();
        let layer_width = layer_width.unwrap();

        let distance_between_neurons = Box::new(PowerLawDistance::new(*alpha, *h));

        let neuron_coordinates_resolver = Box::new(RowColCoordinatesResolver::new(
            layer_width,
            field_width,
            field_height,
        ));

        let fields_connector = FullFieldsConnector::new(
            field_width,
            field_height,
            *max_g,
            distance_between_neurons,
            neuron_coordinates_resolver,
        );

        Box::new(fields_connector) as Box<dyn FieldsConnector>
    }

    fn build_cpu_full_processor(
        &self,
        block_sequence: Box<dyn BlockSequence>,
        topology: Box<dyn Topology>,
    ) -> Box<dyn Processor> {
        let BinaryControllerBuilder {
            field_width,
            field_height,
            layer_width,
            layer_height,
            gamma_dec,
            gamma_inc,
            g_dec,
            g_inc,
            g_0,
            min_g,
            max_g,
            refract_interval,
            threshold,
            ..
        } = self;

        let field_width = field_width.unwrap();
        let field_height = field_height.unwrap();
        let layer_width = layer_width.unwrap();
        let layer_height = layer_height.unwrap();

        let mut memory = MemoryCpuFull::new(field_width, field_height, layer_width, layer_height);

        let signal_transferer = SignalTransfererCpuFull::new(SignalTransfererCpuFullParams {
            field_width,
            field_height,
            layer_width,
            layer_height,
            threshold: *threshold,
            gamma_inc: *gamma_inc,
            gamma_dec: *gamma_dec,
            g_inc: *g_inc,
            g_dec: *g_dec,
            min_g: *min_g,
            max_g: *max_g,
        });

        let refract_recounter = RefractRecounterCpu::new(*refract_interval);

        let mut fields_connector = self.build_fields_connector();

        topology.fill(block_sequence, fields_connector.as_mut(), &mut memory);

        Box::new(ProcessorCpuFull::new(
            ProcessorCpuFullParams {
                g_0: *g_0,
                field_width,
                field_height,
            },
            Box::new(memory) as Box<dyn ProcessorCpuFullMemory>,
            Box::new(signal_transferer),
            Box::new(refract_recounter),
        ))
    }

    pub fn build(&self) -> Box<dyn BinaryController> {
        let BinaryControllerBuilder {
            layer_width,
            layer_height,
            ..
        } = self;

        let layer_width = layer_width.unwrap();
        let layer_height = layer_height.unwrap();

        let block_sequence = Box::new(BlockSequenceSpiral::new(layer_width, layer_height));

        let topology = Box::new(TopologyRestoreFirst::new());

        let processor = self.build_cpu_full_processor(block_sequence, topology);

        let tick_controller = SimpleTickController::new(processor);

        Box::new(SyncBinaryController::new(
            Box::new(tick_controller),
            Box::new(|| Box::new(SimpleResultReader::new()) as Box<dyn ResultReader>),
        ))
    }
}
