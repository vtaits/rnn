use std::println;

use ocl::{Buffer, Context, Device, Kernel, Program, Queue};
use processor_opencl_full::ProcessorOpenCLFullSignalTransferer;

use crate::excite_neurons_with_partitions::excite_neurons_with_partitions;

pub struct SignalTransfererOpenCLWinnerParams<'a> {
    pub context: &'a Context,
    pub device: Device,
    pub field_width: usize,
    pub field_height: usize,
    pub layer_width: usize,
    pub layer_height: usize,
    pub threshold: f32,
    pub gamma_inc: f32,
    pub gamma_dec: f32,
    pub partitions: Vec<usize>,
}

pub struct SignalTransfererOpenCLWinner {
    kernel: Kernel,
    queue: Queue,
    buffer_signals_to: Buffer<f32>,
    layer_size: usize,
    field_size: usize,
    field_count: usize,
    threshold: f32,
    gamma_inc: f32,
    gamma_dec: f32,
    partitions: Vec<usize>,
}

impl SignalTransfererOpenCLWinner {
    pub fn new(params: SignalTransfererOpenCLWinnerParams) -> Self {
        let SignalTransfererOpenCLWinnerParams {
            context,
            device,
            field_width,
            field_height,
            layer_width,
            layer_height,
            threshold,
            gamma_inc,
            gamma_dec,
            partitions,
        } = params;

        let source = include_str!("signal_transferer_winner_opencl.cl");

        let queue = Queue::new(&context, device, None).unwrap();

        let program = Program::builder()
            .src(source)
            .devices(device)
            .build(&context)
            .unwrap();

        let layer_size = field_width * field_height * layer_width * layer_height;

        let kernel = Kernel::builder()
            .program(&program)
            .name("signal_transferer_winner_opencl")
            .queue(queue.clone())
            .global_work_size(layer_size)
            .arg_named("g_0", 0.0_f32)
            .arg_named("neurons_from", None::<&Buffer<u8>>)
            .arg_named("signals_to", None::<&Buffer<f32>>)
            .arg_named("refract_intervals_to", None::<&Buffer<u8>>)
            .arg_named("synapses", None::<&Buffer<f32>>)
            .arg_named("distances", None::<&Buffer<f32>>)
            .arg_named("layer_size", 0_u32)
            .arg_named("gamma_inc", 0.0_f32)
            .arg_named("gamma_dec", 0.0_f32)
            .build()
            .unwrap();

        let buffer_signals_to = Buffer::<f32>::builder()
            .queue(queue.clone())
            .flags(ocl::flags::MEM_READ_WRITE)
            .len(layer_size)
            .build()
            .unwrap();

        Self {
            queue,
            kernel,
            buffer_signals_to,
            layer_size,
            threshold,
            gamma_inc,
            gamma_dec,
            field_size: field_width * field_height,
            field_count: layer_width * layer_height,
            partitions,
        }
    }
}

impl ProcessorOpenCLFullSignalTransferer for SignalTransfererOpenCLWinner {
    fn get_queue(&self) -> &Queue {
        &self.queue
    }

    fn transfer(
        &self,
        g_0: f32,
        neurons_from: &[u8],
        neurons_to: &mut [u8],
        refract_intervals_to: &[u8],
        synapses: &Buffer<f32>,
        distances: &Buffer<f32>,
    ) {
        /* println!("=================");
        for (index, neuron) in neurons_from.iter().enumerate() {
            if index % 81 == 0 {
                println!();
            }
            print!("{}", if *neuron > 0 { "+" } else { "." });
        }
        println!(); */

        let buffer_neurons_from = Buffer::<u8>::builder()
            .queue(self.queue.clone())
            .len(neurons_from.len())
            .copy_host_slice(neurons_from)
            .build()
            .unwrap();

        let buffer_refract_intervals_to = Buffer::<u8>::builder()
            .queue(self.queue.clone())
            .len(refract_intervals_to.len())
            .copy_host_slice(refract_intervals_to)
            .build()
            .unwrap();

        unsafe {
            self.kernel.set_arg("g_0", g_0).unwrap();
            self.kernel
                .set_arg("neurons_from", &buffer_neurons_from)
                .unwrap();
            self.kernel
                .set_arg("signals_to", &self.buffer_signals_to)
                .unwrap();
            self.kernel
                .set_arg("refract_intervals_to", &buffer_refract_intervals_to)
                .unwrap();
            self.kernel.set_arg("synapses", synapses).unwrap();
            self.kernel.set_arg("distances", distances).unwrap();
            self.kernel
                .set_arg("layer_size", self.layer_size as u32)
                .unwrap();
            self.kernel.set_arg("gamma_inc", self.gamma_inc).unwrap();
            self.kernel.set_arg("gamma_dec", self.gamma_dec).unwrap();
            self.kernel.enq().unwrap();
        }

        let mut signals_to = vec![0.0; self.layer_size];

        self.buffer_signals_to.read(&mut *signals_to).enq().unwrap();

        excite_neurons_with_partitions(
            neurons_to,
            &signals_to,
            &self.field_size,
            &self.field_count,
            &self.threshold,
            &self.partitions,
        );

        /* for (index, neuron) in neurons_to.iter().enumerate() {
            if index % 81 == 0 {
                println!();
            }
            print!("{}", if *neuron > 0 { "+" } else if refract_intervals_to[index] > 0 { "R" } else { "." });
        }
        println!();
        println!("================="); */
    }
}
