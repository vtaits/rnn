use std::println;

use ocl::{Buffer, Kernel, ProQue};
use processor_opencl_full::ProcessorOpenCLFullSignalTransferer;

pub struct SignalTransfererOpenCLFullParams {
    pub field_width: usize,
    pub field_height: usize,
    pub layer_width: usize,
    pub layer_height: usize,
    pub threshold: f32,
    pub gamma_inc: f32,
    pub gamma_dec: f32,
    pub g_inc: f32,
    pub g_dec: f32,
    pub min_g: f32,
    pub max_g: f32,
}

pub struct SignalTransfererOpenCLFull {
    kernel: Kernel,
    pro_que: ProQue,
    layer_size: usize,
    threshold: f32,
    gamma_inc: f32,
    gamma_dec: f32,
    g_inc: f32,
    g_dec: f32,
    min_g: f32,
    max_g: f32,
}

impl SignalTransfererOpenCLFull {
    pub fn new(params: SignalTransfererOpenCLFullParams) -> Self {
        let SignalTransfererOpenCLFullParams {
            field_width,
            field_height,
            layer_width,
            layer_height,
            threshold,
            gamma_inc,
            gamma_dec,
            g_inc,
            g_dec,
            min_g,
            max_g,
        } = params;

        let kernel_source = include_str!("signal_transferer_full_opencl.cl");

        let pro_que = ProQue::builder().src(kernel_source).build().unwrap();

        let layer_size = field_width * field_height * layer_width * layer_height;

        let kernel = Kernel::builder()
            .program(pro_que.program())
            .name("signal_transferer_full_opencl")
            .queue(pro_que.queue().clone())
            .global_work_size(layer_size)
            .arg_named("g_0", 0.0_f32)
            .arg_named("neurons_from", None::<&Buffer<u8>>)
            .arg_named("neurons_to", None::<&Buffer<u8>>)
            .arg_named("refract_intervals_to", None::<&Buffer<u8>>)
            .arg_named("synapses", None::<&Buffer<f32>>)
            .arg_named("distances", None::<&Buffer<f32>>)
            .arg_named("layer_size", 0_u32)
            .arg_named("threshold", 0.0_f32)
            .arg_named("gamma_inc", 0.0_f32)
            .arg_named("gamma_dec", 0.0_f32)
            .arg_named("g_dec", 0.0_f32)
            .arg_named("g_inc", 0.0_f32)
            .arg_named("min_g", 0.0_f32)
            .arg_named("max_g", 0.0_f32)
            .build()
            .unwrap();

        Self {
            pro_que,
            kernel,
            layer_size,
            threshold,
            gamma_inc,
            gamma_dec,
            g_inc,
            g_dec,
            min_g,
            max_g,
        }
    }
}

impl ProcessorOpenCLFullSignalTransferer for SignalTransfererOpenCLFull {
    fn get_pro_que(&self) -> &ProQue {
        &self.pro_que
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
        println!("=================");
        for (index, neuron) in neurons_from.iter().enumerate() {
            if index % 81 == 0 {
                println!();
            }
            print!("{}", if *neuron > 0 { "+" } else { "." });
        }
        println!();

        let buffer_neurons_from = Buffer::<u8>::builder()
            .queue(self.pro_que.queue().clone())
            .len(neurons_from.len())
            .copy_host_slice(neurons_from)
            .build()
            .unwrap();

        let buffer_refract_intervals_to = Buffer::<u8>::builder()
            .queue(self.pro_que.queue().clone())
            .len(refract_intervals_to.len())
            .copy_host_slice(refract_intervals_to)
            .build()
            .unwrap();

        let buffer_neurons_to = Buffer::<u8>::builder()
            .queue(self.pro_que.queue().clone())
            .flags(ocl::flags::MEM_READ_WRITE)
            .len(self.layer_size)
            .build()
            .unwrap();

        unsafe {
            self.kernel.set_arg("g_0", g_0).unwrap();
            self.kernel
                .set_arg("neurons_from", &buffer_neurons_from)
                .unwrap();
            self.kernel
                .set_arg("neurons_to", &buffer_neurons_to)
                .unwrap();
            self.kernel
                .set_arg("refract_intervals_to", &buffer_refract_intervals_to)
                .unwrap();
            self.kernel.set_arg("synapses", synapses).unwrap();
            self.kernel.set_arg("distances", distances).unwrap();
            self.kernel
                .set_arg("layer_size", self.layer_size as u32)
                .unwrap();
            self.kernel.set_arg("threshold", self.threshold).unwrap();
            self.kernel.set_arg("gamma_inc", self.gamma_inc).unwrap();
            self.kernel.set_arg("gamma_dec", self.gamma_dec).unwrap();
            self.kernel.set_arg("g_dec", self.g_dec).unwrap();
            self.kernel.set_arg("g_inc", self.g_inc).unwrap();
            self.kernel.set_arg("min_g", self.min_g).unwrap();
            self.kernel.set_arg("max_g", self.max_g).unwrap();
            self.kernel.enq().unwrap();
        }

        buffer_neurons_to.read(&mut *neurons_to).enq().unwrap();

        for (index, neuron) in neurons_to.iter().enumerate() {
            if index % 81 == 0 {
                println!();
            }
            print!("{}", if *neuron > 0 { "+" } else { "." });
        }
        println!();
        println!("=================");
    }
}
