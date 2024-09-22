use std::sync::{Arc, Mutex};

use ndarray::{Array1, Array2};
use ocl::{Buffer, Kernel, ProQue};

use crate::structures::CompiledKernel;

pub fn build_apply_synapses_kernel(layer_size: usize) -> ocl::Result<CompiledKernel> {
    let kernel_source = include_str!("apply_synapses.cl");

    let pro_que = ProQue::builder().src(kernel_source).build()?;

    let kernel = Kernel::builder()
        .program(pro_que.program())
        .name("apply_synapses")
        .queue(pro_que.queue().clone())
        .global_work_size(layer_size)
        .arg_named("accumulated_weights", None::<&Buffer<f32>>)
        .arg_named("distance_weights", None::<&Buffer<f32>>)
        .arg_named("neurons_from", None::<&Buffer<f32>>)
        .arg_named("refract_intervals_to", None::<&Buffer<u8>>)
        .arg_named("next_neurons_to", None::<&Buffer<f32>>)
        .arg_named("layer_size", 0_u32)
        .arg_named("initial_refract_interval", 0_u8)
        .arg_named("threshold", 0.0_f32)
        .arg_named("gamma", 0.0_f32)
        .arg_named("g_0", 0.0_f32)
        .build()?;

    Ok(CompiledKernel {
        kernel: Arc::new(Mutex::new(kernel)),
        pro_que,
    })
}

/**
 * Recount receiver layer
 */
pub fn apply_synapses(
    compiled_kernel: &CompiledKernel,
    layer_size: usize,
    accumulated_weights: &Array2<f32>,
    distance_weights: &Array2<f32>,
    neurons_from: &Array1<f32>,
    refract_intervals_to: &Array1<u8>,
    initial_refract_interval: u8,
    threshold: f32,
    gamma: f32,
    g_0: f32,
) -> ocl::Result<Array1<f32>> {
    let buffer_accumulated_weights = Buffer::<f32>::builder()
        .queue(compiled_kernel.pro_que.queue().clone())
        .len(accumulated_weights.len())
        .copy_host_slice(accumulated_weights.as_slice().unwrap())
        .build()?;

    let buffer_distance_weights = Buffer::<f32>::builder()
        .queue(compiled_kernel.pro_que.queue().clone())
        .len(distance_weights.len())
        .copy_host_slice(distance_weights.as_slice().unwrap())
        .build()?;

    let buffer_neurons_from = Buffer::<f32>::builder()
        .queue(compiled_kernel.pro_que.queue().clone())
        .len(neurons_from.len())
        .copy_host_slice(neurons_from.as_slice().unwrap())
        .build()?;

    let buffer_refract_intervals_to = Buffer::<u8>::builder()
        .queue(compiled_kernel.pro_que.queue().clone())
        .len(refract_intervals_to.len())
        .copy_host_slice(refract_intervals_to.as_slice().unwrap())
        .build()?;

    let buffer_next_neurons_to = Buffer::<f32>::builder()
        .queue(compiled_kernel.pro_que.queue().clone())
        .len(layer_size)
        .build()?;

    let kernel = compiled_kernel.kernel.lock().unwrap();

    unsafe {
        kernel.set_arg("accumulated_weights", &buffer_accumulated_weights)?;
        kernel.set_arg("distance_weights", &buffer_distance_weights)?;
        kernel.set_arg("neurons_from", &buffer_neurons_from)?;
        kernel.set_arg("refract_intervals_to", &buffer_refract_intervals_to)?;
        kernel.set_arg("next_neurons_to", &buffer_next_neurons_to)?;
        kernel.set_arg("layer_size", layer_size as u32)?;
        kernel.set_arg("initial_refract_interval", initial_refract_interval)?;
        kernel.set_arg("threshold", threshold)?;
        kernel.set_arg("gamma", gamma)?;
        kernel.set_arg("g_0", g_0)?;
        kernel.enq()?;
    }

    let mut vec_next_neurons_to = vec![0.0f32; layer_size];
    buffer_next_neurons_to
        .read(&mut vec_next_neurons_to)
        .enq()?;

    let next_neurons_to = Array1::from_vec(vec_next_neurons_to);

    Ok(next_neurons_to)
}
