use std::sync::{Arc, Mutex};

use ndarray::{Array1, Array2};
use ocl::{Buffer, Kernel, ProQue};

use crate::logger::{Logger, LoggerEvent};
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
        .arg_named("strong_synapses", None::<&Buffer<u64>>)
        .arg_named("distance_weights", None::<&Buffer<f32>>)
        .arg_named("neurons_from", None::<&Buffer<u8>>)
        .arg_named("refract_intervals_to", None::<&Buffer<u8>>)
        .arg_named("next_neurons_to", None::<&Buffer<u8>>)
        .arg_named("layer_size", 0_u32)
        .arg_named("initial_refract_interval", 0_u8)
        .arg_named("threshold", 0.0_f32)
        .arg_named("gamma_inc", 0.0_f32)
        .arg_named("gamma_dec", 0.0_f32)
        .arg_named("g_0", 0.0_f32)
        .arg_named("g_dec", 0.0_f32)
        .arg_named("g_inc", 0.0_f32)
        .arg_named("min_g", 0.0_f32)
        .arg_named("max_g", 0.0_f32)
        .arg_named("is_prediction", 0_u8)
        .arg_named("inc_counter", None::<&Buffer<i32>>)
        .arg_named("dec_counter", None::<&Buffer<i32>>)
        .build()?;

    Ok(CompiledKernel {
        kernel: Arc::new(Mutex::new(kernel)),
        pro_que,
    })
}

fn remove_extra_neurons(neurons: &mut Vec<u8>, limit: usize) {
    let excited_count = neurons.iter().filter(|&&x| x != 0).count();

    if excited_count <= limit {
        return;
    }

    let mut neurons_to_remove = excited_count - limit;

    while neurons_to_remove > 0 {
        let index = rand::random_range(0..neurons.iter().count());

        if neurons[index] > 0 {
            neurons[index] = 0;
            neurons_to_remove -= 1;
        }
    }
}

/**
 * Recount receiver layer
 */
pub fn apply_synapses(
    compiled_kernel: &CompiledKernel,
    is_prediction: bool,
    layer_size: usize,
    accumulated_weights: &mut Array2<f32>,
    strong_synapses: &Array1<u64>,
    distance_weights: &Array2<f32>,
    neurons_from: &Array1<u8>,
    neurons_to: &mut Array1<u8>,
    refract_intervals_to: &Array1<u8>,
    initial_refract_interval: u8,
    threshold: f32,
    gamma_inc: f32,
    gamma_dec: f32,
    g_0: f32,
    excited_neurons_limit: usize,
    g_dec: f32,
    g_inc: f32,
    min_g: f32,
    max_g: f32,
    layer_index: usize,
    logger: &mut Option<Box<dyn Logger>>,
) -> ocl::Result<()> {
    let mut accumulated_weights_flat = accumulated_weights.as_slice().unwrap().to_vec();

    let buffer_accumulated_weights = Buffer::<f32>::builder()
        .queue(compiled_kernel.pro_que.queue().clone())
        .flags(ocl::flags::MEM_READ_WRITE)
        .len(accumulated_weights.len())
        .copy_host_slice(accumulated_weights.as_slice().unwrap())
        .build()?;

    let buffer_strong_synapses = Buffer::<u64>::builder()
        .queue(compiled_kernel.pro_que.queue().clone())
        .len(strong_synapses.len())
        .copy_host_slice(strong_synapses.as_slice().unwrap())
        .build()?;

    let buffer_distance_weights = Buffer::<f32>::builder()
        .queue(compiled_kernel.pro_que.queue().clone())
        .len(distance_weights.len())
        .copy_host_slice(distance_weights.as_slice().unwrap())
        .build()?;

    let buffer_neurons_from = Buffer::<u8>::builder()
        .queue(compiled_kernel.pro_que.queue().clone())
        .len(neurons_from.len())
        .copy_host_slice(neurons_from.as_slice().unwrap())
        .build()?;

    let buffer_refract_intervals_to = Buffer::<u8>::builder()
        .queue(compiled_kernel.pro_que.queue().clone())
        .len(refract_intervals_to.len())
        .copy_host_slice(refract_intervals_to.as_slice().unwrap())
        .build()?;

    let buffer_next_neurons_to = Buffer::<u8>::builder()
        .queue(compiled_kernel.pro_que.queue().clone())
        .flags(ocl::flags::MEM_READ_WRITE)
        .len(layer_size)
        .build()?;

    let buffer_inc_counter = Buffer::<i32>::builder()
        .queue(compiled_kernel.pro_que.queue().clone())
        .flags(ocl::flags::MEM_READ_WRITE)
        .len(1)
        .fill_val(0)
        .build()
        .unwrap();

    let buffer_dec_counter = Buffer::<i32>::builder()
        .queue(compiled_kernel.pro_que.queue().clone())
        .flags(ocl::flags::MEM_READ_WRITE)
        .len(1)
        .fill_val(0)
        .build()
        .unwrap();

    let kernel = compiled_kernel.kernel.lock().unwrap();

    unsafe {
        kernel.set_arg("accumulated_weights", &buffer_accumulated_weights)?;
        kernel.set_arg("strong_synapses", &buffer_strong_synapses)?;
        kernel.set_arg("distance_weights", &buffer_distance_weights)?;
        kernel.set_arg("neurons_from", &buffer_neurons_from)?;
        kernel.set_arg("refract_intervals_to", &buffer_refract_intervals_to)?;
        kernel.set_arg("next_neurons_to", &buffer_next_neurons_to)?;
        kernel.set_arg("layer_size", layer_size as u32)?;
        kernel.set_arg("initial_refract_interval", initial_refract_interval)?;
        kernel.set_arg("threshold", threshold)?;
        kernel.set_arg("gamma_inc", gamma_inc)?;
        kernel.set_arg("gamma_dec", gamma_dec)?;
        kernel.set_arg("g_0", g_0)?;
        kernel.set_arg("g_dec", g_dec)?;
        kernel.set_arg("g_inc", g_inc)?;
        kernel.set_arg("min_g", min_g)?;
        kernel.set_arg("max_g", max_g)?;
        kernel.set_arg("is_prediction", if is_prediction { 1_u8 } else { 0_u8 })?;
        kernel.set_arg("inc_counter", &buffer_inc_counter)?;
        kernel.set_arg("dec_counter", &buffer_dec_counter)?;
        kernel.enq()?;
    }

    let mut neurons_to_flat = neurons_to.as_slice().unwrap().to_vec();

    buffer_next_neurons_to.read(&mut neurons_to_flat).enq()?;

    if is_prediction {
        remove_extra_neurons(&mut neurons_to_flat, excited_neurons_limit);
    }

    neurons_to
        .as_slice_mut()
        .unwrap()
        .copy_from_slice(&neurons_to_flat);

    buffer_accumulated_weights
        .read(&mut accumulated_weights_flat)
        .enq()?;

    accumulated_weights
        .as_slice_mut()
        .unwrap()
        .copy_from_slice(&accumulated_weights_flat);

    let mut inc_counter_result = vec![0i32; 1];
    buffer_inc_counter
        .read(&mut inc_counter_result)
        .enq()
        .unwrap();

    let mut dec_counter_result = vec![0i32; 1];
    buffer_dec_counter
        .read(&mut dec_counter_result)
        .enq()
        .unwrap();

    if let Some(logger) = logger {
        logger.log_event(LoggerEvent::ChangeLayerWeights(
            layer_index,
            inc_counter_result[0] as u8,
            dec_counter_result[0] as u8,
        ));
    }

    Ok(())
}
