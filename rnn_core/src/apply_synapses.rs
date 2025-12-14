use std::sync::{Arc, Mutex};

use ocl::{Buffer, Kernel, ProQue};

use crate::excite_neurons_with_partitions::excite_neurons_with_partitions;
use crate::excite_neurons_without_partitions::excite_neurons_without_partitions;
use crate::logger::{Logger, LoggerEvent};
use crate::structures::{CompiledKernel, ComputedParams};
use crate::{LayerParams, SynapseParams};

pub fn build_apply_synapses_kernel(layer_size: usize) -> ocl::Result<CompiledKernel> {
    let kernel_source = include_str!("apply_synapses.cl");

    let pro_que = ProQue::builder().src(kernel_source).build()?;

    let kernel = Kernel::builder()
        .program(pro_que.program())
        .name("apply_synapses")
        .queue(pro_que.queue().clone())
        .global_work_size(layer_size)
        .arg_named("forward_synapses", None::<&Buffer<f32>>)
        .arg_named("distance_weights", None::<&Buffer<f32>>)
        .arg_named("offsets", None::<&Buffer<u64>>)
        .arg_named("restore_offsets", None::<&Buffer<u64>>)
        .arg_named("restore_offsets_count", 0_u32)
        .arg_named("restore_synapses", None::<&Buffer<f32>>)
        .arg_named("neurons_from", None::<&Buffer<u8>>)
        .arg_named("refract_intervals_to", None::<&Buffer<u8>>)
        .arg_named("neurons_to", None::<&Buffer<u8>>)
        .arg_named("signals_to", None::<&Buffer<f32>>)
        .arg_named("layer_size", 0_u32)
        .arg_named("field_size", 0_u32)
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
    buffer_forward_synapses: &Buffer<f32>,
    buffer_distance_weights: &Buffer<f32>,
    buffer_offsets: &Buffer<u64>,
    buffer_restore_offsets: Option<&Buffer<u64>>,
    buffer_restore_synapses: Option<&Buffer<f32>>,
    restore_offsets_count: &usize,
    neurons_from: &Vec<u8>,
    neurons_to: &mut Vec<u8>,
    refract_intervals_to: &Vec<u8>,
    layer_params: &LayerParams,
    synapse_params: &SynapseParams,
    computed_params: &ComputedParams,
    threshold: f32,
    layer_index: usize,
    logger: &mut Option<Box<dyn Logger>>,
) -> ocl::Result<()> {
    let LayerParams { partitions, .. } = layer_params;

    let SynapseParams {
        gamma_inc,
        gamma_dec,
        g_0,
        g_dec,
        g_inc,
        min_g,
        max_g,
        ..
    } = synapse_params;

    let ComputedParams {
        field_size,
        field_count,
        excited_neurons_limit,
        ..
    } = computed_params;

    let buffer_neurons_from = Buffer::<u8>::builder()
        .queue(compiled_kernel.pro_que.queue().clone())
        .len(neurons_from.len())
        .copy_host_slice(neurons_from.as_slice())
        .build()?;

    let buffer_refract_intervals_to = Buffer::<u8>::builder()
        .queue(compiled_kernel.pro_que.queue().clone())
        .len(refract_intervals_to.len())
        .copy_host_slice(refract_intervals_to.as_slice())
        .build()?;

    let buffer_neurons_to = Buffer::<u8>::builder()
        .queue(compiled_kernel.pro_que.queue().clone())
        .flags(ocl::flags::MEM_READ_WRITE)
        .len(layer_size)
        .build()?;

    let buffer_signals_to = Buffer::<f32>::builder()
        .queue(compiled_kernel.pro_que.queue().clone())
        .flags(ocl::flags::MEM_READ_WRITE)
        .len(if is_prediction { layer_size } else { 1 })
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
        kernel.set_arg("forward_synapses", buffer_forward_synapses)?;
        kernel.set_arg("distance_weights", buffer_distance_weights)?;
        kernel.set_arg("offsets", buffer_offsets)?;
        kernel.set_arg("restore_offsets", buffer_restore_offsets)?;
        kernel.set_arg("restore_offsets_count", *restore_offsets_count as u32)?;
        kernel.set_arg("restore_synapses", buffer_restore_synapses)?;
        kernel.set_arg("neurons_from", &buffer_neurons_from)?;
        kernel.set_arg("refract_intervals_to", &buffer_refract_intervals_to)?;
        kernel.set_arg("neurons_to", &buffer_neurons_to)?;
        kernel.set_arg("signals_to", &buffer_signals_to)?;
        kernel.set_arg("layer_size", layer_size as u32)?;
        kernel.set_arg("field_size", *field_size as u32)?;
        kernel.set_arg("threshold", threshold)?;
        kernel.set_arg("gamma_inc", gamma_inc)?;
        kernel.set_arg("gamma_dec", gamma_dec)?;
        kernel.set_arg("g_0", if layer_index == 2 { g_0 } else { &0.0 })?;
        kernel.set_arg("g_dec", g_dec)?;
        kernel.set_arg("g_inc", g_inc)?;
        kernel.set_arg("min_g", min_g)?;
        kernel.set_arg("max_g", max_g)?;
        kernel.set_arg("is_prediction", if is_prediction { 1_u8 } else { 0_u8 })?;
        kernel.set_arg("inc_counter", &buffer_inc_counter)?;
        kernel.set_arg("dec_counter", &buffer_dec_counter)?;
        kernel.enq()?;
    }
    if is_prediction {
        let mut signals_to = vec![0.0_f32; layer_size];

        buffer_signals_to.read(&mut signals_to).enq()?;

        if let Some(partitions) = partitions {
            excite_neurons_with_partitions(
                neurons_to,
                &signals_to,
                *field_size,
                *field_count,
                threshold,
                partitions,
            );
        } else {
            excite_neurons_without_partitions(neurons_to, &signals_to, threshold);

            remove_extra_neurons(neurons_to, *excited_neurons_limit);
        }
    } else {
        let mut neurons_to_flat = neurons_to.as_slice().to_vec();

        buffer_neurons_to.read(&mut neurons_to_flat).enq()?;

        neurons_to.copy_from_slice(&neurons_to_flat);
    }

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
