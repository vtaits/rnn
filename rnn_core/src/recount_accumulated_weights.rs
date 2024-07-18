use ndarray::{Array1, Array2};
use ocl::{Buffer, Kernel, ProQue};

use crate::structures::CompiledKernel;

pub fn build_recount_accumulated_weights_kernel(layer_size: usize) -> ocl::Result<CompiledKernel> {
    let kernel_source = include_str!("recount_accumulated_weights.cl");

    let pro_que = ProQue::builder().src(kernel_source).build()?;

    let kernel = Kernel::builder()
        .program(pro_que.program())
        .name("recount_accumulated_weights")
        .queue(pro_que.queue().clone())
        .global_work_size(layer_size)
        .arg_named("accumulated_weights", None::<&Buffer<f32>>)
        .arg_named("neurons_from", None::<&Buffer<f32>>)
        .arg_named("neurons_to", None::<&Buffer<f32>>)
        .arg_named("refract_intervals_to", None::<&Buffer<u8>>)
        .arg_named("next_accumulated_weights", None::<&Buffer<f32>>)
        .arg_named("layer_size", 0_u32)
        .arg_named("g_dec", 0.0_f32)
        .arg_named("g_inc", 0.0_f32)
        .build()?;

    Ok(CompiledKernel { kernel, pro_que })
}

pub fn recount_accumulated_weights(
    compiled_kernel: &CompiledKernel,
    layer_size: usize,
    accumulated_weights: &Array2<f32>,
    neurons_from: &Array1<f32>,
    neurons_to: &Array1<f32>,
    refract_intervals_to: &Array1<u8>,
    g_dec: f32,
    g_inc: f32,
) -> ocl::Result<Array2<f32>> {
    let buffer_accumulated_weights = Buffer::<f32>::builder()
        .queue(compiled_kernel.pro_que.queue().clone())
        .len(accumulated_weights.len())
        .copy_host_slice(accumulated_weights.as_slice().unwrap())
        .build()?;

    let buffer_neurons_from = Buffer::<f32>::builder()
        .queue(compiled_kernel.pro_que.queue().clone())
        .len(neurons_from.len())
        .copy_host_slice(neurons_from.as_slice().unwrap())
        .build()?;

    let buffer_neurons_to = Buffer::<f32>::builder()
        .queue(compiled_kernel.pro_que.queue().clone())
        .len(neurons_to.len())
        .copy_host_slice(neurons_to.as_slice().unwrap())
        .build()?;

    let buffer_refract_intervals_to = Buffer::<u8>::builder()
        .queue(compiled_kernel.pro_que.queue().clone())
        .len(refract_intervals_to.len())
        .copy_host_slice(refract_intervals_to.as_slice().unwrap())
        .build()?;

    let buffer_next_accumulated_weights = Buffer::<f32>::builder()
        .queue(compiled_kernel.pro_que.queue().clone())
        .len(accumulated_weights.len())
        .build()?;

    unsafe {
        compiled_kernel
            .kernel
            .set_arg("accumulated_weights", &buffer_accumulated_weights)?;
        compiled_kernel
            .kernel
            .set_arg("neurons_from", &buffer_neurons_from)?;
        compiled_kernel
            .kernel
            .set_arg("neurons_to", &buffer_neurons_to)?;
        compiled_kernel
            .kernel
            .set_arg("refract_intervals_to", &buffer_refract_intervals_to)?;
        compiled_kernel
            .kernel
            .set_arg("next_accumulated_weights", &buffer_next_accumulated_weights)?;
        compiled_kernel
            .kernel
            .set_arg("layer_size", layer_size as u32)?;
        compiled_kernel.kernel.set_arg("g_dec", g_dec)?;
        compiled_kernel.kernel.set_arg("g_inc", g_inc)?;
        compiled_kernel.kernel.enq()?;
    }

    let mut vec_next_accumulated_weights = vec![0.0f32; layer_size * layer_size];
    buffer_next_accumulated_weights
        .read(&mut vec_next_accumulated_weights)
        .enq()?;

    let next_accumulated_weights =
        Array2::from_shape_vec((layer_size, layer_size), vec_next_accumulated_weights).unwrap();

    Ok(next_accumulated_weights)
}
