use ndarray::{Array1, Array2};
use ocl::{ProQue, Buffer, Kernel};

pub struct CompiledKernel {
    kernel: Kernel,
    pro_que: ProQue
}

pub fn build_kernel(layer_size: usize) -> ocl::Result<CompiledKernel> {
    let kernel_source = include_str!("apply_synapses.cl");

    // Создаем контекст и очередь
    let pro_que = ProQue::builder()
        .src(kernel_source)
        .build()?;

    let kernel = Kernel::builder()
        .program(&pro_que.program())
        .name("apply_synapses")
        .queue(pro_que.queue().clone())
        .global_work_size(layer_size)
        .arg_named("synapses", None::<&Buffer<f32>>)
        .arg_named("neurons", None::<&Buffer<f32>>)
        .arg_named("refract_intervals", None::<&Buffer<u8>>)
        .arg_named("result", None::<&Buffer<f32>>)
        .arg_named("layer_size", 0 as u32)
        .arg_named("initial_refract_interval", 0 as u8)
        .arg_named("threshold", 0.0 as f32)
        .build()?;

    return Ok(CompiledKernel {
        kernel,
        pro_que
    });
}

pub fn apply_synapses(
    compiled_kernel: &CompiledKernel,
    layer_size: usize,
    synapses: &Array2<f32>,
    neurons: &Array1<f32>,
    refract_intervals: &mut Array1<u8>,
    initial_refract_interval: u8,
    threshold: f32,
) -> ocl::Result<Array1<f32>> {
    // Создаем буферы
    let buffer_synapses = Buffer::<f32>::builder()
        .queue(compiled_kernel.pro_que.queue().clone())
        .len(synapses.len())
        .copy_host_slice(synapses.as_slice().unwrap())
        .build()?;

    let buffer_neurons = Buffer::<f32>::builder()
        .queue(compiled_kernel.pro_que.queue().clone())
        .len(neurons.len())
        .copy_host_slice(neurons.as_slice().unwrap())
        .build()?;

    let buffer_refract_intervals = Buffer::<u8>::builder()
        .queue(compiled_kernel.pro_que.queue().clone())
        .flags(ocl::flags::MEM_READ_WRITE)
        .len(refract_intervals.len())
        .copy_host_slice(refract_intervals.as_slice().unwrap())
        .build()?;

    let buffer_result = Buffer::<f32>::builder()
        .queue(compiled_kernel.pro_que.queue().clone())
        .len(layer_size)
        .build()?;

    unsafe {
        compiled_kernel.kernel.set_arg("synapses", &buffer_synapses)?;
        compiled_kernel.kernel.set_arg("neurons", &buffer_neurons)?;
        compiled_kernel.kernel.set_arg("refract_intervals", &buffer_refract_intervals)?;
        compiled_kernel.kernel.set_arg("result", &buffer_result)?;
        compiled_kernel.kernel.set_arg("layer_size", layer_size as u32)?;
        compiled_kernel.kernel.set_arg("initial_refract_interval", initial_refract_interval)?;
        compiled_kernel.kernel.set_arg("threshold", threshold)?;
        compiled_kernel.kernel.enq()?;
    }

    // Считываем результаты
    let mut vec_result = vec![0.0f32; layer_size];
    buffer_result.read(&mut vec_result).enq()?;

    buffer_refract_intervals.read(refract_intervals.as_slice_mut().unwrap()).enq().unwrap();

    let result = Array1::from_vec(vec_result);

    return Ok(result);
}
