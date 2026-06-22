use processor_cpu_full::ProcessorCpuFullRefractRecounter;

pub struct RefractRecounterCpu {
    max_refract_interval: u8,
}

impl RefractRecounterCpu {
    pub fn new(max_refract_interval: u8) -> Self {
        Self {
            max_refract_interval,
        }
    }
}

impl ProcessorCpuFullRefractRecounter for RefractRecounterCpu {
    fn recount(&self, neurons: &[bool], refract_intervals: &mut [u8]) {
        for (index, neuron) in neurons.iter().enumerate() {
            if *neuron {
                refract_intervals[index] = self.max_refract_interval.clone();
                continue;
            }

            if refract_intervals[index] > 0 {
                refract_intervals[index] -= 1;
            }
        }
    }
}
