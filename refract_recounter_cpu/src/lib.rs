use cpu_refract_recounter::CpuRefractRecounter;

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

impl CpuRefractRecounter for RefractRecounterCpu {
    fn recount<'a>(&self, neurons: Box<dyn Iterator<Item = bool> + 'a>, refract_intervals: &mut [u8]) {
        for (index, neuron) in neurons.enumerate() {
            if neuron {
                refract_intervals[index] = self.max_refract_interval.clone();
                continue;
            }

            if refract_intervals[index] > 0 {
                refract_intervals[index] -= 1;
            }
        }
    }
}
