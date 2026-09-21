use cpu_refract_recounter::CpuRefractRecounter;

pub struct RefractRecounterSkip {}

impl RefractRecounterSkip {
    pub fn new() -> Self {
        Self {}
    }
}

impl CpuRefractRecounter for RefractRecounterSkip {
    fn recount<'a>(
        &self,
        _neurons: Box<dyn Iterator<Item = bool> + 'a>,
        _refract_intervals: &mut [u8],
    ) {
    }
}
