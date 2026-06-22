pub trait ProcessorCpuFullRefractRecounter {
    fn recount(&self, neurons: &[bool], refract_intervals: &mut [u8]);
}
