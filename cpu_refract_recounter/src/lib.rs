pub trait CpuRefractRecounter {
    fn recount<'a>(
        &self,
        neurons: Box<dyn Iterator<Item = bool> + 'a>,
        refract_intervals: &mut [u8],
    );
}
