use std::collections::HashMap;

use rnn_architecture::DistanceBetweenNeurons;

pub struct PowerLawDistance {
    alpha: f32,
    h: f32,
    cache: HashMap<(i32, i32), f32>,
}

fn get_cache_key(dx: i32, dy: i32) -> (i32, i32) {
    let dx_abs = dx.abs();
    let dy_abs = dy.abs();

    if dx_abs < dy_abs {
        (dx_abs, dy_abs)
    } else {
        (dy_abs, dx_abs)
    }
}

impl PowerLawDistance {
    pub fn new(alpha: f32, h: f32) -> Self {
        Self {
            alpha,
            h,
            cache: HashMap::new(),
        }
    }
}

impl DistanceBetweenNeurons for PowerLawDistance {
    fn get_distance(&mut self, dx: i32, dy: i32) -> f32 {
        let cache_key = get_cache_key(dx, dy);

        if let Some(cached_vale) = self.cache.get(&cache_key) {
            return *cached_vale;
        }

        let distance = ((dx * dx + dy * dy) as f32).sqrt();

        let value = 1.0 / (1.0 + self.alpha * distance.powf(self.h));

        self.cache.insert(cache_key, value);

        value
    }
}
