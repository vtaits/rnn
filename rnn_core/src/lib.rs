mod apply_synapses;
mod data_layer;
mod get_synapse_mask;
mod network;
mod recount_accumulated_weights;
mod recount_refract_intervals;
mod spiral;
mod structures;

pub use data_layer::{DataLayer, DataLayerParams};
pub use network::{Network, NetworkParseError};
pub use structures::{LayerParams, SynapseParams};
