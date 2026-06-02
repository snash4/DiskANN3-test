/*
 * `inputs`: a faithful re-creation of the config types from
 * `diskann-benchmark/src/inputs/disk.rs` (and the `SimilarityMeasure` /
 * `DataType` they reference). These are `pub(crate)` in the upstream crate, so
 * we redefine them here with the same names, fields and serde shape.
 */

use anyhow::bail;
use diskann_providers::storage::{
    get_compressed_pq_file, get_disk_index_file, get_pq_pivot_file, StorageReadProvider,
};
use diskann_vector::distance::Metric;
use serde::{Deserialize, Serialize};

/// Mirrors `diskann_benchmark::utils::SimilarityMeasure` + its `Into<Metric>`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum SimilarityMeasure {
    SquaredL2,
    InnerProduct,
    Cosine,
    CosineNormalized,
}

impl From<SimilarityMeasure> for Metric {
    fn from(value: SimilarityMeasure) -> Self {
        match value {
            SimilarityMeasure::SquaredL2 => Metric::L2,
            SimilarityMeasure::InnerProduct => Metric::InnerProduct,
            SimilarityMeasure::Cosine => Metric::Cosine,
            SimilarityMeasure::CosineNormalized => Metric::CosineNormalized,
        }
    }
}

/// Mirrors `diskann_benchmark_runner::utils::datatype::DataType` (subset used here).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DataType {
    Float32,
    Float16,
    UInt8,
    Int8,
}

/// Mirrors `DiskIndexOperation { source, search_phase }`.
#[derive(Debug, Serialize, Deserialize)]
pub struct DiskIndexOperation {
    pub source: DiskIndexSource,
    pub search_phase: DiskSearchPhase,
}

/// Mirrors `DiskIndexSource` (a tagged enum of "load an existing index" vs
/// "build then search"). We implement the `Load` path; `Build` is kept for
/// fidelity but the driver reports it as out of scope for this example.
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "disk-index-source")]
pub enum DiskIndexSource {
    Load(DiskIndexLoad),
    Build(DiskIndexBuild),
}

/// Mirrors `DiskIndexLoad { data_type, load_path }`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiskIndexLoad {
    pub data_type: DataType,
    /// Path *prefix* of the index (no `_disk.index` suffix).
    pub load_path: String,
}

/// Mirrors `DiskIndexBuild` (kept minimal — building is out of scope here).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiskIndexBuild {
    pub data_type: DataType,
    pub data: String,
    pub distance: SimilarityMeasure,
    pub dim: usize,
    pub max_degree: usize,
    pub l_build: usize,
    pub save_path: String,
}

/// Mirrors `DiskSearchPhase`. `groundtruth` is optional here so the example can
/// run without a truthset; if present, recall@k is computed.
#[derive(Debug, Serialize, Deserialize)]
pub struct DiskSearchPhase {
    pub queries: String,
    pub groundtruth: Option<String>,
    pub num_threads: usize,
    pub beam_width: usize,
    pub search_list: Vec<u32>,
    pub recall_at: u32,
    pub is_flat_search: bool,
    pub distance: SimilarityMeasure,
    pub vector_filters_file: Option<String>,
    pub num_nodes_to_cache: Option<usize>,
    pub search_io_limit: Option<usize>,
}

impl DiskIndexOperation {
    /// Mirrors the upstream `validate(...)`: dispatch on the source, then the
    /// search phase. Uses the storage provider's `exists` to check files.
    pub fn validate(&self, storage: &impl StorageReadProvider) -> anyhow::Result<()> {
        match &self.source {
            DiskIndexSource::Load(load) => load.validate(storage)?,
            DiskIndexSource::Build(_) => {
                bail!("`Build` source is not implemented in this example; use `Load`")
            }
        }
        self.search_phase.validate(storage)
    }
}

impl DiskIndexLoad {
    /// Mirrors upstream `DiskIndexLoad::validate`: the three index files derived
    /// from the prefix must all exist.
    pub fn validate(&self, storage: &impl StorageReadProvider) -> anyhow::Result<()> {
        let files = [
            (get_pq_pivot_file(&self.load_path), "pq pivot file"),
            (get_compressed_pq_file(&self.load_path), "compressed pq file"),
            (get_disk_index_file(&self.load_path), "disk index file"),
        ];
        for (path, label) in files {
            if !storage.exists(&path) {
                bail!("{label} {path} does not exist");
            }
        }
        if self.data_type != DataType::Float32 {
            bail!("this example only implements DataType::Float32 (got {:?})", self.data_type);
        }
        Ok(())
    }
}

impl DiskSearchPhase {
    pub fn validate(&self, storage: &impl StorageReadProvider) -> anyhow::Result<()> {
        if !storage.exists(&self.queries) {
            bail!("queries file {} does not exist", self.queries);
        }
        if let Some(gt) = &self.groundtruth {
            if !storage.exists(gt) {
                bail!("groundtruth file {gt} does not exist");
            }
        }
        if self.search_list.is_empty() {
            bail!("search_list must have at least one value");
        }
        if self.search_list.iter().any(|&l| l == 0 || l < self.recall_at) {
            bail!("every search_list value must be > 0 and >= recall_at");
        }
        if self.beam_width == 0 {
            bail!("beam_width must be positive");
        }
        if self.recall_at == 0 {
            bail!("recall_at must be positive");
        }
        if self.num_threads == 0 {
            bail!("num_threads must be positive");
        }
        if self.vector_filters_file.is_some() {
            bail!("vector filters are not implemented in this example");
        }
        Ok(())
    }
}

/// Mirrors the upstream `Example` impl so users can see the JSON shape.
pub fn example() -> DiskIndexOperation {
    DiskIndexOperation {
        source: DiskIndexSource::Load(DiskIndexLoad {
            data_type: DataType::Float32,
            load_path: "sample_index_l50_r32".to_string(),
        }),
        search_phase: DiskSearchPhase {
            queries: "queries.fbin".to_string(),
            groundtruth: Some("groundtruth.ibin".to_string()),
            num_threads: 8,
            beam_width: 4,
            search_list: vec![64, 128, 256],
            recall_at: 10,
            is_flat_search: false,
            distance: SimilarityMeasure::SquaredL2,
            vector_filters_file: None,
            num_nodes_to_cache: None,
            search_io_limit: None,
        },
    }
}