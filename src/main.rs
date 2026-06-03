//! Minimal DiskANN3 example: load a *disk* index from disk and run a KNN search.
//!
//! Reproduces the structure of DiskANN3's own `diskann-benchmark` disk-index
//! job: a JSON spec deserializes into `DiskIndexOperation { source, search_phase }`,
//! is validated, then dispatched. The `Load` source loads an existing disk index
//! and runs the search phase over each `L` in `search_list`.
//!
//! Usage:
//!     diskann3-bench-style <job.json>
//!     diskann3-bench-style --print-example      # prints a sample job spec
//!
//! Example job.json:
//! {
//!   "source":      { "disk-index-source": "Load",
//!                    "data_type": "Float32", "load_path": "sample_index_l50_r32" },
//!   "search_phase":{ "queries": "queries.fbin", "groundtruth": "groundtruth.ibin",
//!                    "num_threads": 8, "beam_width": 4, "search_list": [64,128,256],
//!                    "recall_at": 10, "is_flat_search": false, "distance": "SquaredL2",
//!                    "vector_filters_file": null, "num_nodes_to_cache": null,
//!                    "search_io_limit": null }
//! }

mod backend;
mod inputs;

use std::env;

use anyhow::{bail, Context, Result};
use diskann_providers::storage::FileStorageProvider;
use inputs::{DataType, DiskIndexOperation, DiskIndexSource};


fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
  
    if args.iter().any(|a| a == "--print-example") {
        println!("{}", serde_json::to_string_pretty(&inputs::example())?);
        return Ok(());
    }
    let spec_path = args.get(1).cloned().unwrap_or_else(|| {
        eprintln!("usage: {} <job.json> | --print-example", args[0]);
        std::process::exit(2);
    });
    let json = std::fs::read_to_string(&spec_path)
        .with_context(|| format!("reading job spec {spec_path}"))?;

    let op: DiskIndexOperation =
        serde_json::from_str(&json).with_context(|| format!("parsing job spec {spec_path}"))?;

    // FileStorageProvider = read index/query/gt files from the local filesystem.
    let storage = FileStorageProvider;

    // Validate, then dispatch on the source (Load implemented; Build rejected).
    op.validate(&storage)?;

match &op.source {
        DiskIndexSource::Load(load) => {
            println!("Disk Index Load: prefix = {}\n", load.load_path);
            let stats = match load.data_type {
                DataType::Float32 => {
                    backend::run::<f32, _>(load, &op.search_phase, &storage, &[op.search_phase.num_threads])?
                }
                other => bail!("this example only implements Float32 (got {other:?})"),
            };
            println!("\nSearch results (recall@{}, beam={}):", stats.recall_at, stats.beam_width);
            print!("{stats}");
        }
        DiskIndexSource::Build(_) => bail!("`Build` source is out of scope for this example"),
    }

    Ok(())
}