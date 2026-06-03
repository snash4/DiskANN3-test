//! Benchmark-style DiskANN3 disk-index loader/searcher.
//!
//! Reproduces the structure of DiskANN3's own `diskann-benchmark` disk-index
//! job: a JSON spec deserializes into `DiskIndexOperation { source, search_phase }`,
//! is validated, then dispatched. The `Load` source loads an existing disk index
//! and runs the search phase over each `L` in `search_list`.
//!
//! Usage:
//!     diskann3-bench-style <job.json>                 # run at search_phase.num_threads
//!     diskann3-bench-style <job.json> --sweep 1,2,4,8 # sweep thread counts, report QPS
//!     diskann3-bench-style --print-example            # prints a sample job spec
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
        eprintln!("usage: {} <job.json> [--sweep 1,2,4,8] | --print-example", args[0]);
        std::process::exit(2);
    });

    // Optional thread sweep: --sweep 1,2,4,8
    let thread_sweep: Option<Vec<usize>> = args
        .iter()
        .position(|a| a == "--sweep")
        .and_then(|i| args.get(i + 1))
        .map(|s| parse_thread_list(s))
        .transpose()?;

    // Read + parse the JSON job spec (mirrors how diskann-benchmark loads jobs).
    let json = std::fs::read_to_string(&spec_path)
        .with_context(|| format!("reading job spec {spec_path}"))?;
    let op: DiskIndexOperation =
        serde_json::from_str(&json).with_context(|| format!("parsing job spec {spec_path}"))?;

    // FileStorageProvider = read index/query/gt files from the local filesystem.
    let storage = FileStorageProvider;

    // Validate, then dispatch on the source (Load implemented; Build rejected).
    op.validate(&storage)?;

    // Thread counts: the sweep list if given, else just the spec's num_threads.
    let thread_counts = thread_sweep.unwrap_or_else(|| vec![op.search_phase.num_threads]);

    match &op.source {
        DiskIndexSource::Load(load) => {
            println!("Disk Index Load: prefix = {}", load.load_path);
            if thread_counts.len() > 1 {
                println!("Thread sweep: {thread_counts:?}\n");
            } else {
                println!();
            }
            let stats = match load.data_type {
                DataType::Float32 => {
                    backend::run::<f32, _>(load, &op.search_phase, &storage, &thread_counts)?
                }
                other => bail!("this example only implements Float32 (got {other:?})"),
            };
            println!("\nResults:\n");
            print!("{stats}");
        }
        DiskIndexSource::Build(_) => bail!("`Build` source is out of scope for this example"),
    }

    Ok(())
}

/// Parse a comma-separated list of positive thread counts (e.g. "1,2,4,8").
fn parse_thread_list(s: &str) -> Result<Vec<usize>> {
    let mut counts = Vec::new();
    for part in s.split(',') {
        let n: usize = part
            .trim()
            .parse()
            .with_context(|| format!("invalid thread count '{part}' in --sweep"))?;
        if n == 0 {
            bail!("thread counts must be positive (got 0)");
        }
        if !counts.contains(&n) {
            counts.push(n);
        }
    }
    if counts.is_empty() {
        bail!("--sweep needs at least one thread count");
    }
    counts.sort_unstable();
    Ok(counts)
}