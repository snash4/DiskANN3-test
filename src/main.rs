//! Minimal DiskANN3 example: load a *disk* index from disk and run a KNN search.
//!
//! A DiskANN3 disk index is a set of files that share a common path prefix:
//!     <prefix>_disk.index          (the graph + full-precision vectors)
//!     <prefix>_pq_pivots.bin       (PQ pivot / centroid table)
//!     <prefix>_pq_compressed.bin   (PQ-compressed vectors held in memory)
//!
//! The path helpers in `diskann_providers::storage` turn the prefix into those
//! three concrete file names, so you only ever pass the prefix around.
//!
//! Usage:
//!     diskann3-load-index <index_prefix> <queries.fbin> [k] [L] [beam] [metric]
//!
//!     index_prefix   path prefix of the index (no _disk.index suffix)
//!     queries.fbin   DiskANN .fbin file: [u32 num][u32 dim][f32 num*dim]
//!     k              neighbors to return            (default 10)
//!     L              search-list size, must be >= k (default 100)
//!     beam           beam width                     (default 4)
//!     metric         l2 | cosine | mips | cosine_norm (default l2)

use std::env;
use std::fs::File;
use std::io::BufReader;

use anyhow::{bail, Context, Result};

use diskann_disk::{
    data_model::{AdHoc, CachingStrategy},
    search::provider::{
        disk_provider::DiskIndexSearcher,
        disk_vertex_provider_factory::DiskVertexProviderFactory,
    },
    storage::disk_index_reader::DiskIndexReader,
    utils::AlignedFileReaderFactory,
};
use diskann_providers::storage::{
    get_compressed_pq_file, get_disk_index_file, get_pq_pivot_file, FileStorageProvider,
};
use diskann_utils::{io::read_bin, views::Matrix};
use diskann_vector::distance::Metric;

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "usage: {} <index_prefix> <queries.fbin> [k] [L] [beam] [metric]",
            args.first().map(String::as_str).unwrap_or("diskann3-load-index")
        );
        std::process::exit(2);
    }

    let index_prefix = &args[1];
    let queries_path = &args[2];
    let k: u32 = parse_arg(&args, 3, 10)?;
    let l: u32 = parse_arg(&args, 4, 100)?;
    let beam: usize = parse_arg(&args, 5, 4)?;
    let metric = parse_metric(args.get(6).map(String::as_str).unwrap_or("l2"))?;

    if l < k {
        bail!("L ({l}) must be >= k ({k})");
    }

    // --- 1. Resolve the three index files from the prefix ---------------------
    let pivot_path = get_pq_pivot_file(index_prefix);
    let pq_data_path = get_compressed_pq_file(index_prefix);
    let disk_index_path = get_disk_index_file(index_prefix);
    println!("Loading disk index:");
    println!("  graph:        {disk_index_path}");
    println!("  pq pivots:    {pivot_path}");
    println!("  pq compressed:{pq_data_path}");

    // FileStorageProvider reads index files straight off the local filesystem.
    let storage = FileStorageProvider;

    // --- 2. Load PQ data (pivots + compressed vectors) into memory ------------
    // This also auto-detects the number of points from the compressed-PQ header.
    let index_reader = DiskIndexReader::<f32>::new(pivot_path, pq_data_path, &storage)
        .context("failed to load PQ pivot / compressed data")?;
    println!("Loaded PQ data: {} points", index_reader.get_num_points());

    // --- 3. Wire up the on-disk graph reader ----------------------------------
    // CachingStrategy::StaticCacheWithBfsNodes(n) pre-loads the n nodes nearest
    // the entry point into RAM; None keeps everything on disk.
    let caching_strategy = CachingStrategy::None;
    let reader_factory = AlignedFileReaderFactory::new(disk_index_path);
    let vertex_provider_factory = DiskVertexProviderFactory::new(reader_factory, caching_strategy)
        .context("failed to create disk vertex provider factory")?;

    // --- 4. Build the searcher ------------------------------------------------
    // AdHoc<f32> = "f32 vectors, u32 ids, no associated payload per node".
    // search_io_limit caps IOs per query; usize::MAX = unbounded.
    let num_threads = 1;
    let search_io_limit = usize::MAX;
    let searcher = DiskIndexSearcher::<AdHoc<f32>, _>::new(
        num_threads,
        search_io_limit,
        &index_reader,
        vertex_provider_factory,
        metric,
        None, // let it build its own current-thread Tokio runtime
    )
    .context("failed to construct DiskIndexSearcher")?;
    println!("Index loaded.\n");

    // --- 5. Load queries and search the first one -----------------------------
    // DiskANN3's own .fbin/.bin reader: 8-byte header (u32 npoints, u32 ndims)
    // followed by row-major f32 payload. Returns a Matrix<f32>.
    let mut reader = BufReader::new(File::open(queries_path).with_context(|| format!("opening {queries_path}"))?);
    let queries: Matrix<f32> =
        read_bin::<f32>(&mut reader).with_context(|| format!("reading fbin {queries_path}"))?;
    if queries.nrows() == 0 {
        bail!("query file contained no vectors");
    }
    println!("Loaded {} query vector(s), dim = {}", queries.nrows(), queries.ncols());

    let query: &[f32] = queries.row(0);
    let result = searcher
        .search(
            query, // &[f32]
            k,     // neighbors to return
            l,     // search-list size
            Some(beam),
            None,  // no per-vector filter
            false, // is_flat_search: false = graph search (not brute force)
        )
        .context("search failed")?;

    println!("\nTop {k} neighbors for query 0:");
    for (rank, item) in result.results.iter().enumerate() {
        println!("  {:>2}. id = {:<10} distance = {:.6}", rank + 1, item.vertex_id, item.distance);
    }
    println!(
        "\nstats: comparisons = {}, results = {}",
        result.stats.cmps, result.stats.result_count
    );

    Ok(())
}

fn parse_arg<T: std::str::FromStr>(args: &[String], idx: usize, default: T) -> Result<T>
where
    T::Err: std::fmt::Display,
{
    match args.get(idx) {
        Some(s) => s
            .parse::<T>()
            .map_err(|e| anyhow::anyhow!("invalid value '{s}' for arg {idx}: {e}")),
        None => Ok(default),
    }
}

fn parse_metric(s: &str) -> Result<Metric> {
    Ok(match s.to_ascii_lowercase().as_str() {
        "l2" | "euclidean" => Metric::L2,
        "cosine" => Metric::Cosine,
        "cosine_norm" | "cosine_normalized" => Metric::CosineNormalized,
        "mips" | "ip" | "inner_product" => Metric::InnerProduct,
        other => bail!("unknown metric '{other}' (use l2|cosine|mips|cosine_norm)"),
    })
}