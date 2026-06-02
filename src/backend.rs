/*
 * `backend`: a trimmed re-creation of
 * `diskann-benchmark/src/backend/disk_index/search.rs::search_disk_index`,
 * extended with a thread-count sweep.
 *
 * Same flow as upstream: resolve the index files from the prefix, load PQ data,
 * wire the on-disk graph reader, build ONE `DiskIndexSearcher`, then run every
 * query in parallel across a rayon pool. Upstream shares a single searcher across
 * the pool (each query is one thread + concurrent IO via beam search), so we can
 * reuse that one searcher across pools of different sizes to measure how QPS
 * scales with threads and where the device saturates.
 */

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use rayon::prelude::*;

use diskann::utils::VectorRepr;
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
    get_compressed_pq_file, get_disk_index_file, get_pq_pivot_file, StorageReadProvider,
};
use diskann_utils::{io::read_bin, views::Matrix};

use crate::inputs::{DiskIndexLoad, DiskSearchPhase};

/// Result of one search pass (one thread count, one `L`).
#[derive(Debug, Clone, Copy)]
pub struct PassResult {
    pub qps: f32,
    pub mean_latency_us: f64,
    pub mean_comparisons: f64,
}

/// All `L` passes for a single thread count (`passes` is parallel to `search_list`).
#[derive(Debug)]
pub struct ThreadSweepRow {
    pub num_threads: usize,
    pub passes: Vec<PassResult>,
}

/// Full sweep result.
#[derive(Debug)]
pub struct SweepStats {
    pub recall_at: u32,
    pub beam_width: usize,
    pub search_list: Vec<u32>,
    pub recalls: Vec<Option<f32>>, // one per L (recall is thread-count independent)
    pub rows: Vec<ThreadSweepRow>,  // one per thread count
}

/// Load a disk index once and run the search phase at each thread count in
/// `thread_counts` (pass a single value for a normal, non-sweep run).
pub fn run<T, S>(
    index_load: &DiskIndexLoad,
    search: &DiskSearchPhase,
    storage: &S,
    thread_counts: &[usize],
) -> Result<SweepStats>
where
    T: VectorRepr + bytemuck::Pod,
    S: StorageReadProvider,
{
    if thread_counts.is_empty() {
        bail!("thread_counts must not be empty");
    }
    let max_threads = thread_counts.iter().copied().max().unwrap_or(1).max(1);

    // --- Load queries (DiskANN3's own .fbin reader) --------------------------
    let queries: Matrix<T> = read_bin::<T>(
        &mut storage
            .open_reader(&search.queries)
            .with_context(|| format!("opening queries {}", search.queries))?,
    )
    .with_context(|| format!("reading queries {}", search.queries))?;
    let num_queries = queries.nrows();
    println!("Loaded {num_queries} queries, dim = {}", queries.ncols());

    // Optional groundtruth (ibin: ids are non-negative, read as u32).
    let groundtruth: Option<Matrix<u32>> = match &search.groundtruth {
        Some(gt) => Some(
            read_bin::<u32>(
                &mut storage
                    .open_reader(gt)
                    .with_context(|| format!("opening groundtruth {gt}"))?,
            )
            .with_context(|| format!("reading groundtruth {gt}"))?,
        ),
        None => None,
    };

    // --- Resolve the three index files and load PQ data ----------------------
    let pivot_path = get_pq_pivot_file(&index_load.load_path);
    let pq_data_path = get_compressed_pq_file(&index_load.load_path);
    let disk_index_path = get_disk_index_file(&index_load.load_path);

    let index_reader = DiskIndexReader::<T>::new(pivot_path, pq_data_path, storage)
        .context("failed to load PQ pivots / compressed data")?;
    println!("Loaded PQ data: {} points", index_reader.get_num_points());

    let caching_strategy = match search.num_nodes_to_cache {
        Some(n) => CachingStrategy::StaticCacheWithBfsNodes(n),
        None => CachingStrategy::None,
    };
    let reader_factory = AlignedFileReaderFactory::new(disk_index_path);
    let vertex_provider_factory = DiskVertexProviderFactory::new(reader_factory, caching_strategy)
        .context("failed to create disk vertex provider factory")?;

    // ONE searcher, thread-hint sized to the largest sweep value.
    let searcher = DiskIndexSearcher::<AdHoc<T>, _>::new(
        max_threads,
        search.search_io_limit.unwrap_or(usize::MAX),
        &index_reader,
        vertex_provider_factory,
        search.distance.into(),
        None,
    )
    .context("failed to construct DiskIndexSearcher")?;
    println!("Index loaded.\n");

    let k = search.recall_at;
    let beam_width = search.beam_width;
    let is_flat_search = search.is_flat_search;

    // Run all queries for a given (pool, L). Reusable across thread counts.
    let run_pass = |pool: &rayon::ThreadPool, l: u32| -> Result<(PassResult, Vec<u32>)> {
        let mut result_ids = vec![0u32; k as usize * num_queries];
        let mut cmps_per_query = vec![0u32; num_queries];
        let mut latency_us = vec![0.0f64; num_queries];
        let failed = AtomicBool::new(false);

        let start = Instant::now();
        pool.install(|| {
            queries
                .par_row_iter()
                .zip(result_ids.par_chunks_mut(k as usize))
                .zip(cmps_per_query.par_iter_mut())
                .zip(latency_us.par_iter_mut())
                .for_each(|(((query, id_chunk), cmps), lat)| {
                    let t = Instant::now();
                    match searcher.search(query, k, l, Some(beam_width), None, is_flat_search) {
                        Ok(res) => {
                            *cmps = res.stats.cmps;
                            for (slot, item) in id_chunk.iter_mut().zip(res.results.iter()) {
                                *slot = item.vertex_id;
                            }
                        }
                        Err(e) => {
                            eprintln!("search failed (L={l}): {e:?}");
                            failed.store(true, Ordering::Relaxed);
                        }
                    }
                    *lat = t.elapsed().as_micros() as f64;
                });
        });
        let elapsed = start.elapsed();
        if failed.load(Ordering::Relaxed) {
            bail!("one or more queries failed at L={l}");
        }

        let total_cmps: u64 = cmps_per_query.iter().map(|&c| c as u64).sum();
        let pass = PassResult {
            qps: num_queries as f32 / elapsed.as_secs_f32(),
            mean_latency_us: latency_us.iter().sum::<f64>() / num_queries as f64,
            mean_comparisons: total_cmps as f64 / num_queries as f64,
        };
        Ok((pass, result_ids))
    };

    // Sweep: one rayon pool per thread count.
    let mut rows = Vec::with_capacity(thread_counts.len());
    let mut recalls: Vec<Option<f32>> = Vec::with_capacity(search.search_list.len());

    for (ti, &tc) in thread_counts.iter().enumerate() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(tc)
            .build()
            .with_context(|| format!("failed to build rayon pool with {tc} threads"))?;

        let mut passes = Vec::with_capacity(search.search_list.len());
        for &l in &search.search_list {
            let (pass, ids) = run_pass(&pool, l)?;
            if ti == 0 {
                // recall is deterministic in (L, beam), so compute it once.
                recalls.push(groundtruth.as_ref().map(|g| recall_at_k(&ids, g, num_queries, k)));
            }
            passes.push(pass);
        }
        println!("  swept num_threads = {tc}");
        rows.push(ThreadSweepRow { num_threads: tc, passes });
    }

    Ok(SweepStats {
        recall_at: k,
        beam_width: search.beam_width,
        search_list: search.search_list.clone(),
        recalls,
        rows,
    })
}

/// recall@k = mean over queries of |found ∩ truth[..k]| / k.
fn recall_at_k(result_ids: &[u32], gt: &Matrix<u32>, num_queries: usize, k: u32) -> f32 {
    let k = k as usize;
    let gt_dim = gt.ncols();
    let mut hits = 0usize;
    for q in 0..num_queries {
        let found = &result_ids[q * k..q * k + k];
        let truth = gt.row(q);
        let take = k.min(gt_dim);
        for &id in &found[..take] {
            if truth[..take].contains(&id) {
                hits += 1;
            }
        }
    }
    hits as f32 / (num_queries * k) as f32
}

impl std::fmt::Display for SweepStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (li, &l) in self.search_list.iter().enumerate() {
            let recall = match self.recalls.get(li).copied().flatten() {
                Some(v) => format!("{:.4}", v),
                None => "n/a".to_string(),
            };
            writeln!(f, "L = {l}  (recall@{} = {recall}, beam = {})", self.recall_at, self.beam_width)?;
            writeln!(
                f,
                "  {:>8}  {:>12}  {:>9}  {:>13}  {:>12}",
                "threads", "QPS", "speedup", "mean_lat_us", "mean_comps"
            )?;
            let base_qps = self.rows.first().map(|r| r.passes[li].qps).unwrap_or(1.0);
            let mut prev_qps: Option<f32> = None;
            for row in &self.rows {
                let p = row.passes[li];
                let speedup = if base_qps > 0.0 { p.qps / base_qps } else { 0.0 };
                let marker = match prev_qps {
                    Some(pq) if row.num_threads > 1 && pq > 0.0 && p.qps < pq * 1.15 => {
                        "   <- saturating"
                    }
                    _ => "",
                };
                writeln!(
                    f,
                    "  {:>8}  {:>12.1}  {:>8.2}x  {:>13.1}  {:>12.1}{}",
                    row.num_threads, p.qps, speedup, p.mean_latency_us, p.mean_comparisons, marker
                )?;
                prev_qps = Some(p.qps);
            }
            writeln!(f)?;
        }
        Ok(())
    }
}