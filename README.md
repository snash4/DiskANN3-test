# DiskANN3-test
play around DiskANN3

Minimal DiskANN3 example: load a *disk* index from disk and run a KNN search.

A DiskANN3 disk index is a set of files that share a common path prefix:
`<prefix>_disk.index`          (the graph + full-precision vectors)
`<prefix>_pq_pivots.bin`       (PQ pivot / centroid table)
`<prefix>_pq_compressed.bin`   (PQ-compressed vectors held in memory)

 The path helpers in `diskann_providers::storage` turn the prefix into those three concrete file names, so you only ever pass the prefix around.

#### Usage:
``` DiskANN3-test <index_prefix> <queries.fbin> [k] [L] [beam] [metric] ```

#### Build
``` cargo build --release ```

#### Run
``` ./target/release/DiskANN3-test <index_path>   <queries.bin>  K L Beam_width l2 ```