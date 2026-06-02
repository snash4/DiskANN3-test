# DiskANN3-test

Minimal DiskANN3 example: load a *disk* index from disk and run a KNN search.

Reproduces the structure of DiskANN3's own `diskann-benchmark` disk-index job: a JSON spec deserializes into `DiskIndexOperation { source, search_phase }`, is validated, then dispatched. The `Load` source loads an existing disk index and runs the search phase over each `L` in `search_list`.

#### Build
``` cargo build --release ```

#### Usage:
To get json format input. 
``` cargo run --release -- --print-example > job.json```

Update the parameters and run
#### Run
``` cargo run --release -- job.json```