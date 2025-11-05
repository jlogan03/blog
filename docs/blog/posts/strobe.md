# 2023-10-21: Array Expressions without Allocation (in Rust)

<div>
  <a href=https://github.com/jlogan03>
    <img src=https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white height="28" style="padding-right:20px">
  </a>
  <a href=https://hachyderm.io/@ponderingpothos>
    <img src=https://joinmastodon.org/logos/wordmark-white-text.svg width="105" height="25">
  </a>
</div>

### Overview

- Github: https://github.com/jlogan03/strobe
- Docs: https://docs.rs/crate/strobe/latest
- Benchmarks: https://github.com/jlogan03/strobe/pull/6
- Tooling
  - Checking assembly & vectorization recipes: Compiler Explorer & cargo-asm
  - Benchmarking & perf analysis: criterion & valgrind
  - Linting: rustfmt & clippy
  - Release & versioning: release-plz & cargo-semver-checks

Strobe provides fast, low-memory, elementwise array expressions on the stack.
It is compatible with **no-std** (and **no-alloc**) environments, but _can_ allocate
for outputs as a convenience.
Its only required dependencies are `num-traits`, `libm`, and rust `core`.

In essence, this allows a similar pattern to operating over chunks of input
arrays, but extends this to **expressions of arbitrary depth**, and provides
useful guarantees to the compiler while **eliminating the need for allocation**
of intermediate results and **enabling consistent vectorization**.

### Development Process

This crate started as a thought-experiment using Rust's benchmarking utilities
on nightly to explore how much of a typical array operation's runtime was related
to allocation and how much was the actual numerical operations. As it turned out,
under sporadic compute loads (meaning, with a cold heap) more than half of the
run time of a vectorized array multiplication was spent just allocating storage.

That observation won't be evident in the benchmarks below, because criterion
does a number of cycles before the start of measurement in order to warm up
the system and prevent allocation noise from being observed. This gives better
quality data, but it's also important to keep in mind that under completely
normal conditions, the pre-allocated and non-pre-allocated versions of these
calculations give completely different performance results.

I take this observation as evidence of a need to eliminate allocation from the
evaluation of array expressions to the maximum possible extent - possibly leaving
a single allocation at the end of the expression for the outputs.

With this in mind, the bucket of parts that would become strobe moved to a single
file in Compiler Explorer and experienced a number of iterations through making changes
then doing ctrl-F for `mulpd` in the assembly to check whether the resulting calcs were
vectorizing properly.

The next key observation was that even with pre-allocated storage available on the heap,
it was faster to copy segments of data into mutable storage on the stack than to send
data back and forth from heap-allocated storage, even though this sometimes results in
more copy events in the code as it is written.

At this point, there's (1) a lot to gain by eliminating allocation for intermediate results
and (2) not much to lose by copying data around in intermediate storage on the stack. This
gives a natural form for the expression data structure and evaluation:

- Structure: Each expression node carries a fixed-size segment of storage just large
  enough to take advantage of vector operations, with controlled memory alignment to make
  sure we don't do any unaligned read/write.
- Evaluation: Each expression node populates its fixed storage by operating on data
  from the fixed storage of its child nodes.

Dead simple, no allocation, and no clever tricks to confuse the user. Each expression node
does not need any information other than what its inputs are and what operation to perform -
no multidirectional mutable linkages handled by a third orchestrator, no graph data structures,
and exactly one lifetime bound in play.

The evaluation takes place by pumping inputs and intermediate values from the leaves (inputs)
through each operator node to the root:

<iframe src="https://drive.google.com/file/d/1tmTpXXSpo6Sy5Hp3cD1lf51jjNYYGlot/preview" width="640" height="480" allow="autoplay"></iframe>

That's it. It's just a tree-structured expression, with no unnecessary fanciness.
There's nothing in there that could be removed and leave it still working.

### Benchmarks

#### Criterion

For large source arrays in nontrivial expressions, it is about **2-3x faster**
than the usual method for ergonomic array operations (allocating storage for each
intermediate result).

In fact, because it provides guarantees about the size and
memory alignment of the chunks of data over which it operates, it is even faster
than performing array operations with the full intermediate arrays pre-allocated
for most nontrivial applications (at least 3 total array operations).

This is demonstrated in a set of benchmarks run using Criterion, which
makes every effort to warm up the system in order to reduce the overhead
associated with heap allocations. Three categories of benchmarks are run,
each evaluating the same mathematical operations over the same data:

1. Slice-style vectorization, allocating for each intermediate result & the output
2. Slice-style vectorization, evaluating intermediate results into pre-allocated storage,
   then allocating once for the output
3. Strobe, using fixed-size intermediate storage on the stack, then allocating once for the output

While the allocation for the output array might give an unnecessarily pessimistic view of
the maximum throughput of the pre-allocated and Strobe versions of the calc, I believe this
gives a comparison that is more relevant to common use cases.

By the time we reach 4 multiplication operations in an expression, strobe sees about double
the throughput of the other methods for array sizes larger than about 1e5 elements, and is
only slightly slower than the fastest alternative for smaller sizes. The benefit for large
array operations is large, and the penalty for small ones is small.

![4x mul in an expression](https://user-images.githubusercontent.com/1596770/270112797-8b037c34-82d2-4582-b5b8-ce407e75575a.png)

The worst case, where Strobe is used gratuitously for a single operation with a small array,
is about 3x worse than just doing a contiguous slice operation:

![gratuitous use for a single mul](https://user-images.githubusercontent.com/1596770/270112744-6d06ab50-0432-468e-ba96-fbbdc82a4f63.png)

The above results are obtained with a 2018 Macbook Pro (Intel x86). Similar scalings are obtained
with a Ryzen 5 3600, with slight differences in the crossover point between methods, likely due to
differences in cache size and memory performance. Results from both systems used for benchmarking
and for more calculations are available [here](https://github.com/jlogan03/strobe/pull/6).

#### Cachegrind

Unit tests use arrays of size 67, 3 more than the intermediate storage size of 64, in order to
make sure that we visit the behavior of the system when it sees storage that is neither full
nor empty in a given cycle. The first 64 values will be consistently vectorized and cached properly,
as will the first two in the group of 3 values, so we can expect around 1/67th (1.5%) of the evaluations to
miss the cache due to swapping from the vector loop to the scalar loop. Cachegrind run over the unit tests
supports this napkin estimate:

```
jlogan@jlogan-MS-7C56:~/git/strobe$ valgrind --tool=cachegrind ./target/release/deps/strobe-5cb1bc954f8de3f1
==24414== Cachegrind, a cache and branch-prediction profiler
==24414== Copyright (C) 2002-2017, and GNU GPL'd, by Nicholas Nethercote et al.
==24414== Using Valgrind-3.18.1 and LibVEX; rerun with -h for copyright info
==24414== Command: ./target/release/deps/strobe-5cb1bc954f8de3f1
==24414==
--24414-- warning: L3 cache found, using its data for the LL simulation.

running 33 tests
test test::test_atanh ... ok
...(truncated)...
test test_std::test_std ... ok

test result: ok. 33 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.16s

==24414==
==24414== I   refs:      1,992,045
==24414== I1  misses:       22,729
==24414== LLi misses:        4,408
==24414== I1  miss rate:      1.14%
==24414== LLi miss rate:      0.22%
==24414==
==24414== D   refs:        736,224  (451,692 rd   + 284,532 wr)
==24414== D1  misses:       15,937  (  8,971 rd   +   6,966 wr)
==24414== LLd misses:        7,806  (  3,756 rd   +   4,050 wr)
==24414== D1  miss rate:       2.2% (    2.0%     +     2.4%  )
==24414== LLd miss rate:       1.1% (    0.8%     +     1.4%  )
==24414==
==24414== LL refs:          38,666  ( 31,700 rd   +   6,966 wr)
==24414== LL misses:        12,214  (  8,164 rd   +   4,050 wr)
==24414== LL miss rate:        0.4% (    0.3%     +     1.4%  )

```
