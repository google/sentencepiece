# Testing sentencepiece

The commands below are tested in the poolside-sentencepiece Docker image
([Dockerfile](https://github.com/poolsideai/orchestrator/blob/main/encoder/Dockerfile.dev)).

## Build

For a clean build, delete the build folder if one already exists. Then run the
following command which uses `ninja-build`:

```bash
cmake -D CMAKE_BUILD_TYPE=RelWithDebInfo -D SPM_ENABLE_TCMALLOC=on \
 -D SPM_ENABLE_NFKC_COMPILE=on -D SPM_ENABLE_SHARED=off \
 -D SPM_SANITIZE=off -D SPM_DISABLE_LTO=off \
 -S . -B build -G Ninja
```

Note: The `-S` option specifies path to source, and `-B` specifies path to
build.

Then you have to call the following command to actually build the targets.

```
cmake --build build
```

## Test

For testing, you have to build the project as before, but you have to pass in
`-DSPM_BUILD_TEST=ON` to build the test targets:

```bash
cmake -D SPM_BUILD_TEST=ON -D CMAKE_BUILD_TYPE=RelWithDebInfo -D SPM_ENABLE_TCMALLOC=on \
 -D SPM_ENABLE_NFKC_COMPILE=on -D SPM_ENABLE_SHARED=off \
 -D SPM_SANITIZE=off -D SPM_DISABLE_LTO=off \
 -S . -B build -GNinja
```

Run the following command as before:

```bash
cmake --build build
```

Finally, run the following command to build and run the tests.

```
cmake --build build --target test
```
