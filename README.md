# dezero_rs

`dezero_rs` is a Rust-based clone of the `dezero` framework, which is originally designed for deep learning experiments. This project aims to replicate `dezero`'s functionalities in pure Rust, utilizing a minimal number of external crates.

## Features

- **Pure Rust Implementation:** `dezero_rs` is built entirely in Rust, aiming to leverage the language's safety and performance characteristics.
- **Manual Linear Algebra:** Linear algebra operations are implemented manually using vectors. This approach, while educational, results in slower computation speeds.

## Current Limitations

- **Performance:** Due to the manual implementation of linear algebra components, `dezero_rs` suffers from performance issues, particularly in computation-intensive tasks.
- **Learning Capability:** Currently, `dezero_rs` is unable to successfully learn from the MNIST dataset, a fundamental benchmark in machine learning for handwritten digit classification.

## Contributions

We welcome contributions to `dezero_rs`, especially in areas that could help overcome its current limitations:

- **Performance Optimization:** Suggestions and contributions that can speed up linear algebra operations or overall computation.
- **Bug Fixes:** Identifying and resolving issues that prevent successful learning from the MNIST dataset.
- **Feature Enhancement:** Adding features to more closely align with the original `dezero` framework's capabilities.

For more information on contributing, please refer to the [CONTRIBUTING.md](CONTRIBUTING.md) file.

## License

`dezero_rs` is released under the MIT License. See the [LICENSE](LICENSE) file for more details.
