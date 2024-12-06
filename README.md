### MicroTorch

A good-performance, open-source simple autograd engine (similar to Pytorch) written in C++.

### Introduction

This autograd engine provides a powerful framework for automatic differentiation and backpropagation. It is designed to be efficient, and flexible. it is mainly designed to understand the internal concepts underlying deep learning.

### Key Features

- Automatic Differentiation : The engine supports numerical automatic differentiation, allowing you to compute gradients of complex functions with ease.
- Good-Performance
- Flexibility : The engine is designed to be modular and extensible, allowing developers to easily add new features and integrations.

### Usage

#### Building the Engine

To build the engine, simply run `bash build.sh` in the root directory. This will compile the source code into a shared library that can be linked against other projects.

#### Unit tests

To run unit tests you simply run this program `./build/bin/micro_torch_unit_tests.exe` after building the root directory.

Unit tests covers

- Basic operations like (+, -, \*, /, ...).
- Automatic differentiation.
- Simple networks like (not, and, or) gates.

#### Using the Engine

Here is a simple network (Not Gate)

```cpp
#include <tensor.hpp>

int main() {
    Tensor data({2}), weights({1}), bias({1}), out({2});
    data = {0.f, 1.f};
    out = {1.f, 0.f};

    weights = {rand() / (float)RAND_MAX};
    bias = {rand() / (float)RAND_MAX};

    weights.requires_grad(true);
    bias.requires_grad(true);

    float lr = 0.01;
    for (int i = 0; i < 100; i++) {
        auto pred = data * weights + bias;
        auto loss = pred - out;
        loss = loss * loss;
        loss.backward();

        auto weights_grad = weights.grad();
        auto bias_grad = bias.grad();

        weights = weights - weights_grad * lr;
        bias = bias - bias_grad * lr;
    }

    auto pred = data * weights + bias;

    LOG(INFO) << pred;

    EXPECT_GE((float)pred[{0}], 0.5f);
    EXPECT_LE((float)pred[{1}], 0.5f);

    return 0;
}
```

### License

This project is licensed under the MIT License. See LICENSE.txt for details.

### Contributing

We welcome contributions to this project! If you'd like to contribute, please submit a pull request with your changes and follow our guidelines for code reviews.
