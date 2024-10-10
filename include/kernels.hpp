#pragma once
#include "includes.hpp"

class Tensor;

Tensor get_element_wise_empty_output(const Tensor& in1, const Tensor& in2);

void add(const Tensor& in1, const Tensor& in2, Tensor& out);
void sub(const Tensor& in1, const Tensor& in2, Tensor& out);
void mul(const Tensor& in1, const Tensor& in2, Tensor& out);
void div(const Tensor& in1, const Tensor& in2, Tensor& out);