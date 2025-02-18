#include "tensor.hpp"

namespace micro {

static bool enable_global_grad = true;

void with_no_grad() { enable_global_grad = false; }

void with_grad() { enable_global_grad = true; }

std::ostream& operator<<(std::ostream& os, const Type& type) {
#define ToOStream(type, st) \
    case type: {            \
        os << #st;          \
    } break

    switch (type) {
        ToOStream(Type::UINT32, uint32);
        ToOStream(Type::INT32, int32);
        ToOStream(Type::FLOAT32, float32);
        ToOStream(Type::UNKONWN, unknown);
        default:
            break;
    }

#undef ToOStream

    return os;
}

size_t Tensor::size() const {
    LOG_IF(FATAL, m_shape.size() == 0);
    size_t size = 1;
    for (size_t i = 0; i < m_shape.size(); i++) {
        size *= m_shape[i];
    }

    return size;
}

void Tensor::set_default_strides() {
    LOG_IF(FATAL, m_shape.size() == 0);

    m_stride.resize(m_shape.size());
    m_stride[m_shape.size() - 1] = 1;

    for (int8_t i = (int8_t)m_shape.size() - 2; i >= 0; i--) {
        m_stride[i] = m_stride[i + 1] * m_shape[i + 1];
    }
}

Tensor Tensor::grad() {
    LOG_IF(FATAL, !m_saved_context) << "Trying to read gradients from a tensor without gradients";
    LOG_IF(FATAL, !m_saved_context->grad()) << "Trying to read gradients from a tensor without gradients";
    return *(m_saved_context->grad());
}

void Tensor::reset_grad() {
    LOG_IF(FATAL, !m_saved_context) << "Trying to read gradients from a tensor without gradients";
    m_saved_context->grad() = nullptr;
}

Element& Tensor::operator[](const std::vector<uint32_t>& indices) {
    LOG_IF(FATAL, indices.size() != m_shape.size())
        << "Indices size=" << indices.size() << " don't match the full_shape=" << m_shape.size();
    uint32_t offset = 0, i = 0;

    for (auto idx : indices) {
        LOG_IF(FATAL, idx >= m_shape[i]) << "index is out of range, full_shape= " << m_shape[i] << " and index=" << idx;
        offset += idx * m_stride[i];
        i++;
    }

    return this->operator[](offset);
}

Tensor Tensor::operator+(const Tensor& other) const {
    Tensor out = get_element_wise_empty_output(*this, other);
    add_forward_impl(*this, other, out);

    if (!enable_global_grad || !(this->m_requires_grad || other.m_requires_grad)) return out;

    out.m_saved_context->save_for_backward({*this, other});
    out.m_requires_grad = true;
    out.m_grad_fn = add_backward_impl;
    return out;
}

Tensor Tensor::operator-(const Tensor& other) const {
    Tensor out = get_element_wise_empty_output(*this, other);
    sub_forward_impl(*this, other, out);

    if (!enable_global_grad || !(this->m_requires_grad || other.m_requires_grad)) return out;

    out.m_saved_context->save_for_backward({*this, other});
    out.m_requires_grad = true;
    out.m_grad_fn = sub_backward_impl;
    return out;
}

Tensor Tensor::operator*(const Tensor& other) const {
    Tensor out = get_element_wise_empty_output(*this, other);
    mul_forward_impl(*this, other, out);

    if (!enable_global_grad || !(this->m_requires_grad || other.m_requires_grad)) return out;

    out.m_saved_context->save_for_backward({*this, other});
    out.m_requires_grad = true;
    out.m_grad_fn = mul_backward_impl;
    return out;
}

Tensor Tensor::operator/(const Tensor& other) const {
    Tensor out = get_element_wise_empty_output(*this, other);
    div_forward_impl(*this, other, out);

    if (!enable_global_grad || !(this->m_requires_grad || other.m_requires_grad)) return out;

    out.m_saved_context->save_for_backward({*this, other});
    out.m_requires_grad = true;
    out.m_grad_fn = div_backward_impl;
    return out;
}

Tensor Tensor::mm(const Tensor& other) const {
    Tensor out = get_matmul_empty_output(*this, other);
    matmul_forward_impl(*this, other, out);
    if (!enable_global_grad || !(this->m_requires_grad || other.m_requires_grad)) return out;

    out.m_saved_context->save_for_backward({*this, other});
    out.m_requires_grad = true;
    out.m_grad_fn = matmul_backward_impl;

    return out;
}

Tensor Tensor::sum(uint32_t dim, bool keep_dims) const {
    auto out_shape = this->m_shape;
    LOG_IF(FATAL, dim >= (uint32_t)out_shape.size()) << "Trying to sum over non-existing dimension";

    out_shape[dim] = 1;

    Tensor out(out_shape, this->m_dtype);

    sum_forward_impl(*this, dim, out);

    if (!keep_dims && out_shape.size() > 1) {
        out.m_shape.erase(out.m_shape.begin() + dim);
        out.set_default_strides();
    }

    if (!enable_global_grad || !this->m_requires_grad) return out;

    out.m_saved_context->save_for_backward({*this});
    out.m_requires_grad = true;
    out.m_grad_fn = sum_backward_impl;

    return out;
}

Element Tensor::broadcasted_read(const std::vector<uint32_t>& indices) const {
    int32_t nindecies = indices.size();
    int32_t ndims = m_shape.size();
    int32_t i = 0, j = 0;
    uint32_t offset = 0;

    for (auto idx : indices) {
        if ((nindecies - i) > ndims) {
            i++;
            continue;
        }

        if (idx < m_shape[j]) {
            offset += idx * m_stride[j];
        } else {
            LOG_IF(FATAL, m_shape[j] != 1) << "Broadcasting failed";
        }

        j++;
    }

    return this->operator[](offset);
}

void Tensor::topological_sort(Tensor& curr, std::vector<Tensor>& list,
                              std::unordered_set<std::shared_ptr<AutogradContext>>& visited) {
    if (visited.count(curr.m_saved_context)) return;
    visited.insert(curr.m_saved_context);

    auto parents = curr.m_saved_context->get_saved_variables();

    for (auto p : parents) {
        topological_sort(p, list, visited);
    }

    list.push_back(curr);
}

void Tensor::backward() {
    if (!this->m_requires_grad) return;
    // Need to topologically sort the graph
    std::vector<Tensor> list;
    std::unordered_set<std::shared_ptr<AutogradContext>> visited;

    topological_sort(*this, list, visited);
    m_saved_context->grad() = std::make_shared<Tensor>(m_shape);
    *(m_saved_context->grad()) = 1;

    for (int32_t i = int32_t(list.size()) - 1; i >= 0; i--) {
        if (list[i].m_grad_fn == nullptr) continue;
        list[i].m_grad_fn(list[i]);
    }
}

};  // namespace micro