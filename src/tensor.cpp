#include "tensor.hpp"

bool Tensor::enable_global_grad = true;

uint32_t Tensor::Size() const {
    LOG_IF(FATAL, m_shape.size() == 0);
    uint32_t size = 1;
    for (size_t i = 0; i < m_shape.size(); i++) {
        size *= m_shape[i];
    }

    return size;
}

void Tensor::SetDefaultStrides() {
    LOG_IF(FATAL, m_shape.size() == 0);

    m_stride.resize(m_shape.size());
    m_stride[m_shape.size() - 1] = 1;

    for (int8_t i = (int8_t)m_shape.size() - 2; i >= 0; i--) {
        m_stride[i] = m_stride[i + 1] * m_shape[i + 1];
    }
}

Tensor::Element& Tensor::operator[](const std::vector<uint32_t>& indices) {
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
    add_impl(*this, other, out);

    if (Tensor::enable_global_grad) {
        if (this->m_requires_grad) {
            out.m_parents.push_back(std::shared_ptr<Tensor>(const_cast<Tensor*>(this)));
        } else {
            out.m_parents.push_back(std::make_shared<Tensor>(*this));
        }

        if (other.m_requires_grad) {
            out.m_parents.push_back(std::shared_ptr<Tensor>(&const_cast<Tensor&>(other)));
        } else {
            out.m_parents.push_back(std::make_shared<Tensor>(other));
        }

        if (this->m_requires_grad || other.m_requires_grad) {
            out.m_requires_grad = true;
            out.m_grad_fn = Tensor::add_backward_impl;
        }
    }
    return out;
}

Tensor Tensor::operator-(const Tensor& other) const {
    Tensor out = get_element_wise_empty_output(*this, other);
    sub_impl(*this, other, out);

    if (Tensor::enable_global_grad) {
        if (this->m_requires_grad) {
            out.m_parents.push_back(std::shared_ptr<Tensor>(const_cast<Tensor*>(this)));
        } else {
            out.m_parents.push_back(std::make_shared<Tensor>(*this));
        }

        if (other.m_requires_grad) {
            out.m_parents.push_back(std::shared_ptr<Tensor>(&const_cast<Tensor&>(other)));
        } else {
            out.m_parents.push_back(std::make_shared<Tensor>(other));
        }

        if (this->m_requires_grad || other.m_requires_grad) {
            out.m_requires_grad = true;
            out.m_grad_fn = Tensor::sub_backward_impl;
        }
    }
    return out;
}

Tensor Tensor::operator*(const Tensor& other) const {
    Tensor out = get_element_wise_empty_output(*this, other);
    mul_impl(*this, other, out);

    if (Tensor::enable_global_grad) {
        if (this->m_requires_grad) {
            out.m_parents.push_back(std::shared_ptr<Tensor>(const_cast<Tensor*>(this)));
        } else {
            out.m_parents.push_back(std::make_shared<Tensor>(*this));
        }

        if (other.m_requires_grad) {
            out.m_parents.push_back(std::shared_ptr<Tensor>(&const_cast<Tensor&>(other)));
        } else {
            out.m_parents.push_back(std::make_shared<Tensor>(other));
        }

        if (this->m_requires_grad || other.m_requires_grad) {
            out.m_requires_grad = true;
            out.m_grad_fn = Tensor::mul_backward_impl;
        }
    }
    return out;
}

Tensor Tensor::operator/(const Tensor& other) const {
    Tensor out = get_element_wise_empty_output(*this, other);
    div_impl(*this, other, out);

    if (Tensor::enable_global_grad) {
        if (this->m_requires_grad) {
            out.m_parents.push_back(std::shared_ptr<Tensor>(const_cast<Tensor*>(this)));
        } else {
            out.m_parents.push_back(std::make_shared<Tensor>(*this));
        }

        if (other.m_requires_grad) {
            out.m_parents.push_back(std::shared_ptr<Tensor>(&const_cast<Tensor&>(other)));
        } else {
            out.m_parents.push_back(std::make_shared<Tensor>(other));
        }

        if (this->m_requires_grad || other.m_requires_grad) {
            out.m_requires_grad = true;
            out.m_grad_fn = Tensor::div_backward_impl;
        }
    }
    return out;
}

Tensor Tensor::mm(const Tensor& other) const {
    Tensor out = get_matmul_empty_output(*this, other);
    matmul_impl(*this, other, out);

    if (Tensor::enable_global_grad) {
        if (this->m_requires_grad) {
            out.m_parents.push_back(std::shared_ptr<Tensor>(const_cast<Tensor*>(this)));
        } else {
            out.m_parents.push_back(std::make_shared<Tensor>(*this));
        }

        if (other.m_requires_grad) {
            out.m_parents.push_back(std::shared_ptr<Tensor>(&const_cast<Tensor&>(other)));
        } else {
            out.m_parents.push_back(std::make_shared<Tensor>(other));
        }

        if (this->m_requires_grad || other.m_requires_grad) {
            out.m_requires_grad = true;
            out.m_grad_fn = Tensor::matmul_backward_impl;
        }
    }
    return out;
}

Tensor::Element& Tensor::operator[](uint32_t offset) {
    LOG_IF(FATAL, offset >= Size()) << "index out of range";
    return *reinterpret_cast<Element*>(m_storage.at((m_offset + offset) * sizeof(Element)));
}

Tensor::Element Tensor::broadcasted_read(const std::vector<uint32_t>& indices) const {
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

void Tensor::topological_sort(Tensor& curr, std::vector<Tensor*>& list, std::unordered_set<Tensor*>& visited) {
    if (visited.count(&curr)) return;
    visited.insert(&curr);
    for (auto& p : curr.m_parents) {
        topological_sort(*p, list, visited);
    }
    list.push_back(&curr);
}

void Tensor::backward() {
    if (!this->m_requires_grad) return;
    // Need to topologically sort the graph
    std::vector<Tensor*> list;
    std::unordered_set<Tensor*> visited;

    topological_sort(*this, list, visited);
    this->grad = std::make_shared<Tensor>(this->m_shape);
    *(this->grad) = 1;

    for (int32_t i = int32_t(list.size()) - 1; i >= 0; i--) {
        if (list[i]->m_grad_fn == nullptr) continue;
        list[i]->m_grad_fn(*list[i]);
    }
}