#include "tensor.hpp"

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
    return out;
}

Tensor Tensor::operator-(const Tensor& other) const {
    Tensor out = get_element_wise_empty_output(*this, other);
    sub_impl(*this, other, out);
    return out;
}

Tensor Tensor::operator*(const Tensor& other) const {
    Tensor out = get_element_wise_empty_output(*this, other);
    mul_impl(*this, other, out);
    return out;
}

Tensor Tensor::operator/(const Tensor& other) const {
    Tensor out = get_element_wise_empty_output(*this, other);
    div_impl(*this, other, out);
    return out;
}

Tensor Tensor::mm(const Tensor& other) const {
    Tensor out = get_matmul_empty_output(*this, other);
    matmul_impl(*this, other, out);
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