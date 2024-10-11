#pragma once
#include <cassert>

#include "includes.hpp"
#include "storage.hpp"

class Tensor {
   public:
    enum class Type : uint8_t { UINT32 = 0, INT32, FLOAT32, UNKONWN };

    struct Element {
        union Data {
            uint32_t u32 = 0;
            int32_t i32;
            float f32;
        } data;

        Element() = default;

        template <typename T>
        Element(T value) {
            if constexpr (std::is_same_v<T, int32_t>) {
                data.i32 = value;
            } else if constexpr (std::is_same_v<T, uint32_t>) {
                data.i32 = value;
            } else if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
                data.f32 = float(value);
            } else {
                LOG(FATAL) << "Element Type is not supported";
            }
        }

        operator float() const { return this->data.f32; }

        operator float&() { return this->data.f32; }

        operator int32_t() const { return this->data.i32; }

        operator int32_t&() { return this->data.i32; }

        operator uint32_t() const { return this->data.u32; }

        operator uint32_t&() { return this->data.u32; }
    };

   public:
    Tensor(std::vector<uint32_t> shape, Type dtype = Type::FLOAT32) : m_dtype(dtype), m_shape(std::move(shape)) {
        SetDefaultStrides();
        m_storage = Storage(NumberOfBytes());
    }

    Tensor(std::vector<uint32_t> shape, std::vector<uint32_t>& stride, Type dtype = Type::FLOAT32)
        : m_dtype(dtype), m_shape(std::move(shape)), m_stride(std::move(stride)) {
        LOG_IF(FATAL, shape.size() != stride.size());

        m_storage = Storage(NumberOfBytes());
    }

    uint32_t Size() const {
        LOG_IF(FATAL, m_shape.size() == 0);
        uint32_t size = 1;
        for (size_t i = 0; i < m_shape.size(); i++) {
            size *= m_shape[i];
        }

        return size;
    }

    uint32_t NumberOfBytes() const { return Size() * sizeof(Element); }

    void SetDefaultStrides() {
        LOG_IF(FATAL, m_shape.size() == 0);

        m_stride.resize(m_shape.size());
        m_stride[m_shape.size() - 1] = 1;

        for (int8_t i = (int8_t)m_shape.size() - 2; i >= 0; i--) {
            m_stride[i] = m_stride[i + 1] * m_shape[i + 1];
        }
    }

    Element operator[](const std::vector<uint32_t>& indices) const {
        return const_cast<Tensor*>(this)->operator[](indices);
    }

    Element& operator[](const std::vector<uint32_t>& indices) {
        LOG_IF(FATAL, indices.size() != m_shape.size())
            << "Indices size=" << indices.size() << " don't match the full_shape=" << m_shape.size();
        uint32_t offset = 0, i = 0;

        for (auto idx : indices) {
            LOG_IF(FATAL, idx >= m_shape[i])
                << "index is out of range, full_shape= " << m_shape[i] << " and index=" << idx;
            offset += idx * m_stride[i];
            i++;
        }

        return this->operator[](offset);
    }

    Element& at(const std::vector<uint32_t>& indices) { return this->operator[](indices); }

    Tensor operator+(const Tensor& other) {
        Tensor out = get_element_wise_empty_output(*this, other);
        add_impl(*this, other, out);
        return out;
    }

    Tensor operator-(const Tensor& other) {
        Tensor out = get_element_wise_empty_output(*this, other);
        sub_impl(*this, other, out);
        return out;
    }

    Tensor operator*(const Tensor& other) {
        Tensor out = get_element_wise_empty_output(*this, other);
        mul_impl(*this, other, out);
        return out;
    }

    Tensor operator/(const Tensor& other) {
        Tensor out = get_element_wise_empty_output(*this, other);
        div_impl(*this, other, out);
        return out;
    }

#define WRITE_ELEMENT(out, value)                                           \
    {                                                                       \
        switch (m_dtype) {                                                  \
            case Type::UINT32:                                              \
                out.data.i32 = uint32_t(value);                             \
                break;                                                      \
            case Type::INT32:                                               \
                out.data.i32 = int32_t(value);                              \
                break;                                                      \
            case Type::FLOAT32:                                             \
                out.data.f32 = float(value);                                \
                break;                                                      \
            default:                                                        \
                LOG(FATAL) << "Can't write value with unknown tensor type"; \
        }                                                                   \
    }

    template <typename T>
    Tensor operator+(T value) {
        Tensor tensor({1});
        WRITE_ELEMENT(tensor[0], value);
        return this->operator+(tensor);
    }

    template <typename T>
    void operator+=(T value) {
        Tensor tensor({1});
        WRITE_ELEMENT(tensor[0], value);
        add_impl(*this, tensor, *this);
    }

    template <typename T>
    Tensor operator-(T value) {
        Tensor tensor({1});
        WRITE_ELEMENT(tensor[0], value);
        return this->operator-(tensor);
    }

    template <typename T>
    void operator-=(T value) {
        Tensor tensor({1});
        WRITE_ELEMENT(tensor[0], value);
        sub_impl(*this, tensor, *this);
    }

    template <typename T>
    Tensor operator*(T value) {
        Tensor tensor({1});
        WRITE_ELEMENT(tensor[0], value);
        return this->operator*(tensor);
    }

    template <typename T>
    void operator*=(T value) {
        Tensor tensor({1});
        WRITE_ELEMENT(tensor[0], value);
        mul_impl(*this, tensor, *this);
    }

    template <typename T>
    Tensor operator/(T value) {
        Tensor tensor({1});
        WRITE_ELEMENT(tensor[0], value);
        return this->operator/(tensor);
    }

    template <typename T>
    void operator/=(T value) {
        Tensor tensor({1});
        WRITE_ELEMENT(tensor[0], value);
        div_impl(*this, tensor, *this);
    }

    template <typename T>
    void operator=(T value) {
        for (uint32_t i = 0; i < Size(); i++) {
            WRITE_ELEMENT(this->operator[](i), value);
        }
    }

#undef WRITE_ELEMENT

    Tensor mm(const Tensor& other) {
        Tensor out = get_matmul_empty_output(*this, other);
        matmul_impl(*this, other, out);
        return out;
    }

    friend std::ostream& operator<<(std::ostream& os, Tensor& t);
    friend Tensor get_element_wise_empty_output(const Tensor& in1, const Tensor& in2);
    friend Tensor get_matmul_empty_output(const Tensor& in1, const Tensor& in2);
    friend void add_impl(const Tensor& in1, const Tensor& in2, Tensor& out);
    friend void sub_impl(const Tensor& in1, const Tensor& in2, Tensor& out);
    friend void mul_impl(const Tensor& in1, const Tensor& in2, Tensor& out);
    friend void div_impl(const Tensor& in1, const Tensor& in2, Tensor& out);
    friend void matmul_impl(const Tensor& in1, const Tensor& in2, Tensor& out);

   private:
    Element operator[](uint32_t offset) const { return const_cast<Tensor*>(this)->operator[](offset); }

    Element& operator[](uint32_t offset) {
        LOG_IF(FATAL, offset >= Size()) << "index out of range";
        return *reinterpret_cast<Element*>(m_storage.at((m_offset + offset) * sizeof(Element)));
    }

    Element broadcasted_read(const std::vector<uint32_t>& indices) const {
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

   private:
    Tensor::Type m_dtype = Tensor::Type::FLOAT32;
    int64_t m_offset = 0;

    std::vector<uint32_t> m_shape, m_stride;
    Storage m_storage;
};