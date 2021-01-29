//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file DeviceVector.hh
//---------------------------------------------------------------------------//
#pragma once

#include <type_traits>
#include "Span.hh"
#include "Types.hh"
#include "detail/InitializedValue.hh"
#include <alpaka/alpaka.hpp>

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Host-compiler-friendly vector for uninitialized device-storage data.
 *
 * This class does *not* perform initialization on the data. The host code
 * must define and copy over suitable data. For more complex data usage
 * (dynamic resizing and assignment without memory reallocation) uses \c
 * thrust::device_vector.
 *
 * \code
    DeviceVector<double> myvec(100);
    myvec.copy_to_device(make_span(hostvec));
    myvec.copy_to_host(make_span(hostvec));
   \endcode
 */
template<class T>
class DeviceVector
{
    static_assert(std::is_trivially_copyable<T>::value,
                  "DeviceVector element is not trivially copyable");

  public:
    //@{
    //! Type aliases
    using value_type  = T;
    using Span_t      = span<T>;
    using constSpan_t = span<const T>;
    //@}

  public:
    // Construct with no elements - no longer uses default keyword, because alpaka class data members
    //in this class do not have default constructors to use!
    DeviceVector();

    // Construct with a number of elements
    explicit DeviceVector(size_type count);

    // Swap with another vector
    inline void swap(DeviceVector& other) noexcept;

    // >>> ACCESSORS

    //! Get the number of elements allocated
    size_type size() const { return bufferExtent_.max(); }

    //! Whether any elements are stored
    bool empty() const { return bufferExtent_.max() == 0; }

    // >>> DEVICE ACCESSORS

    // Copy data to device
    inline void copy_to_device(constSpan_t host_data);

    // Copy data to host
    inline void copy_to_host(Span_t host_data) const;

    // Get a mutable view to device data
    inline Span_t device_pointers();

    // Get a const view to device data
    inline constSpan_t device_pointers() const;

  private:  
    alpaka::vec::Vec<alpaka::dim::DimInt<1>, uint32_t> bufferExtent_;
    alpaka::mem::buf::BufUniformCudaHipRt<T, alpaka::dim::DimInt<1UL>, uint32_t> allocatedMemory_;    
};

// Swap two vectors
template<class T>
inline void swap(DeviceVector<T>&, DeviceVector<T>&) noexcept;

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "DeviceVector.i.hh"
