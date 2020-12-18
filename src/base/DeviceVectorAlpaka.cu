//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file DeviceVector.i.hh
//---------------------------------------------------------------------------//

#include "base/DeviceVectorAlpaka.hh"

#include <alpaka/alpaka.hpp>

using namespace alpaka;
using Dim = dim::DimInt<1>;

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with a number of allocated elements.
 */
template<class T>
DeviceVectorAlpaka<T>::DeviceVectorAlpaka(size_type count)
    : size_(count)
{

  //Hard code for GPU right now for initial testing.
  ///Eventually template class on Acc so we can switch?
  using Acc = acc::AccGpuCudaRt<Dim, uint32_t>;
  //Get the first device (GPU) available
  auto const device = pltf::getDevByIdx<Acc>(0u);
  //Set the size of our list of data  
  vec::Vec<Dim, uint32_t> bufferExtent{count};
  //Allocate the memory for T bufferExtent times on the device,
  auto test = alpaka::mem::buf::alloc<T,uint32_t>(device,bufferExtent);

}

//---------------------------------------------------------------------------//
/*!
 * Get the device data pointer.
 */
template<class T>
void DeviceVectorAlpaka<T>::swap(DeviceVectorAlpaka& other) noexcept
{
    using std::swap;
    swap(size_, other.size_);
}

//---------------------------------------------------------------------------//
/*!
 * Copy data to device.
 */
template<class T>
void DeviceVectorAlpaka<T>::copy_to_device(constSpan_t data)
{
    REQUIRE(data.size() == this->size());
    //allocation_.copy_to_device(
    //    {reinterpret_cast<const byte*>(data.data()), data.size() * sizeof(T)});
}

//---------------------------------------------------------------------------//
/*!
 * Copy data to host.
 */
template<class T>
void DeviceVectorAlpaka<T>::copy_to_host(Span_t data) const
{
    REQUIRE(data.size() == this->size());
    //allocation_.copy_to_host(
    //    {reinterpret_cast<byte*>(data.data()), data.size() * sizeof(T)});
}

//---------------------------------------------------------------------------//
/*!
 * Get an on-device view to the data.
 */
template<class T>
auto DeviceVectorAlpaka<T>::device_pointers() -> Span_t
{
    //return {reinterpret_cast<T*>(allocation_.device_pointers().data()),
    //        this->size()};
    //Create a span_t from Alpaka
    return nullptr;
}

//---------------------------------------------------------------------------//
/*!
 * Get an on-device view to the data.
 */
template<class T>
auto DeviceVectorAlpaka<T>::device_pointers() const -> constSpan_t
{
    //return {reinterpret_cast<const T*>(allocation_.device_pointers().data()),
    //        this->size()};
    //Create a constSpan_t from Alpaka
    return nullptr;
}

//---------------------------------------------------------------------------//
/*!
 * Swap two vectors.
 */
template<class T>
void swap(DeviceVectorAlpaka<T>& a, DeviceVectorAlpaka<T>& b) noexcept
{
    return a.swap(b);
}

//---------------------------------------------------------------------------//
} // namespace celeritas
