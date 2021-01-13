//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file DeviceVector.i.hh
//---------------------------------------------------------------------------//

using Dim = alpaka::dim::DimInt<1>;
//Hard code for GPU right now for initial testing.
//Eventually template class on Acc so we can switch?
using Acc = alpaka::acc::AccGpuCudaRt<Dim, uint32_t>;
//Make the first device (GPU) available
static auto const device = alpaka::pltf::getDevByIdx<Acc>(0u);

namespace celeritas
{

//Default constructor (assumes sixe of zero)
template<class T>
DeviceVector<T>::DeviceVector() :  allocation_(), size_(0), bufferExtent_(static_cast<uint32_t>(0)), allocatedMemory_(alpaka::mem::buf::alloc<T,uint32_t>(device,bufferExtent_)) {};


//---------------------------------------------------------------------------//
/*!
 * Construct with a number of allocated elements.
 */
template<class T>
DeviceVector<T>::DeviceVector(size_type count)
    :  allocation_(count * sizeof(T)), size_(count), bufferExtent_{static_cast<uint32_t>(count)},  allocatedMemory_(alpaka::mem::buf::alloc<T,uint32_t>(device,bufferExtent_))
{
}

//---------------------------------------------------------------------------//
/*!
 * Get the device data pointer.
 */
template<class T>
void DeviceVector<T>::swap(DeviceVector& other) noexcept
{
    using std::swap;
    swap(size_, other.size_);
    swap(allocation_, other.allocation_);
}

//---------------------------------------------------------------------------//
/*!
 * Copy data to device.
 */
template<class T>
void DeviceVector<T>::copy_to_device(constSpan_t data)
{
    REQUIRE(data.size() == this->size());
    allocation_.copy_to_device(
        {reinterpret_cast<const byte*>(data.data()), data.size() * sizeof(T)});
}

//---------------------------------------------------------------------------//
/*!
 * Copy data to host.
 */
template<class T>
void DeviceVector<T>::copy_to_host(Span_t data) const
{
    REQUIRE(data.size() == this->size());
    allocation_.copy_to_host(
        {reinterpret_cast<byte*>(data.data()), data.size() * sizeof(T)});
}

//---------------------------------------------------------------------------//
/*!
 * Get an on-device view to the data.
 */
template<class T>
auto DeviceVector<T>::device_pointers() -> Span_t
{
    return {reinterpret_cast<T*>(allocation_.device_pointers().data()),
            this->size()};
}

//---------------------------------------------------------------------------//
/*!
 * Get an on-device view to the data.
 */
template<class T>
auto DeviceVector<T>::device_pointers() const -> constSpan_t
{
    return {reinterpret_cast<const T*>(allocation_.device_pointers().data()),
            this->size()};
}

//---------------------------------------------------------------------------//
/*!
 * Swap two vectors.
 */
template<class T>
void swap(DeviceVector<T>& a, DeviceVector<T>& b) noexcept
{
    return a.swap(b);
}

//---------------------------------------------------------------------------//
} // namespace celeritas
