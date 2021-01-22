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
//Make the host CPU available
static auto devHost = alpaka::pltf::getDevByIdx<alpaka::dev::DevCpu>(0u);
//Make a queue we can use (later on this is going to have to be made accessible in some nice way)
//for copying data to and from the device.
static auto queue = alpaka::queue::Queue<Acc,alpaka::queue::Blocking>{device};

namespace celeritas
{

//Default constructor (assumes sixe of zero)
template<class T>
DeviceVector<T>::DeviceVector() : bufferExtent_(static_cast<uint32_t>(0)), 
allocatedMemory_(alpaka::mem::buf::alloc<T,uint32_t>(device,bufferExtent_)) {};


//---------------------------------------------------------------------------//
/*!
 * Construct with a number of allocated elements.
 */
template<class T>
DeviceVector<T>::DeviceVector(size_type count)
    :  bufferExtent_{static_cast<uint32_t>(count)},  allocatedMemory_(alpaka::mem::buf::alloc<T,uint32_t>(device,bufferExtent_))    
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
    swap(bufferExtent_, other.bufferExtent_);
    swap(allocatedMemory_, other.allocatedMemory_);
}

//---------------------------------------------------------------------------//
/*!
 * Copy data to device.
 */
template<class T>
void DeviceVector<T>::copy_to_device(constSpan_t data)
{
    REQUIRE(data.size() == this->size());
    //Create a local memory buffer for the host data
    auto hostMemoryBuffer(alpaka::mem::buf::alloc<T,uint32_t>(devHost,bufferExtent_));
    //Now make the pointers in that buffer point at the data in the Span (data) passed into this function
    for (uint32_t counter = 0; counter < this->size(); counter++) alpaka::mem::view::getPtrNative(hostMemoryBuffer)[counter] = data[counter];
    //Finally copy the host data into the device memory allocatedMemory_
    alpaka::mem::view::copy(queue,allocatedMemory_, hostMemoryBuffer,bufferExtent_);    
}

//---------------------------------------------------------------------------//
/*!
 * Copy data to host.
 */
template<class T>
void DeviceVector<T>::copy_to_host(Span_t data) const
{    
    REQUIRE(data.size() == this->size());
    //Create a local memory buffer for the host data
    auto hostMemoryBuffer(alpaka::mem::buf::alloc<T,uint32_t>(devHost,bufferExtent_));
    //Now copy the device data into that buffer
    alpaka::mem::view::copy(queue,hostMemoryBuffer,allocatedMemory_,bufferExtent_);    
    //Now copy the data pointed at by the alpaka host memory buffer into the Span (data)
    for (uint32_t counter = 0; counter < this->size(); counter++) data[counter] = alpaka::mem::view::getPtrNative(hostMemoryBuffer)[counter];    
}

//---------------------------------------------------------------------------//
/*!
 * Get an on-device view to the data.
 */

template<class T>
auto DeviceVector<T>::device_pointers() -> Span_t
{   
  T* devicePtr = alpaka::mem::view::getPtrNative(allocatedMemory_);  
  return {devicePtr,static_cast<size_type>((bufferExtent_.maxElem()+1))};
}

//---------------------------------------------------------------------------//
/*!
 * Get an on-device view to the data.
 */

template<class T>
auto DeviceVector<T>::device_pointers() const -> constSpan_t
{    
    const T* devicePtr = reinterpret_cast<const T*>(alpaka::mem::view::getPtrNative(allocatedMemory_));
    return {devicePtr,static_cast<size_type>((bufferExtent_.maxElem()+1))};
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
