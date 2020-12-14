//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Utils.cuda.cc
//---------------------------------------------------------------------------//
#include "Utils.hh"

#include "celeritas_config.h"
#if CELERITAS_USE_CUDA
#    include <cuda_runtime_api.h>
#endif
#include "base/Assert.hh"

#include <alpaka/alpaka.hpp>

namespace celeritas
{
//---------------------------------------------------------------------------//
// Initialize device in a round-robin fashion from a communicator
void initialize_device(const Communicator& comm)
{
#if CELERITAS_USE_CUDA
    // Get number of devices    
    using Dim = alpaka::dim::DimInt<1>;
    using Idx = uint32_t;
    using Acc = alpaka::acc::AccGpuCudaRt<Dim, Idx>;
    std::size_t num_devices = alpaka::pltf::getDevCount<Acc>();    
    CHECK(num_devices > 0);

    // Set device based on communicator
    //Alpaka has no equivalant of cudaSetDevice - they seem to claim it is not needed.
    //int device_id = comm.rank() % num_devices;
    //CELER_CUDA_CALL(cudaSetDevice(device_id));
#else
    (void)sizeof(comm);
#endif
}

//---------------------------------------------------------------------------//
} // namespace celeritas
