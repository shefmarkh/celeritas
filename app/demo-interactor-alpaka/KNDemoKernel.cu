//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file KNDemoKernel.cu
//---------------------------------------------------------------------------//
#include "KNDemoKernel.hh"

#include <alpaka/alpaka.hpp>

#include "base/ArrayUtils.hh"
#include "base/Assert.hh"
#include "base/KernelParamCalculator.cuda.hh"
#include "physics/base/ParticleTrackView.hh"
#include "base/StackAllocator.hh"
#include "physics/em/detail/KleinNishinaInteractor.hh"
#include "random/RngEngine.hh"
#include "physics/grid/XsCalculator.hh"
#include "Detector.hh"
#include "KernelUtils.hh"

using namespace celeritas;
using celeritas::detail::KleinNishinaInteractor;

namespace demo_interactor
{
  //Toggle between using CUDA and Alpaka kernel
  bool useAlpaka = false;
  //Setup infrastructure for Alpaka usage
  using namespace alpaka;
  //Define shortcuts for some alpaka items we will use
  using Dim = dim::DimInt<1>;
  using Idx = uint32_t;
  //Define the alpaka accelerator to be GPU
  using Acc = acc::AccGpuCudaRt<Dim,Idx>;  
namespace
{
//---------------------------------------------------------------------------//
// KERNELS
//---------------------------------------------------------------------------//
/*!
 * Kernel to initialize particle data.
 */
__global__ void initialize_kernel(ParamsDeviceRef const params,
                                  StateDeviceRef const  states,
                                  InitialPointers const init)
{
    
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Exit if out of range or already dead
    if (tid >= states.size())
    {
        return;
    }
    
    //printf("Create particle for thread id %d \n", tid);
    ParticleTrackView particle(params.particle, states.particle, ThreadId(tid));
    particle = init.particle;
    
    // Particles begin alive and in the +z direction
    states.direction[tid] = {0, 0, 1};
    states.position[tid]  = {0, 0, 0};
    states.time[tid]      = 0;
    states.alive[tid]     = true;
    
}

struct initialize_kernel_alpaka{
  template <typename Acc>
  ALPAKA_FN_ACC void operator()(Acc const &acc,ParamsDeviceRef const params,StateDeviceRef const  states,InitialPointers const init) const {

    
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= states.size()) return;
        
    ParticleTrackView particle(params.particle, states.particle, ThreadId(tid));
    particle = init.particle;

    // Particles begin alive and in the +z direction
    states.direction[tid] = {0, 0, 1};
    states.position[tid]  = {0, 0, 0};
    states.time[tid]      = 0;
    states.alive[tid]     = true;
    
  }
};

//---------------------------------------------------------------------------//
/*!
 * Sample cross sections and move to the collision point.
 */
__global__ void
move_kernel(ParamsDeviceRef const params, StateDeviceRef const states)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Exit if out of range or already dead
    if (tid >= states.size() || !states.alive[tid])
    {
        return;
    }

    // Construct particle accessor from immutable and thread-local data
    ParticleTrackView particle(params.particle, states.particle, ThreadId(tid));
    RngEngine         rng(states.rng, ThreadId(tid));

    // Move to collision
    XsCalculator calc_xs(params.tables.xs, params.tables.reals);
    demo_interactor::move_to_collision(particle,
                                       calc_xs,
                                       states.direction[tid],
                                       &states.position[tid],
                                       &states.time[tid],
                                       rng);
}

struct move_kernel_alpaka{
  template <typename Acc>
  ALPAKA_FN_ACC void operator()(Acc const &acc,ParamsDeviceRef const params, StateDeviceRef const states) const{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Exit if out of range or already dead
    if (tid >= states.size() || !states.alive[tid])
    {
        return;
    }

    // Construct particle accessor from immutable and thread-local data
    ParticleTrackView particle(params.particle, states.particle, ThreadId(tid));
    RngEngine         rng(states.rng, ThreadId(tid));

    // Move to collision
    XsCalculator calc_xs(params.tables.xs, params.tables.reals);
    demo_interactor::move_to_collision(particle,
                                       calc_xs,
                                       states.direction[tid],
                                       &states.position[tid],
                                       &states.time[tid],
                                       rng);
  }
};

//---------------------------------------------------------------------------//
/*!
 * Perform the iteraction plus cleanup.
 *
 * The interaction:
 * - Allocates and emits a secondary
 * - Kills the secondary, depositing its local energy
 * - Applies the interaction (updating track direction and energy)
 */
__global__ void
interact_kernel(ParamsDeviceRef const params, StateDeviceRef const states)
{
    StackAllocator<Secondary> allocate_secondaries(states.secondaries);
    unsigned int              tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Exit if out of range or already dead
    if (tid >= states.size() || !states.alive[tid])
    {
        return;
    }

    // Construct particle accessor from immutable and thread-local data
    ParticleTrackView particle(params.particle, states.particle, ThreadId(tid));
    RngEngine         rng(states.rng, ThreadId(tid));

    Detector detector(params.detector, states.detector);

    Hit h;
    h.pos    = states.position[tid];
    h.dir    = states.direction[tid];
    h.thread = ThreadId(tid);
    h.time   = states.time[tid];

    if (particle.energy() < units::MevEnergy{0.01})
    {
        // Particle is below interaction energy
        h.energy_deposited = particle.energy();

        // Deposit energy and kill
        detector.buffer_hit(h);
        states.alive[tid] = false;
        return;
    }

    // Construct RNG and interaction interfaces
    KleinNishinaInteractor interact(
        params.kn_interactor, particle, h.dir, allocate_secondaries);

    // Perform interaction: should emit a single particle (an electron)
    Interaction interaction = interact(rng);
    CELER_ASSERT(interaction);

    // Deposit energy from the secondary (effectively, an infinite energy
    // cutoff)
    {
        const auto& secondary = interaction.secondaries.front();
        h.dir                 = secondary.direction;
        h.energy_deposited    = secondary.energy;
        detector.buffer_hit(h);
    }

    // Update post-interaction state (apply interaction)
    states.direction[tid] = interaction.direction;
    particle.energy(interaction.energy);
}

struct interact_kernel_alpaka {
  template <typename Acc>
  ALPAKA_FN_ACC void operator()(Acc const &acc,ParamsDeviceRef const params, StateDeviceRef const states) const{
    StackAllocator<Secondary> allocate_secondaries(states.secondaries);
    unsigned int              tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Exit if out of range or already dead
    if (tid >= states.size() || !states.alive[tid])
    {
        return;
    }

    // Construct particle accessor from immutable and thread-local data
    ParticleTrackView particle(params.particle, states.particle, ThreadId(tid));
    RngEngine         rng(states.rng, ThreadId(tid));

    Detector detector(params.detector, states.detector);

    Hit h;
    h.pos    = states.position[tid];
    h.dir    = states.direction[tid];
    h.thread = ThreadId(tid);
    h.time   = states.time[tid];

    if (particle.energy() < units::MevEnergy{0.01})
    {
        // Particle is below interaction energy
        h.energy_deposited = particle.energy();

        // Deposit energy and kill
        detector.buffer_hit(h);
        states.alive[tid] = false;
        return;
    }

    // Construct RNG and interaction interfaces
    KleinNishinaInteractor interact(
        params.kn_interactor, particle, h.dir, allocate_secondaries);

    // Perform interaction: should emit a single particle (an electron)
    Interaction interaction = interact(rng);
    CELER_ASSERT(interaction);

    // Deposit energy from the secondary (effectively, an infinite energy
    // cutoff)
    {
        const auto& secondary = interaction.secondaries.front();
        h.dir                 = secondary.direction;
        h.energy_deposited    = secondary.energy;
        detector.buffer_hit(h);
    }

    // Update post-interaction state (apply interaction)
    states.direction[tid] = interaction.direction;
    particle.energy(interaction.energy);
  }
};

//---------------------------------------------------------------------------//
/*!
 * Bin detector hits.
 */
__global__ void
process_hits_kernel(ParamsDeviceRef const params, StateDeviceRef const states)
{
    Detector        detector(params.detector, states.detector);
    Detector::HitId hid{blockIdx.x * blockDim.x + threadIdx.x};

    if (hid < detector.num_hits())
    {
        detector.process_hit(hid);
    }
}

struct process_hits_kernel_alpaka{
  template <typename Acc>
  ALPAKA_FN_ACC void operator()(Acc const &acc,ParamsDeviceRef const params, StateDeviceRef const states) const{
    Detector        detector(params.detector, states.detector);
    Detector::HitId hid{blockIdx.x * blockDim.x + threadIdx.x};

    if (hid < detector.num_hits())
    {
        detector.process_hit(hid);
    }
  }
};

//---------------------------------------------------------------------------//
/*!
 * Clear secondaries and detector hits.
 */
__global__ void
cleanup_kernel(ParamsDeviceRef const params, StateDeviceRef const states)
{
    unsigned int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    Detector     detector(params.detector, states.detector);
    StackAllocator<Secondary> allocate_secondaries(states.secondaries);

    if (thread_idx == 0)
    {
        allocate_secondaries.clear();
        detector.clear_buffer();
    }
}

struct cleanup_kernel_alpaka{
  template <typename Acc>
  ALPAKA_FN_ACC void operator()(Acc const &acc,ParamsDeviceRef const params, StateDeviceRef const states){
    unsigned int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    Detector     detector(params.detector, states.detector);
    StackAllocator<Secondary> allocate_secondaries(states.secondaries);

    if (thread_idx == 0)
    {
        allocate_secondaries.clear();
        detector.clear_buffer();
    }
  }
};

} // namespace

//---------------------------------------------------------------------------//
// KERNEL INTERFACES
//---------------------------------------------------------------------------//
#define CDE_LAUNCH_KERNEL(NAME, BLOCK_SIZE, THREADS, ARGS...)       \
    do                                                              \
    {                                                               \
        static const KernelParamCalculator calc_kernel_params_(     \
            NAME##_kernel, #NAME, BLOCK_SIZE);                      \
        auto grid_ = calc_kernel_params_(THREADS);                  \
                                                                    \
        NAME##_kernel<<<grid_.grid_size, grid_.block_size>>>(ARGS); \
        CELER_CUDA_CHECK_ERROR();                                   \
    } while (0)

/*!
 * Initialize particle states.
 */
void initialize(const CudaGridParams&  opts,
                const ParamsDeviceRef& params,
                const StateDeviceRef&  states,
                const InitialPointers& initial)
{
    CELER_EXPECT(states.alive.size() == states.size());
    CELER_EXPECT(states.rng.size() == states.size());
    if (!useAlpaka){
      CDE_LAUNCH_KERNEL(
          initialize, opts.block_size, states.size(), params, states, initial);
    }
    else{      
      //Get the first device available of type GPU (i.e should be our sole GPU)/device
      auto const device = pltf::getDevByIdx<Acc>(0u);
      //Create a blocking queue for that device
      auto queue = queue::Queue<Acc, queue::Blocking>{device};  
      //Calculate the parameters for the kernel (blocks etc)
      auto grid_size = ( (states.size()/opts.block_size) + (states.size() % opts.block_size != 0) );
      auto workDiv = workdiv::WorkDivMembers<Dim, Idx>{static_cast<uint32_t>(opts.block_size), static_cast<uint32_t>(grid_size), static_cast<uint32_t>(1)};
      //Create a task that we can run and then run it via our queue
      initialize_kernel_alpaka initialize_kernel_alpaka;
      auto taskInitialize = kernel::createTaskKernel<Acc>(workDiv,initialize_kernel_alpaka,params,states,initial);
      queue::enqueue(queue, taskInitialize);
    }

}

//---------------------------------------------------------------------------//
/*!
 * Run an iteration.
 */
void iterate(const CudaGridParams&  opts,
             const ParamsDeviceRef& params,
             const StateDeviceRef&  states)
{

    if (!useAlpaka) {
      // Move to the collision site
      CDE_LAUNCH_KERNEL(move, opts.block_size, states.size(), params, states);
      // Perform the interaction
      CDE_LAUNCH_KERNEL(interact, opts.block_size, states.size(), params, states);
    }
    else{
      //Get the first device available of type GPU (i.e should be our sole GPU)/device
      auto const device = pltf::getDevByIdx<Acc>(0u);
      //Create a blocking queue for that device
      auto queue = queue::Queue<Acc, queue::Blocking>{device};  
      //Calculate the parameters for the kernel (blocks etc)
      auto grid_size = ( (states.size()/opts.block_size) + (states.size() % opts.block_size != 0) );
      auto workDiv = workdiv::WorkDivMembers<Dim, Idx>{static_cast<uint32_t>(opts.block_size), static_cast<uint32_t>(grid_size), static_cast<uint32_t>(1)};
      //Create tasks that we can run and then run it via our queue
      // Move to the collision site
      move_kernel_alpaka move_kernel_alpaka;
      auto taskMove = kernel::createTaskKernel<Acc>(workDiv,move_kernel_alpaka,params,states);
      queue::enqueue(queue, taskMove);
      // Perform the interaction
      interact_kernel_alpaka interact_kernel_alpaka;
      auto taskInteract = kernel::createTaskKernel<Acc>(workDiv,interact_kernel_alpaka,params,states);
      queue::enqueue(queue, taskInteract);
    }

    if (opts.sync)
    {
        // Synchronize for granular kernel timing diagnostics
        CELER_CUDA_CALL(cudaDeviceSynchronize());
    }
}

//---------------------------------------------------------------------------//
/*!
 * Clean up after an iteration.
 */
void cleanup(const CudaGridParams&  opts,
             const ParamsDeviceRef& params,
             const StateDeviceRef&  states)
{
    if(!useAlpaka){
      // Process hits from buffer to grid
      CDE_LAUNCH_KERNEL(process_hits,
                        opts.block_size,
                        states.detector.capacity(),
                        params,
                        states);

      // Clear buffers
      CDE_LAUNCH_KERNEL(cleanup, 32, 1, params, states);
    }
    else{
      //Get the first device available of type GPU (i.e should be our sole GPU)/device
      auto const device = pltf::getDevByIdx<Acc>(0u);
      //Create a blocking queue for that device
      auto queue = queue::Queue<Acc, queue::Blocking>{device};  
      //Calculate the parameters for the process_hits kernel (blocks etc)
      auto grid_size = ( (states.detector.capacity()/opts.block_size) + (states.detector.capacity() % opts.block_size != 0) );
      auto workDiv = workdiv::WorkDivMembers<Dim, Idx>{static_cast<uint32_t>(opts.block_size), static_cast<uint32_t>(grid_size), static_cast<uint32_t>(1)};
      //Create tasks that we can run and then run it via our queue
      // Process hits from buffer to grid
      process_hits_kernel_alpaka process_hits_kernel_alpaka;
      auto taskProcessHits = kernel::createTaskKernel<Acc>(workDiv,process_hits_kernel_alpaka,params,states);
      queue::enqueue(queue, taskProcessHits);
      // Clear buffers
      //Calculate different parameters for this cleanup kernel (blocks etc)
      auto grid_size_cleanup = ( (1/32) + (1 % 32 != 0) );
      auto workDiv_cleanup = workdiv::WorkDivMembers<Dim, Idx>{static_cast<uint32_t>(32), static_cast<uint32_t>(grid_size_cleanup), static_cast<uint32_t>(1)};
      cleanup_kernel_alpaka cleanup_kernel_alpaka;
      auto taskCleanup = kernel::createTaskKernel<Acc>(workDiv_cleanup,cleanup_kernel_alpaka,params,states);
      queue::enqueue(queue, taskProcessHits);

    }

    if (opts.sync)
    {
        CELER_CUDA_CALL(cudaDeviceSynchronize());
    }
}

//---------------------------------------------------------------------------//
} // namespace demo_interactor