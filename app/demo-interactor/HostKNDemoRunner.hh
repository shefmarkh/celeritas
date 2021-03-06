//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file HostKNDemoRunner.hh
//---------------------------------------------------------------------------//
#pragma once

#include "physics/base/ParticleParams.hh"
#include "physics/base/ParticleStatePointers.hh"
#include "physics/em/KleinNishinaInteractorPointers.hh"
#include "KNDemoIO.hh"
#include "PhysicsArrayParams.hh"

namespace demo_interactor
{
//---------------------------------------------------------------------------//
/*!
 * Run interactions on the host CPU.
 *
 * This is an analog to the demo_interactor::KNDemoRunner for device simulation
 * but does all the transport directly on the CPU side.
 */
class HostKNDemoRunner
{
  public:
    //@{
    //! Type aliases
    using size_type   = celeritas::size_type;
    using result_type = demo_interactor::KNDemoResult;
    using constSPParticleParams
        = std::shared_ptr<const celeritas::ParticleParams>;
    using constSPPhysicsArrayParams
        = std::shared_ptr<const celeritas::PhysicsArrayParams>;
    //@}

  private:
    // CPU Version of the particle StatePointers
    struct StatePointers
    {
        celeritas::ParticleStatePointers particle;
        celeritas::Real3                 position;
        celeritas::Real3                 direction;
        celeritas::real_type             time;
        bool                             alive;
    };

  public:
    // Construct with parameters
    HostKNDemoRunner(constSPParticleParams     particles,
                     constSPPhysicsArrayParams xs);

    // Run given number of particles
    result_type operator()(demo_interactor::KNDemoRunArgs args);

  private:
    constSPParticleParams                     pparams_;
    constSPPhysicsArrayParams                 xsparams_;
    celeritas::KleinNishinaInteractorPointers kn_pointers_;
};

//---------------------------------------------------------------------------//
} // namespace demo_interactor