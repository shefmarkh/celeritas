//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file LivermoreInteractor.i.hh
//---------------------------------------------------------------------------//

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with shared and state data.
 */
CELER_FUNCTION LivermoreInteractor::LivermoreInteractor(
    const LivermoreInteractorPointers& shared,
    const ParticleTrackView&           particle,
    const Real3&                       inc_direction,
    SecondaryAllocatorView&            allocate)
    : shared_(shared)
    , inc_energy_(particle.energy().value())
    , inc_direction_(inc_direction)
    , allocate_(allocate)
{
    REQUIRE(inc_energy_ >= this->min_incident_energy()
            && inc_energy_ <= this->max_incident_energy());
    REQUIRE(particle.def_id() == shared_.gamma_id); // XXX
}

//---------------------------------------------------------------------------//
/*!
 * Sample using the XXX model.
 */
template<class Engine>
CELER_FUNCTION Interaction LivermoreInteractor::operator()(Engine& rng)
{
    // Allocate space for XXX (electron, multiple particles, ...)
    Secondary* secondaries = this->allocate_(0); // XXX
    if (secondaries == nullptr)
    {
        // Failed to allocate space for a secondary
        return Interaction::from_failure();
    }

    // XXX sample
    (void)sizeof(rng);

    // Construct interaction for change to primary (incident) particle
    Interaction result;
    result.action      = Action::scattered;                     // XXX
    result.energy      = units::MevEnergy{inc_energy_.value()}; // XXX
    result.direction   = inc_direction_;
    result.secondaries = {secondaries, 1}; // XXX

    // Save outgoing secondary data
    secondaries[0].def_id    = shared_.electron_id; // XXX
    secondaries[0].energy    = units::MevEnergy{0}; // XXX
    secondaries[0].direction = {0, 0, 0};           // XXX

    return result;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
