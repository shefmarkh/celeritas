//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file RadialDistribution.i.hh
//---------------------------------------------------------------------------//

#include <cmath>
#include "base/Assert.hh"
#include "GenerateCanonical.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with defaults.
 */
template<typename T>
CELER_FUNCTION RadialDistribution<T>::RadialDistribution(real_type radius)
    : radius_(radius)
{
    REQUIRE(radius_ > 0);
}

//---------------------------------------------------------------------------//
/*!
 * Sample a random number according to the distribution.
 */
template<class T>
template<class Generator>
CELER_FUNCTION auto RadialDistribution<T>::operator()(Generator& rng)
    -> result_type
{
    constexpr auto one_third = static_cast<T>(1.0 / 3.0);
    return std::pow(generate_canonical<T>(rng), one_third) * radius_;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
