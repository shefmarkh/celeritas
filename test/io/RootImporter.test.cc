//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file RootImporter.test.cc
//---------------------------------------------------------------------------//
#include "io/RootImporter.hh"
#include "io/ImportTableType.hh"
#include "io/ImportProcessType.hh"
#include "io/ImportProcess.hh"
#include "io/ImportModel.hh"
#include "physics/base/ParticleMd.hh"
#include "base/Types.hh"
#include "base/Range.hh"

#include "celeritas_test.hh"

using celeritas::elem_id;
using celeritas::GdmlGeometryMap;
using celeritas::ImportMaterial;
using celeritas::ImportMaterialState;
using celeritas::ImportModel;
using celeritas::ImportParticle;
using celeritas::ImportPhysicsTable;
using celeritas::ImportPhysicsVectorType;
using celeritas::ImportProcess;
using celeritas::ImportProcessType;
using celeritas::ImportTableType;
using celeritas::ImportVolume;
using celeritas::mat_id;
using celeritas::ParticleDef;
using celeritas::ParticleDefId;
using celeritas::ParticleParams;
using celeritas::PDGNumber;
using celeritas::real_type;
using celeritas::RootImporter;
using celeritas::vol_id;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class RootImporterTest : public celeritas::Test
{
  protected:
    void SetUp() override
    {
        root_filename_ = this->test_data_path("io", "geant-exporter-data.root");
    }

    std::string root_filename_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
TEST_F(RootImporterTest, import_particles)
{
    RootImporter import(root_filename_.c_str());
    auto         data = import();

    EXPECT_EQ(19, data.particle_params->size());

    EXPECT_GE(data.particle_params->find(PDGNumber(11)).get(), 0);
    ParticleDefId electron_id = data.particle_params->find(PDGNumber(11));
    ParticleDef   electron    = data.particle_params->get(electron_id);

    EXPECT_SOFT_EQ(0.510998910, electron.mass.value());
    EXPECT_EQ(-1, electron.charge.value());
    EXPECT_EQ(0, electron.decay_constant);

    std::vector<std::string> loaded_names;
    std::vector<int>         loaded_pdgs;
    for (const auto& md : data.particle_params->md())
    {
        loaded_names.push_back(md.name);
        loaded_pdgs.push_back(md.pdg_code.get());
    }

    // clang-format off
    const std::string expected_loaded_names[] = {"gamma", "e+", "e-", "mu+",
        "mu-", "pi-", "pi+", "kaon-", "kaon+", "anti_proton", "proton",
        "anti_deuteron", "deuteron", "anti_He3", "He3", "anti_triton",
        "triton", "anti_alpha", "alpha"};
    const int expected_loaded_pdgs[] = {22, -11, 11, -13, 13, -211, 211, -321,
        321, -2212, 2212, -1000010020, 1000010020, -1000020030, 1000020030,
        -1000010030, 1000010030, -1000020040, 1000020040};
    // clang-format on

    EXPECT_VEC_EQ(expected_loaded_names, loaded_names);
    EXPECT_VEC_EQ(expected_loaded_pdgs, loaded_pdgs);
}

//---------------------------------------------------------------------------//
TEST_F(RootImporterTest, import_tables)
{
    RootImporter import(root_filename_.c_str());
    auto         data = import();

    EXPECT_GE(data.physics_tables->size(), 0);

    bool lambda_kn_gamma_table = false;

    for (auto table : *data.physics_tables)
    {
        EXPECT_GE(table.physics_vectors.size(), 0);

        if (table.particle == PDGNumber{celeritas::pdg::gamma()}
            && table.table_type == ImportTableType::lambda
            && table.process == ImportProcess::compton
            && table.model == ImportModel::klein_nishina)
        {
            lambda_kn_gamma_table = true;
            break;
        }
    }
    EXPECT_TRUE(lambda_kn_gamma_table);
}

//---------------------------------------------------------------------------//
TEST_F(RootImporterTest, import_geometry)
{
    RootImporter import(root_filename_.c_str());
    auto         data = import();

    auto map = data.geometry->volid_to_matid_map();
    EXPECT_EQ(map.size(), 4257);

    // Fetch a given ImportVolume provided a vol_id
    vol_id       volid  = 10;
    ImportVolume volume = data.geometry->get_volume(volid);
    EXPECT_EQ(volume.name, "TrackerPatchPannel");

    // Fetch respective mat_id and ImportMaterial from the given vol_id
    auto           matid    = data.geometry->get_matid(volid);
    ImportMaterial material = data.geometry->get_material(matid);

    // Material
    EXPECT_EQ(matid, 31);
    EXPECT_EQ(material.name, "Air");
    EXPECT_EQ(material.state, ImportMaterialState::gas);
    EXPECT_SOFT_EQ(material.temperature, 293.15);          // [K]
    EXPECT_SOFT_EQ(material.density, 0.00121399936124299); // [g/cm^3]
    EXPECT_SOFT_EQ(material.electron_density,
                   3.6523656201748414e+20); // [1/cm^3]
    EXPECT_SOFT_EQ(material.atomic_density, 5.0756589772243755e+19); // [1/cm^3]
    EXPECT_SOFT_EQ(material.radiation_length, 30152.065419629631);   // [cm]
    EXPECT_SOFT_EQ(material.nuclear_int_length, 70408.106699294673); // [cm]
    EXPECT_EQ(material.elements_fractions.size(), 4);

    // Elements within material
    std::string elements_name[4] = {"N", "O", "Ar", "H"};
    real_type   fraction[4]      = {0.7494, 0.2369, 0.0129, 0.0008};
    int         atomic_number[4] = {7, 8, 18, 1};
    // [AMU]
    real_type atomic_mass[4]
        = {14.00676896, 15.999390411, 39.94769335110001, 1.007940752665138};

    int i = 0;
    for (auto const& iter : material.elements_fractions)
    {
        auto elid    = iter.first;
        auto element = data.geometry->get_element(elid);

        EXPECT_EQ(element.name, elements_name[i]);
        EXPECT_EQ(element.atomic_number, atomic_number[i]);
        EXPECT_SOFT_EQ(element.atomic_mass, atomic_mass[i]);
        EXPECT_SOFT_EQ(iter.second, fraction[i]);
        i++;
    }
}
