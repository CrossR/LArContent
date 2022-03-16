/**
 *  @file   larpandoracontent/LArObjects/LArMCParticle.h
 *
 *  @brief  Header file for the lar mc particle class.
 *
 *  $Log: $
 */
#ifndef LAR_MC_PARTICLE_H
#define LAR_MC_PARTICLE_H 1

#include "Objects/MCParticle.h"

#include "Pandora/ObjectCreation.h"
#include "Pandora/PandoraObjectFactories.h"

#include "Persistency/BinaryFileReader.h"
#include "Persistency/BinaryFileWriter.h"
#include "Persistency/XmlFileReader.h"
#include "Persistency/XmlFileWriter.h"

namespace lar_content
{

// Enumeration maps onto G4 process IDs from QGSP_BERT and EM standard physics lists, plus an ID for the incident neutrino
enum MCProcess
{
    MC_PROC_INCIDENT_NU = -1,
    MC_PROC_UNKNOWN,
    MC_PROC_PRIMARY,
    MC_PROC_COMPT,
    MC_PROC_PHOT,
    MC_PROC_ANNIHIL,
    MC_PROC_E_IONI,
    MC_PROC_E_BREM,
    MC_PROC_CONV,
    MC_PROC_MU_IONI,
    MC_PROC_MU_MINUS_CAPTURE_AT_REST,
    MC_PROC_NEUTRON_INELASTIC,
    MC_PROC_N_CAPTURE,
    MC_PROC_HAD_ELASTIC,
    MC_PROC_DECAY,
    MC_PROC_COULOMB_SCAT,
    MC_PROC_MU_BREM,
    MC_PROC_MU_PAIR_PROD,
    MC_PROC_PHOTON_INELASTIC,
    MC_PROC_HAD_IONI,
    MC_PROC_PROTON_INELASTIC,
    MC_PROC_PI_PLUS_INELASTIC,
    MC_PROC_CHIPS_NUCLEAR_CAPTURE_AT_REST,
    MC_PROC_PI_MINUS_INELASTIC,
    MC_PROC_TRANSPORTATION,
    MC_PROC_RAYLEIGH,
    MC_PROC_HAD_BREM,
    MC_PROC_HAD_PAIR_PROD,
    MC_PROC_ION_IONI,
    MC_PROC_NEUTRON_KILLER,
    MC_PROC_ION_INELASTIC,
    MC_PROC_HE3_INELASTIC,
    MC_PROC_ALPHA_INELASTIC,
    MC_PROC_ANTI_HE3_INELASTIC,
    MC_PROC_ANTI_ALPHA_INELASTIC,
    MC_PROC_HAD_FRITIOF_CAPTURE_AT_REST,
    MC_PROC_ANTI_DEUTERON_INELASTIC,
    MC_PROC_ANTI_NEUTRON_INELASTIC,
    MC_PROC_ANTI_PROTON_INELASTIC,
    MC_PROC_ANTI_TRITON_INELASTIC,
    MC_PROC_DEUTERON_INELASTIC,
    MC_PROC_ELECTRON_NUCLEAR,
    MC_PROC_PHOTON_NUCLEAR,
    MC_PROC_KAON_PLUS_INELASTIC,
    MC_PROC_KAON_MINUS_INELASTIC,
    MC_PROC_HAD_BERTINI_CAPTURE_AT_REST,
    MC_PROC_LAMBDA_INELASTIC,
    MC_PROC_MU_NUCLEAR,
    MC_PROC_TRITON_INELASTIC,
    MC_PROC_PRIMARY_BACKGROUND
};

/**
 *  @brief  LAr mc particle parameters
 */
class LArMCParticleParameters : public object_creation::MCParticle::Parameters
{
public:
    pandora::InputInt m_nuanceCode; ///< The nuance code
    pandora::InputInt m_process;    ///< The process creating the particle
    std::vector<pandora::InputCartesianVector> m_mcStepPositions;   ///< The positions of the geant4 steps
    std::vector<pandora::InputCartesianVector> m_mcStepMomentas;    ///< The momenta of the geant4 steps
};

//------------------------------------------------------------------------------------------------------------------------------------------

/**
 *  @brief  LAr mc particle class
 */
class LArMCParticle : public object_creation::MCParticle::Object
{
public:
    /**
     *  @brief  Constructor
     *
     *  @param  parameters the lar mc particle parameters
     */
    LArMCParticle(const LArMCParticleParameters &parameters);

    /**
     *  @brief  Get the nuance code
     *
     *  @return the nuance code
     */
    int GetNuanceCode() const;

    /**
     *  @brief  Get the position of the geant4 steps
     *
     *  @return vector of the positions
     */
    std::vector<pandora::CartesianVector> GetMCStepPositions() const;

    /**
     *  @brief  Get the momenta of the geant4 steps
     *
     *  @return vector of the momenta
     */
    std::vector<pandora::CartesianVector> GetMCStepMomentas() const;

    /**
     *  @brief  Fill the parameters associated with this MC particle
     *
     *  @param  parameters the output parameters
     */
    void FillParameters(LArMCParticleParameters &parameters) const;

    /**
     *  @brief  Get the process
     *
     *  @return the process
     */
    MCProcess GetProcess() const;

private:
    int m_nuanceCode; ///< The nuance code
    int m_process;    ///< The process that created the particle
    std::vector<pandora::CartesianVector> m_mcStepPositions;  ///< The positions of the geant4 steps
    std::vector<pandora::CartesianVector> m_mcStepMomentas;   ///< The momenta of the geant4 steps
};

//------------------------------------------------------------------------------------------------------------------------------------------

/**
 *  @brief  LArMCParticleFactory responsible for object creation
 */
class LArMCParticleFactory : public pandora::ObjectFactory<object_creation::MCParticle::Parameters, object_creation::MCParticle::Object>
{
public:
    /**
     *  @brief  Constructor
     *
     *  @param  version the LArMCParticle version
     */
    LArMCParticleFactory(const unsigned int version = 2, const bool readPositions = false, const bool writePositions = false);

    /**
     *  @brief  Create new parameters instance on the heap (memory-management to be controlled by user)
     *
     *  @return the address of the new parameters instance
     */
    Parameters *NewParameters() const;

    /**
     *  @brief  Read any additional (derived class only) object parameters from file using the specified file reader
     *
     *  @param  parameters the parameters to pass in constructor
     *  @param  fileReader the file reader, used to extract any additional parameters from file
     */
    pandora::StatusCode Read(Parameters &parameters, pandora::FileReader &fileReader) const;

    /**
     *  @brief  Persist any additional (derived class only) object parameters using the specified file writer
     *
     *  @param  pObject the address of the object to persist
     *  @param  fileWriter the file writer
     */
    pandora::StatusCode Write(const Object *const pObject, pandora::FileWriter &fileWriter) const;

    /**
     *  @brief  Create an object with the given parameters
     *
     *  @param  parameters the parameters to pass in constructor
     *  @param  pObject to receive the address of the object created
     */
    pandora::StatusCode Create(const Parameters &parameters, const Object *&pObject) const;

private:
    unsigned int m_version; ///< The LArMCParticle version
    bool m_writePositions;  ///< Write MC Particle Positions and Momentas
    bool m_readPositions;   ///< Read MC Particle Positions and Momentas
};

//------------------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------------------

inline LArMCParticle::LArMCParticle(const LArMCParticleParameters &parameters) :
    object_creation::MCParticle::Object(parameters),
    m_nuanceCode(parameters.m_nuanceCode.Get()),
    m_process(parameters.m_process.Get())
{
    for (auto const &mcStepPosition : parameters.m_mcStepPositions)
        m_mcStepPositions.emplace_back(mcStepPosition.Get());

    for (auto const &mcStepMomenta : parameters.m_mcStepMomentas)
        m_mcStepMomentas.emplace_back(mcStepMomenta.Get());
}

//------------------------------------------------------------------------------------------------------------------------------------------

inline int LArMCParticle::GetNuanceCode() const
{
    return m_nuanceCode;
}

//------------------------------------------------------------------------------------------------------------------------------------------

inline std::vector<pandora::CartesianVector> LArMCParticle::GetMCStepPositions() const
{
    return m_mcStepPositions;
}

inline void LArMCParticle::FillParameters(LArMCParticleParameters &parameters) const
{
    parameters.m_nuanceCode = this->GetNuanceCode();
    parameters.m_process = this->GetProcess();
    parameters.m_energy = this->GetEnergy();
    parameters.m_momentum = this->GetMomentum();
    parameters.m_vertex = this->GetVertex();
    parameters.m_endpoint = this->GetEndpoint();
    parameters.m_particleId = this->GetParticleId();
    parameters.m_mcParticleType = this->GetMCParticleType();
    // ATTN Set the parent address to the original owner of the mc particle
    parameters.m_pParentAddress = static_cast<const void *>(this);
}

//------------------------------------------------------------------------------------------------------------------------------------------

inline std::vector<pandora::CartesianVector> LArMCParticle::GetMCStepMomentas() const
{
    return m_mcStepMomentas;
}

//------------------------------------------------------------------------------------------------------------------------------------------

inline MCProcess LArMCParticle::GetProcess() const
{
    return MCProcess(m_process);
}

//------------------------------------------------------------------------------------------------------------------------------------------

inline LArMCParticleFactory::LArMCParticleFactory(const unsigned int version, const bool readPositions, const bool writePositions) : 
    m_version(version),
    m_readPositions(readPositions),
    m_writePositions(writePositions)
{
}

//------------------------------------------------------------------------------------------------------------------------------------------

inline LArMCParticleFactory::Parameters *LArMCParticleFactory::NewParameters() const
{
    return (new LArMCParticleParameters);
}

//------------------------------------------------------------------------------------------------------------------------------------------

inline pandora::StatusCode LArMCParticleFactory::Create(const Parameters &parameters, const Object *&pObject) const
{
    const LArMCParticleParameters &larMCParticleParameters(dynamic_cast<const LArMCParticleParameters &>(parameters));
    pObject = new LArMCParticle(larMCParticleParameters);

    return pandora::STATUS_CODE_SUCCESS;
}

//------------------------------------------------------------------------------------------------------------------------------------------

inline pandora::StatusCode LArMCParticleFactory::Read(Parameters &parameters, pandora::FileReader &fileReader) const
{
    // ATTN: To receive this call-back must have already set file reader mc particle factory to this factory
    int nuanceCode(0);
    int process(0);

    unsigned int nMCStepPositions(0);
    std::vector<pandora::InputCartesianVector> mcStepPositions;

    unsigned int nMCStepMomentas(0);
    std::vector<pandora::InputCartesianVector> mcStepMomentas;

    if (pandora::BINARY == fileReader.GetFileType())
    {
        pandora::BinaryFileReader &binaryFileReader(dynamic_cast<pandora::BinaryFileReader &>(fileReader));
        PANDORA_RETURN_RESULT_IF(pandora::STATUS_CODE_SUCCESS, !=, binaryFileReader.ReadVariable(nuanceCode));

        if (m_version > 1)
            PANDORA_RETURN_RESULT_IF(pandora::STATUS_CODE_SUCCESS, !=, binaryFileReader.ReadVariable(process));

        if (m_readPositions)
        {
            PANDORA_RETURN_RESULT_IF(pandora::STATUS_CODE_SUCCESS, !=, binaryFileReader.ReadVariable(nMCStepPositions));
            for (unsigned int step = 0; step < nMCStepPositions; ++step)
                {
                    pandora::CartesianVector mcStepPosition(0.0, 0.0, 0.0);
                    PANDORA_RETURN_RESULT_IF(pandora::STATUS_CODE_SUCCESS, !=, binaryFileReader.ReadVariable(mcStepPosition));
                    mcStepPositions.emplace_back(mcStepPosition);
                }

            PANDORA_RETURN_RESULT_IF(pandora::STATUS_CODE_SUCCESS, !=, binaryFileReader.ReadVariable(nMCStepMomentas));
            for (unsigned int step = 0; step < nMCStepMomentas; ++step)
                {
                    pandora::CartesianVector mcStepMomenta(0.0, 0.0, 0.0);
                    PANDORA_RETURN_RESULT_IF(pandora::STATUS_CODE_SUCCESS, !=, binaryFileReader.ReadVariable(mcStepMomenta));
                    mcStepMomentas.emplace_back(mcStepMomenta);
                }
        }
    }
    else if (pandora::XML == fileReader.GetFileType())
    {
        pandora::XmlFileReader &xmlFileReader(dynamic_cast<pandora::XmlFileReader &>(fileReader));
        PANDORA_RETURN_RESULT_IF(pandora::STATUS_CODE_SUCCESS, !=, xmlFileReader.ReadVariable("NuanceCode", nuanceCode));

        if (m_version > 1)
            PANDORA_RETURN_RESULT_IF(pandora::STATUS_CODE_SUCCESS, !=, xmlFileReader.ReadVariable("Process", process));

        if (m_readPositions)
        {
            PANDORA_RETURN_RESULT_IF(pandora::STATUS_CODE_SUCCESS, !=, xmlFileReader.ReadVariable("NumberOfMCStepPositions", nMCStepPositions));
            for (unsigned int step = 0; step < nMCStepPositions; ++step)
                {
                    pandora::CartesianVector mcStepPosition(0.0, 0.0, 0.0);
                    std::stringstream mcStepPositionName;
                    mcStepPositionName << "MCStepPosition" << step;
                    PANDORA_RETURN_RESULT_IF(pandora::STATUS_CODE_SUCCESS, !=, xmlFileReader.ReadVariable(mcStepPositionName.str(), mcStepPosition));
                    mcStepPositions.emplace_back(mcStepPosition);
                }

            PANDORA_RETURN_RESULT_IF(pandora::STATUS_CODE_SUCCESS, !=, xmlFileReader.ReadVariable("NumberOfMCStepMomentas", nMCStepMomentas));
            for (unsigned int step = 0; step < nMCStepMomentas; ++step)
                {
                    pandora::CartesianVector mcStepMomenta(0.0, 0.0, 0.0);
                    std::stringstream mcStepMomentaName;
                    mcStepMomentaName << "MCStepMomenta" << step;
                    PANDORA_RETURN_RESULT_IF(pandora::STATUS_CODE_SUCCESS, !=, xmlFileReader.ReadVariable(mcStepMomentaName.str(), mcStepMomenta));
                    mcStepMomentas.emplace_back(mcStepMomenta);
                }
        }
    }
    else
    {
        return pandora::STATUS_CODE_INVALID_PARAMETER;
    }

    LArMCParticleParameters &larMCParticleParameters(dynamic_cast<LArMCParticleParameters &>(parameters));
    larMCParticleParameters.m_nuanceCode = nuanceCode;
    larMCParticleParameters.m_mcStepPositions = mcStepPositions;
    larMCParticleParameters.m_mcStepMomentas = mcStepMomentas;
    larMCParticleParameters.m_process = process;

    return pandora::STATUS_CODE_SUCCESS;
}

//------------------------------------------------------------------------------------------------------------------------------------------

inline pandora::StatusCode LArMCParticleFactory::Write(const Object *const pObject, pandora::FileWriter &fileWriter) const
{
    // ATTN: To receive this call-back must have already set file writer mc particle factory to this factory
    const LArMCParticle *const pLArMCParticle(dynamic_cast<const LArMCParticle *>(pObject));

    if (!pLArMCParticle)
        return pandora::STATUS_CODE_INVALID_PARAMETER;

    const std::vector<pandora::CartesianVector> &mcStepPositions(pLArMCParticle->GetMCStepPositions());
    const int nMCStepPositions(mcStepPositions.size());

    const std::vector<pandora::CartesianVector> &mcStepMomentas(pLArMCParticle->GetMCStepMomentas());
    const int nMCStepMomentas(mcStepMomentas.size());

    if (pandora::BINARY == fileWriter.GetFileType())
    {
        pandora::BinaryFileWriter &binaryFileWriter(dynamic_cast<pandora::BinaryFileWriter &>(fileWriter));
        PANDORA_RETURN_RESULT_IF(pandora::STATUS_CODE_SUCCESS, !=, binaryFileWriter.WriteVariable(pLArMCParticle->GetNuanceCode()));

        if (m_version > 1)
            PANDORA_RETURN_RESULT_IF(
                pandora::STATUS_CODE_SUCCESS, !=, binaryFileWriter.WriteVariable(static_cast<int>(pLArMCParticle->GetProcess())));

        if (m_writePositions)
        {
            PANDORA_RETURN_RESULT_IF(pandora::STATUS_CODE_SUCCESS, !=, binaryFileWriter.WriteVariable(nMCStepPositions));
            for (auto const &mcStepPosition : mcStepPositions)
            PANDORA_RETURN_RESULT_IF(pandora::STATUS_CODE_SUCCESS, !=, binaryFileWriter.WriteVariable(mcStepPosition));

            PANDORA_RETURN_RESULT_IF(pandora::STATUS_CODE_SUCCESS, !=, binaryFileWriter.WriteVariable(nMCStepMomentas));
            for (auto const &mcStepMomenta : mcStepMomentas)
            PANDORA_RETURN_RESULT_IF(pandora::STATUS_CODE_SUCCESS, !=, binaryFileWriter.WriteVariable(mcStepMomenta));
        }

    }
    else if (pandora::XML == fileWriter.GetFileType())
    {
        pandora::XmlFileWriter &xmlFileWriter(dynamic_cast<pandora::XmlFileWriter &>(fileWriter));
        PANDORA_RETURN_RESULT_IF(pandora::STATUS_CODE_SUCCESS, !=, xmlFileWriter.WriteVariable("NuanceCode", pLArMCParticle->GetNuanceCode()));

        if (m_version > 1)
            PANDORA_RETURN_RESULT_IF(
                pandora::STATUS_CODE_SUCCESS, !=, xmlFileWriter.WriteVariable("Process", static_cast<int>(pLArMCParticle->GetProcess())));

        if (m_writePositions)
        {
            PANDORA_RETURN_RESULT_IF(pandora::STATUS_CODE_SUCCESS, !=, xmlFileWriter.WriteVariable("NumberOfMCStepPositions", nMCStepPositions));
            for (int step = 0; step < nMCStepPositions; ++step)
                {
                    const pandora::CartesianVector &position(mcStepPositions[step]);
                    std::stringstream mcStepPositionName;
                    mcStepPositionName << "MCStepPosition" << step;
                    PANDORA_RETURN_RESULT_IF(pandora::STATUS_CODE_SUCCESS, !=, xmlFileWriter.WriteVariable(mcStepPositionName.str(), position));
                }

            PANDORA_RETURN_RESULT_IF(pandora::STATUS_CODE_SUCCESS, !=, xmlFileWriter.WriteVariable("NumberOfMCStepMomentas", nMCStepMomentas));
            for (int step = 0; step < nMCStepMomentas; ++step)
                {
                    const pandora::CartesianVector &momenta(mcStepMomentas[step]);
                    std::stringstream mcStepMomentaName;
                    mcStepMomentaName << "MCStepMomenta" << step;
                    PANDORA_RETURN_RESULT_IF(pandora::STATUS_CODE_SUCCESS, !=, xmlFileWriter.WriteVariable(mcStepMomentaName.str(), momenta));
                }
        }
    }
    else
    {
        return pandora::STATUS_CODE_INVALID_PARAMETER;
    }

    return pandora::STATUS_CODE_SUCCESS;
}

} // namespace lar_content

#endif // #ifndef LAR_MC_PARTICLE_H
