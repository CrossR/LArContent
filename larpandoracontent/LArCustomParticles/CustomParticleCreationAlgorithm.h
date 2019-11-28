/**
 *  @file   larpandoracontent/LArCustomParticles/CustomParticleCreationAlgorithm.h
 *
 *  @brief  Header file for the 3D particle creation algorithm class.
 *
 *  $Log: $
 */
#ifndef LAR_CUSTOM_PARTICLE_CREATION_ALGORITHM_H
#define LAR_CUSTOM_PARTICLE_CREATION_ALGORITHM_H 1

#include "Pandora/Algorithm.h"

#include "larpandoracontent/LArHelpers/LArPfoHelper.h"

namespace lar_content
{

/**
 *  @brief  CustomParticleCreationAlgorithm class
 */
class CustomParticleCreationAlgorithm : public pandora::Algorithm
{
protected:
    virtual pandora::StatusCode Run();
    virtual pandora::StatusCode ReadSettings(const pandora::TiXmlHandle xmlHandle);

    /**
     *  @brief  Create specialised Pfo from an generic input Pfo
     *
     *  @param  pInputPfo the address of the input Pfo
     *  @param  pOutputPfo the address of the output Pfo
     *  @parm   metricStruct metrics to be populated by the particle creation algorithm.
     */
    virtual void CreatePfo(const pandora::ParticleFlowObject *const pInputPfo, const pandora::ParticleFlowObject*& pOutputPfo,
            threeDMetric &metricStruct) const = 0;


private:
    std::string  m_pfoListName;        ///< The name of the input pfo list
    std::string  m_vertexListName;     ///< The name of the input vertex list
    std::string  m_mcParticleListName; ///< The name of the MC particle list
    bool         m_runMetrics = false; ///< If metric generation should be ran.
};

} // namespace lar_content

#endif // #ifndef LAR_CUSTOM_PARTICLE_CREATION_ALGORITHM_H
