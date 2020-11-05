/**
 *  @file   larpandoracontent/LArCheating/CheatingShowerGrowingAlgorithm.cc
 *
 *  @brief  Implementation file for the cheating shower growing algorithm.
 *
 *  $Log: $
 */

#include "Pandora/AlgorithmHeaders.h"

#include "larpandoracontent/LArCheating/CheatingShowerGrowingAlgorithm.h"
#include "larpandoracontent/LArTrackShowerId/ShowerGrowingAlgorithm.h"


using namespace pandora;

namespace lar_content
{

CheatingShowerGrowingAlgorithm::CheatingShowerGrowingAlgorithm() :
    m_maxClusterFraction(0.25f)
{
}

//------------------------------------------------------------------------------------------------------------------------------------------

StatusCode CheatingShowerGrowingAlgorithm::Run()
{
    return STATUS_CODE_SUCCESS;
}

//------------------------------------------------------------------------------------------------------------------------------------------

ShowerGrowingAlgorithm::AssociationType CheatingShowerGrowingAlgorithm::AreClustersAssociated(const Cluster *const /*pClusterSeed*/, const Cluster *const /*pCluster*/) const
{
   return STRONG; 
}

//------------------------------------------------------------------------------------------------------------------------------------------

StatusCode CheatingShowerGrowingAlgorithm::ReadSettings(const TiXmlHandle xmlHandle)
{
    PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, XmlHelper::ReadVectorOfValues(xmlHandle, "InputClusterListNames", m_inputClusterListNames));

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle,
        "MCParticleListName", m_mcParticleListName));

    return STATUS_CODE_SUCCESS;
}

} // namespace lar_content
