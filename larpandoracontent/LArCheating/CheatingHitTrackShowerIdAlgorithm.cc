/**
 *  @file   larpandoracontent/LArCheating/CheatingHitTrackShowerIdAlgorithm.cc
 *
 *  @brief  Implementation of cheated track shower id algorithm.
 *
 *  $Log: $
 */

#include "Pandora/AlgorithmHeaders.h"

#include "larpandoracontent/LArCheating/CheatingHitTrackShowerIdAlgorithm.h"

#include "larpandoracontent/LArHelpers/LArMCParticleHelper.h"
#include "larpandoracontent/LArObjects/LArCaloHit.h"

using namespace pandora;
using namespace lar_content;

namespace lar_content
{

CheatingHitTrackShowerIdAlgorithm::CheatingHitTrackShowerIdAlgorithm() :
{
}

//------------------------------------------------------------------------------------------------------------------------------------------

CheatingHitTrackShowerIdAlgorithm::~CheatingHitTrackShowerIdAlgorithm()
{
}

//------------------------------------------------------------------------------------------------------------------------------------------

StatusCode CheatingHitTrackShowerIdAlgorithm::Run()
{
    for (const std::string listName : m_caloHitListNames)
    {
        const CaloHitList *pCaloHitList(nullptr);
        PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::GetList(*this, listName, pCaloHitList));
        const MCParticleList *pMCParticleList(nullptr);
        PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::GetCurrentList(*this, pMCParticleList));

        const HitType view{pCaloHitList->front()->GetHitType()};

        if (!(view == TPC_VIEW_U || view == TPC_VIEW_V || view == TPC_VIEW_W))
            return STATUS_CODE_NOT_ALLOWED;

        LArMCParticleHelper::PrimaryParameters parameters;
        parameters.m_minHitsForGoodView = 0;
        parameters.m_maxPhotonPropagation = std::numeric_limits<float>::max();
        LArMCParticleHelper::MCContributionMap targetMCParticleToHitsMap;
        LArMCParticleHelper::SelectReconstructableMCParticles(
            pMCParticleList, pCaloHitList, parameters, LArMCParticleHelper::IsBeamNeutrinoFinalState, targetMCParticleToHitsMap);

        for (const CaloHit *pCaloHit : *pCaloHitList)
        {
            try
            {
                const MCParticle *const pMCParticle(MCParticleHelper::GetMainMCParticle(pCaloHit));
                // Throw away non-reconstructable hits
                if (targetMCParticleToHitsMap.find(pMCParticle) == targetMCParticleToHitsMap.end())
                    continue;
                if (LArMCParticleHelper::IsDescendentOf(pMCParticle, 2112))
                    continue;
                if (pCaloHit->GetInputEnergy() < 0.f)
                    continue;

                const int pdg{std::abs(pMCParticle->GetParticleId())};
                LArCaloHit *pLArCaloHit{const_cast<LArCaloHit *>(dynamic_cast<const LArCaloHit *>(pCaloHit))};
                if (pdg == 11 || pdg == 22) {
                    pLArCaloHit->SetShowerProbability(1.0);
                    pLArCaloHit->SetTrackProbability(0.0);
                } else {
                    pLArCaloHit->SetShowerProbability(0.0);
                    pLArCaloHit->SetTrackProbability(1.0);
                }
            }
            catch (const StatusCodeException &)
            {
                continue;
            }
        }
    }

}

//------------------------------------------------------------------------------------------------------------------------------------------

StatusCode CheatingHitTrackShowerIdAlgorithm::ReadSettings(const TiXmlHandle xmlHandle)
{
    PANDORA_RETURN_RESULT_IF_AND_IF(
        STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadVectorOfValues(xmlHandle, "CaloHitListNames", m_caloHitListNames));

    return STATUS_CODE_SUCCESS;
}

} // namespace lar_dl_content
