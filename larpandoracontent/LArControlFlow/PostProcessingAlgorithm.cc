/**
 *  @file   larpandoracontent/LArControlFlow/PostProcessingAlgorithm.cc
 *
 *  @brief  Implementation of the list moving algorithm class.
 *
 *  $Log: $
 */

#include "Pandora/AlgorithmHeaders.h"

#include "larpandoracontent/LArControlFlow/PostProcessingAlgorithm.h"

// TODO: Debugging, remove
#include "larpandoracontent/LArHelpers/LArMCParticleHelper.h"
#include "Helpers/MCParticleHelper.h"
#include "Objects/MCParticle.h"
#include <fstream>

using namespace pandora;

namespace lar_content
{

PostProcessingAlgorithm::PostProcessingAlgorithm() :
    m_listCounter(0)
{
}

//------------------------------------------------------------------------------------------------------------------------------------------

StatusCode PostProcessingAlgorithm::Reset()
{
    m_listCounter = 0;
    return STATUS_CODE_SUCCESS;
}

//------------------------------------------------------------------------------------------------------------------------------------------

StatusCode PostProcessingAlgorithm::Run()
{
    for (const std::string &listName : m_pfoListNames)
        PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, this->RenameList<PfoList>(listName));

    this->DumpClusterList("ClustersU", "End");
    this->DumpClusterList("ClustersV", "End");
    this->DumpClusterList("ClustersW", "End");

    for (const std::string &listName : m_clusterListNames)
        PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, this->RenameList<ClusterList>(listName));

    for (const std::string &listName : m_vertexListNames)
        PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, this->RenameList<VertexList>(listName));

    for (const std::string &listName : m_caloHitListNames)
        PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, this->RenameList<CaloHitList>(listName));

    PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::DropCurrentList<CaloHit>(*this));
    PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::DropCurrentList<Track>(*this));
    PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::DropCurrentList<MCParticle>(*this));
    PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::DropCurrentList<Cluster>(*this));
    PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::DropCurrentList<ParticleFlowObject>(*this));
    PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::DropCurrentList<Vertex>(*this));

    if (!m_currentPfoListReplacement.empty())
    {
        const std::string replacementListName(m_currentPfoListReplacement + TypeToString(m_listCounter));

        if (STATUS_CODE_SUCCESS != PandoraContentApi::ReplaceCurrentList<Pfo>(*this, replacementListName))
        {
            if (PandoraContentApi::GetSettings(*this)->ShouldDisplayAlgorithmInfo())
                std::cout << "PostProcessingAlgorithm: could not replace current pfo list with list named: " << replacementListName << std::endl;

            PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::DropCurrentList<Pfo>(*this));
        }
    }

    ++m_listCounter;
    return STATUS_CODE_SUCCESS;
}

//------------------------------------------------------------------------------------------------------------------------------------------

void PostProcessingAlgorithm::DumpClusterList(const std::string &clusterListName, const std::string &recoStatus) const
{
    // Find a file name by just picking a file name
    // until an unused one is found.
    std::string fileName;
    int fileNum = 0;

    while (true)
    {
        fileName = "/home/scratch/showerClusters/clusters_" +
            clusterListName + "_" + recoStatus + "_" +
            std::to_string(fileNum);
        std::ifstream testFile(fileName + ".csv");

        if (!testFile.good())
            break;

        testFile.close();
        ++fileNum;
    }

    std::ofstream csvFile;
    csvFile.open(fileName + ".csv");

    const ClusterList *pClusterList = nullptr;

    try {
        PANDORA_THROW_RESULT_IF_AND_IF(
            STATUS_CODE_SUCCESS, STATUS_CODE_NOT_INITIALIZED, !=,
            PandoraContentApi::GetList(*this, clusterListName, pClusterList)
        );
    } catch (StatusCodeException e) {
        std::cout << "Failed to get cluster list: " << e.ToString() << std::endl;
        csvFile.close();
        return;
    }

    // Sanity checks: General cluster size should go up.
    //                Number of clusters should go down.
    //                Purity should remain same-ish
    //                Completeness should go up.

    const MCParticleList *pMCParticleList = nullptr;

    try {
        PANDORA_THROW_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::GetList(*this, "Input", pMCParticleList));
    } catch (StatusCodeException e) {
        std::cout << "Failed to get MCParticleList: " << e.ToString() << std::endl;
        csvFile.close();
        return;
    }

    // MC Setup.
    // Build up Calo Hit -> Map and MC -> CaloHitList.
    LArMCParticleHelper::MCRelationMap mcToTargetMCMap;
    LArMCParticleHelper::GetMCToSelfMap(pMCParticleList, mcToTargetMCMap);

    LArMCParticleHelper::CaloHitToMCMap eventLevelCaloHitToMCMap;
    LArMCParticleHelper::MCContributionMap eventLevelMCToCaloHitMap;

    // For each cluster, build up the two maps (CaloHit -> MC, MC -> CaloHitList).
    // To do this, we need to get all the calo hits from the cluster, then
    // call the helper.
    //
    // Once we've done that add the CalotHit -> MC bits, and merge the
    // MC -> CaloHitList stuff.
    //
    // This is because a hit is to 1 MC particle, whereas the MC to CaloHitList
    // is to a list, and that list is incomplete (only contains hits from the
    // current cluster).
    for (auto const &cluster : *pClusterList) {

        LArMCParticleHelper::CaloHitToMCMap perClusterCaloHitToMCMap;
        LArMCParticleHelper::MCContributionMap perClusterMCToCaloHitMap;

        CaloHitList caloHits;
        for (auto const &clusterHitPair : cluster->GetOrderedCaloHitList()) {
            CaloHitList hitsForCluster(*clusterHitPair.second);
            caloHits.merge(hitsForCluster);
        }
        CaloHitList isolatedHits = cluster->GetIsolatedCaloHitList();
        caloHits.merge(isolatedHits);

        try {
            LArMCParticleHelper::GetMCParticleToCaloHitMatches(
                &(caloHits), mcToTargetMCMap, perClusterCaloHitToMCMap, perClusterMCToCaloHitMap
            );
        } catch (StatusCodeException e) {
            std::cout << "Failed to get matches: " << e.ToString() << std::endl;
        }

        eventLevelCaloHitToMCMap.insert(perClusterCaloHitToMCMap.begin(), perClusterCaloHitToMCMap.end());

        // Merge in the MC -> CaloHitList stuff if needed, otherwise just add all.
        for (auto &mcCaloHitListPair : perClusterMCToCaloHitMap) {
            const auto it = eventLevelMCToCaloHitMap.find(mcCaloHitListPair.first);

            if (it == eventLevelMCToCaloHitMap.end()) {
                eventLevelMCToCaloHitMap.insert(mcCaloHitListPair);
            } else {
                it->second.merge(mcCaloHitListPair.second);
                it->second.unique();
            }
        }
    }

    // ROOT TTree variable setup
    int clusterNumber = -1;
    float completeness = 0.0;
    float purity = 0.0;
    float failedHits = 0.0;
    float matchesMain = 0.0;
    float len = 0.0;
    int mcID = 0;

    const std::string treeName = "showerClustersTree";
    PANDORA_MONITORING_API(Create(this->GetPandora()));

    for (auto const &cluster : *pClusterList) {

        ++clusterNumber;
        completeness = 0.0;
        purity = 0.0;
        len = 0.0;
        matchesMain = 0.0;
        failedHits = 0.0;
        mcID = 0;

        csvFile << "X, Z, Type, PID, IsAvailable, IsShower, MCUid, IsIsolated" << std::endl;

        const MCParticle *pMCParticle = nullptr;
        Uid mcUid = nullptr;
        float hitsInMC = 0.0;

        try {
            pMCParticle = MCParticleHelper::GetMainMCParticle(cluster);
            mcUid = pMCParticle->GetUid();
            mcID = pMCParticle->GetParticleId();
        } catch (const StatusCodeException &) {
            // TODO: Attach debugger and check why!
            int c_size = cluster->GetOrderedCaloHitList().size() + cluster->GetIsolatedCaloHitList().size();
            std::cout << "  ## No MC. Size " << c_size << std::endl;
            mcUid = (const void*) -999;
            mcID = -999;
        }

        auto mcToCaloHit = eventLevelMCToCaloHitMap.find(pMCParticle);
        if (mcToCaloHit != eventLevelMCToCaloHitMap.end()) {
            hitsInMC = mcToCaloHit->second.size();
        } else {
            std::cout << " >> Failed to find MC in MC -> Calo map!" << std::endl;
            hitsInMC = -999;
        }

        const int isShower = std::abs(cluster->GetParticleId()) == MU_MINUS ? 0 : 1;

        for (auto const &clusterHitPair : cluster->GetOrderedCaloHitList()) {
            for (auto const &caloHit : *(clusterHitPair.second)) {
                ++len;

                const CartesianVector pos = caloHit->GetPositionVector();
                csvFile << pos.GetX() << ", "
                        << pos.GetZ() << ", "
                        << recoStatus << ", "
                        << cluster->GetParticleId() << ", "
                        << cluster->IsAvailable() << ", "
                        << isShower << ", "
                        << mcUid << ", "
                        << "0" << std::endl;

                const auto it2 = eventLevelCaloHitToMCMap.find(caloHit);

                if (it2 == eventLevelCaloHitToMCMap.end()) {
                   ++failedHits;
                   continue;
                }

                const auto mc = it2->second;

                if (mc == pMCParticle) {
                    ++matchesMain;
                }
            }
        }

        for (auto const &caloHit : cluster->GetIsolatedCaloHitList()) {
            ++len;

            const CartesianVector pos = caloHit->GetPositionVector();
            csvFile << pos.GetX() << ", "
                    << pos.GetZ() << ", "
                    << recoStatus << ", "
                    << cluster->GetParticleId() << ", "
                    << cluster->IsAvailable() << ", "
                    << isShower << ", "
                    << mcUid << ", "
                    << "1" << std::endl;

            const auto it2 = eventLevelCaloHitToMCMap.find(caloHit);

            if (it2 == eventLevelCaloHitToMCMap.end()) {
               ++failedHits;
               continue;
            }

            const auto mc = it2->second;

            if (mc == pMCParticle) {
                ++matchesMain;
            }
        }

        if (cluster->GetIsolatedCaloHitList().size() != 0)
            std::cout << " #### Isolated hits in this event! ####" << std::endl;

        completeness = matchesMain / hitsInMC;
        purity = matchesMain / len;

        PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), treeName, "clusterNumber", clusterNumber));
        PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), treeName, "completeness", completeness));
        PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), treeName, "purity", purity));
        PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), treeName, "numberOfHits", len));
        PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), treeName, "failedHits", failedHits));
        PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), treeName, "mcID", mcID));
        PANDORA_MONITORING_API(FillTree(this->GetPandora(), treeName));
    }

    PANDORA_MONITORING_API(SaveTree(this->GetPandora(), treeName, fileName + ".root", "RECREATE"));
    // TODO: Maybe something higher level too?
    // Right now, we have "This cluster is X% complete and X% pure".
    // Flipping it to "This MC Particle is spread across X clusters, with X purity" could also be nice.
    PANDORA_MONITORING_API(Delete(this->GetPandora()));
    csvFile.close();
    return;
}

//------------------------------------------------------------------------------------------------------------------------------------------

template <typename T>
StatusCode PostProcessingAlgorithm::RenameList(const std::string &oldListName) const
{
    const std::string newListName(oldListName + TypeToString(m_listCounter));
    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, PandoraContentApi::RenameList<T>(*this, oldListName, newListName));
    return STATUS_CODE_SUCCESS;
}

//------------------------------------------------------------------------------------------------------------------------------------------

StatusCode PostProcessingAlgorithm::ReadSettings(const TiXmlHandle xmlHandle)
{
    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadVectorOfValues(xmlHandle,
        "PfoListNames", m_pfoListNames));

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadVectorOfValues(xmlHandle,
        "ClusterListNames", m_clusterListNames));

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadVectorOfValues(xmlHandle,
        "VertexListNames", m_vertexListNames));

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadVectorOfValues(xmlHandle,
        "CaloHitListNames", m_caloHitListNames));

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle,
        "CurrentPfoListReplacement", m_currentPfoListReplacement));

    return STATUS_CODE_SUCCESS;
}


//------------------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------------------

template pandora::StatusCode PostProcessingAlgorithm::RenameList<PfoList>(const std::string &) const;
template pandora::StatusCode PostProcessingAlgorithm::RenameList<ClusterList>(const std::string &) const;
template pandora::StatusCode PostProcessingAlgorithm::RenameList<VertexList>(const std::string &) const;
template pandora::StatusCode PostProcessingAlgorithm::RenameList<CaloHitList>(const std::string &) const;

} // namespace lar_content
