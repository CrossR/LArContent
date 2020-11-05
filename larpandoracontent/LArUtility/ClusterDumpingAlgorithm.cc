/**
 *  @file   larpandoracontent/LArUtility/ClusterDumpingAlgorithm.cc
 *
 *  @brief  Implementation for the debug cluster dumping algorithm class.
 *
 *  $Log: $
 */

#include "Pandora/AlgorithmHeaders.h"
#include "Helpers/MCParticleHelper.h"
#include "Objects/MCParticle.h"

#include "larpandoracontent/LArUtility/ClusterDumpingAlgorithm.h"

#include "larpandoracontent/LArHelpers/LArMCParticleHelper.h"

#include <fstream>

using namespace pandora;

namespace lar_content
{

StatusCode ClusterDumpingAlgorithm::Run()
{
    for (const std::string &listName : m_clusterListNames)
        this->DumpClusterList(listName);

    return STATUS_CODE_SUCCESS;
}

//------------------------------------------------------------------------------------------------------------------------------------------

void ClusterDumpingAlgorithm::DumpClusterList(const std::string &clusterListName) const
{

    // Pick folder.
    const std::string data_folder = "/home/scratch/showerClusters";
    system(("mkdir -p " + data_folder).c_str());

    // Find a file name by just picking a file name
    // until an unused one is found.
    std::string fileName;
    int fileNum = 0;

    while (true)
    {
        fileName = data_folder + "/clusters_" +
            clusterListName + "_" + m_recoStatus + "_" +
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

        const int cId = cluster->GetParticleId();
        const int isShower = std::abs(cId) == MU_MINUS ? 0 : 1;

        for (auto const &clusterHitPair : cluster->GetOrderedCaloHitList()) {
            for (auto const &caloHit : *(clusterHitPair.second)) {
                ++len;

                const CartesianVector pos = caloHit->GetPositionVector();
                csvFile << pos.GetX() << ", "
                        << pos.GetZ() << ", "
                        << m_recoStatus << ", "
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
                    << m_recoStatus << ", "
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

        completeness = matchesMain / hitsInMC;
        purity = matchesMain / len;

        PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), treeName, "clusterNumber", clusterNumber));
        PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), treeName, "completeness", completeness));
        PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), treeName, "purity", purity));
        PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), treeName, "numberOfHits", len));
        PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), treeName, "failedHits", failedHits));
        PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), treeName, "mcID", mcID));
        PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), treeName, "isShower", isShower));
        PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), treeName, "tsIDCorrect", this->TrackShowerCheck(cId, mcID)));
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

bool ClusterDumpingAlgorithm::TrackShowerCheck(const int cId, const int mcId) const {

    std::vector<int> target;
    std::vector<int> showerLikeParticles({11, 22});
    std::vector<int> trackLikeParticles({13, 211, 2212, 321, 3222});

    if (std::abs(cId) == 11)
        target = showerLikeParticles;
    else
        target = trackLikeParticles;

    const auto it = std::find(target.begin(), target.end(), std::abs(mcId));

    return it == target.end();
}

//------------------------------------------------------------------------------------------------------------------------------------------

StatusCode ClusterDumpingAlgorithm::ReadSettings(const TiXmlHandle xmlHandle)
{

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle, "RecoState", m_recoStatus));
    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadVectorOfValues(xmlHandle,
        "InputClusterListNames", m_clusterListNames));

    return STATUS_CODE_SUCCESS;
}

//------------------------------------------------------------------------------------------------------------------------------------------

} // namespace lar_content
