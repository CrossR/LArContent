/**
 *  @file   larpandoracontent/LArCustomParticles/CustomParticleCreationAlgorithm.cc
 *
 *  @brief  Implementation of the 3D particle creation algorithm class.
 *
 *  $Log: $
 */

#include "Pandora/AlgorithmHeaders.h"

#include "larpandoracontent/LArHelpers/LArPfoHelper.h"
#include "larpandoracontent/LArHelpers/LArObjectHelper.h"
#include "larpandoracontent/LArHelpers/LArMCParticleHelper.h"

#include "larpandoracontent/LArObjects/LArMCParticle.h"
#include "larpandoracontent/LArObjects/LArTrackPfo.h"

#include "larpandoracontent/LArCustomParticles/CustomParticleCreationAlgorithm.h"

#include <fstream>
#include <sys/stat.h>

#ifdef MONITORING
#include "PandoraMonitoringApi.h"
#endif

using namespace pandora;

namespace lar_content
{

#ifdef MONITORING
void initStructForNoReco(threeDMetric &metricStruct) {
    // Set everything to -999, so we know it failed.
    metricStruct.acosDotProductAverage = -999;
    metricStruct.trackDisplacementAverageMC = -999;
    metricStruct.distanceToFitAverage = -999;
    metricStruct.numberOf3DHits = -999;
    metricStruct.lengthOfTrack = -999;
    metricStruct.numberOfErrors = -999;
}


void CustomParticleCreationAlgorithm::plotMetrics(
        const ParticleFlowObject *const pInputPfo,
        threeDMetric &metricStruct
) {
    std::cout << "********** Did not bail out! Metrics will run." << std::endl;

    // Find a file name by just picking a file name
    // until an unused one is found.
    int fileNum = 0;
    std::string fileName = "";
    std::string treeName = "threeDTrackTree";

    while (true) {

        fileName = "/home/scratch/threeDMetricOutput/threeDTrackEff_" +
            std::to_string(fileNum) +
            ".root";
        std::ifstream testFile = std::ifstream(fileName.c_str());

        if (!testFile.good()) {
            break;
        }

        testFile.close();
        ++fileNum;
    }

    // Make an output folder if needed.
    mkdir("/home/scratch/threeDMetricOutput", 0775);

    // If we haven't set the values for some reason, set the values
    // to some sensible defaults for "No reconstruction occurred."
    //
    // TODO: Here we could instead only update the values that are not set to a
    // default one, such that we see the true number of errors and more.
    if (metricStruct.valuesHaveBeenSet != errorCases::SUCCESSFULLY_SET) {
        initStructForNoReco(metricStruct);
    }

    // Calculate the ratio of 2D hits that are converted to 3D hits;
    double convertedRatio = 0.0;
    double totalNumberOf2DHits = 0.0;

    // Get the 2D clusters for this pfo.
    ClusterList clusterList;
    LArPfoHelper::GetTwoDClusterList(pInputPfo, clusterList);

    for (auto cluster : clusterList) {
        totalNumberOf2DHits += cluster->GetNCaloHits();
    }

    // Set the converted ratio.
    // This is going to be between 0 and 1, or -999 in the case of bad reco.
    if (metricStruct.numberOf3DHits == -999) {
        convertedRatio = -999;
    } else if (metricStruct.numberOf3DHits != 0) {
        convertedRatio = metricStruct.numberOf3DHits / totalNumberOf2DHits;
    } else {
        convertedRatio = 0.0;
    }

    double trackWasReconstructed = 0.0;

    switch(metricStruct.valuesHaveBeenSet) {
        case errorCases::SUCCESSFULLY_SET:
            trackWasReconstructed = 1.0;
            break;
        case errorCases::ERROR:
        case errorCases::TRACK_BUILDING_ERROR:
        case errorCases::NO_VERTEX_ERROR:
            trackWasReconstructed = 0.0;
            break;
        case errorCases::NON_NEUTRINO:
        case errorCases::NON_FINAL_STATE:
        case errorCases::NON_TRACK:
        case errorCases::NOT_SET:
            trackWasReconstructed = -999;
            break;
    }

    std::cout << "Number of 2D Hits: " << totalNumberOf2DHits << std::endl;
    std::cout << "Number of 3D Hits: " << metricStruct.numberOf3DHits << std::endl;
    std::cout << "Ratio: " << convertedRatio << std::endl;

    if (convertedRatio < 0.2 && totalNumberOf2DHits > 25 && convertedRatio != -999) {
        std::cout << "#### THE CONVERTED RATIO ("
                  << convertedRatio
                  << ") WAS VERY LOW FOR THIS PARTICLE."
                  << std::endl;
    }

    if (convertedRatio == -999) {
        std::cout << "#### THIS PARTICLE HAS NO 3D HITS."
                  << std::endl;
    }

    double reconstructionState = -999;

    switch(metricStruct.valuesHaveBeenSet) {
        case errorCases::NOT_SET:
            reconstructionState = 0;
            break;
        case errorCases::ERROR:
            reconstructionState = 1;
            break;
        case errorCases::SUCCESSFULLY_SET:
            reconstructionState = 2;
            break;
        case errorCases::NON_NEUTRINO:
            reconstructionState = 3;
            break;
        case errorCases::NON_FINAL_STATE:
            reconstructionState = 4;
            break;
        case errorCases::NON_TRACK:
            reconstructionState = 5;
            break;
        case errorCases::TRACK_BUILDING_ERROR:
            reconstructionState = 6;
            break;
        case errorCases::NO_VERTEX_ERROR:
            reconstructionState = 7;
            break;
        default:
            reconstructionState = -999;
            break;
    }

    // Setup the branches, fill them, and then finish up the file.
    PANDORA_MONITORING_API(Create(this->GetPandora()));
    PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), treeName.c_str(), "acosDotProductAverage", metricStruct.acosDotProductAverage));
    PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), treeName.c_str(), "sqdTrackDisplacementAverageMC", metricStruct.trackDisplacementAverageMC));
    PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), treeName.c_str(), "distanceToFitAverage", metricStruct.distanceToFitAverage));

    PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), treeName.c_str(), "numberOf3DHits", metricStruct.numberOf3DHits));
    PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), treeName.c_str(), "numberOf2DHits", totalNumberOf2DHits));
    PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), treeName.c_str(), "ratioOf3Dto2D", convertedRatio));
    PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), treeName.c_str(), "numberOfErrors", metricStruct.numberOfErrors));
    PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), treeName.c_str(), "lengthOfTrack", metricStruct.lengthOfTrack));
    PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), treeName.c_str(), "trackWasReconstructed", trackWasReconstructed));
    PANDORA_MONITORING_API(SetTreeVariable(this->GetPandora(), treeName.c_str(), "reconstructionState", reconstructionState));

    PANDORA_MONITORING_API(FillTree(this->GetPandora(), treeName.c_str()));
    PANDORA_MONITORING_API(SaveTree(this->GetPandora(), treeName.c_str(), fileName.c_str(), "RECREATE"));
    PANDORA_MONITORING_API(Delete(this->GetPandora()));
    std::cout << "**********" << std::endl;
}
#endif

StatusCode CustomParticleCreationAlgorithm::Run()
{
    const MCParticleList *pMCParticleList = nullptr;
    PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::GetList(*this, m_mcParticleListName, pMCParticleList));

    // Get input Pfo List
    const PfoList *pPfoList(NULL);

    if (STATUS_CODE_SUCCESS != PandoraContentApi::GetList(*this, m_pfoListName, pPfoList))
    {
        if (PandoraContentApi::GetSettings(*this)->ShouldDisplayAlgorithmInfo())
            std::cout << "CustomParticleCreationAlgorithm: cannot find pfo list " << m_pfoListName << std::endl;

        return STATUS_CODE_SUCCESS;
    }

    // Get input Vertex List
    const VertexList *pVertexList(NULL);

    if (STATUS_CODE_SUCCESS != PandoraContentApi::GetList(*this, m_vertexListName, pVertexList))
    {
        if (PandoraContentApi::GetSettings(*this)->ShouldDisplayAlgorithmInfo())
            std::cout << "CustomParticleCreationAlgorithm: cannot find vertex list " << m_vertexListName << std::endl;

        return STATUS_CODE_SUCCESS;
    }

    // Create temporary lists
    const PfoList *pTempPfoList = NULL; std::string tempPfoListName;
    PANDORA_THROW_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::CreateTemporaryListAndSetCurrent(*this, pTempPfoList,
        tempPfoListName));

    const VertexList *pTempVertexList = NULL; std::string tempVertexListName;
    PANDORA_THROW_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::CreateTemporaryListAndSetCurrent(*this, pTempVertexList,
        tempVertexListName));

    // Loop over input Pfos
    PfoList pfoList(pPfoList->begin(), pPfoList->end());
    VertexList vertexList(pVertexList->begin(), pVertexList->end());
    MCParticleList mcList(pMCParticleList->begin(), pMCParticleList->end());

    for (PfoList::const_iterator iter = pfoList.begin(), iterEnd = pfoList.end(); iter != iterEnd; ++iter)
    {
        const ParticleFlowObject *const pInputPfo = *iter;

        threeDMetric metricStruct;
        metricStruct.valuesHaveBeenSet = errorCases::NOT_SET;

        if (pInputPfo->GetVertexList().empty()) {
#ifdef MONITORING
            // Value wasn't set due to an error.
            metricStruct.valuesHaveBeenSet = errorCases::NO_VERTEX_ERROR;

            plotMetrics(pInputPfo, metricStruct);
#endif
            continue;
        }

        const Vertex *const pInputVertex = LArPfoHelper::GetVertex(pInputPfo);
        const MCParticle *const pMCParticle = LArMCParticleHelper::GetMainMCParticle(pInputPfo);

        if (vertexList.end() == std::find(vertexList.begin(), vertexList.end(), pInputVertex)) {
            throw StatusCodeException(STATUS_CODE_FAILURE);
        }

        if (mcList.end() == std::find(mcList.begin(), mcList.end(), pMCParticle)) {
            throw StatusCodeException(STATUS_CODE_FAILURE);
        }

        // Build a new pfo and vertex from the old pfo
        const ParticleFlowObject *pOutputPfo(NULL);

        // Pass over the input and populate the output, whilst also passing
        // over the MC particle for verifying the 3D positions.
        this->CreatePfo(pInputPfo, pOutputPfo, metricStruct);
#ifdef MONITORING

        const LArTrackPfo *const pLArTrackPfo(dynamic_cast<const LArTrackPfo *>(pOutputPfo));

        if (pLArTrackPfo != NULL && m_runMetrics)
        {
            // Build up the required information for the metric generation for plotting stuff out.
            const LArMCParticle *const pLArMCParticle(dynamic_cast<const LArMCParticle *>(pMCParticle));

            CartesianPointVector pointVector;
            CartesianPointVector pointVectorMC;

            // Get the hits to build up the two point vectors for the sliding fits.
            for (const auto &nextPoint : pLArTrackPfo->m_trackStateVector)
                pointVector.push_back(nextPoint.GetPosition());

            for (const auto &nextMCHit : pLArMCParticle->GetMCStepPositions())
                pointVectorMC.push_back(LArObjectHelper::TypeAdaptor::GetPosition(nextMCHit));

            const LArTPC *const pFirstLArTPC(this->GetPandora().GetGeometry()->GetLArTPCMap().begin()->second);
            const float layerPitch(pFirstLArTPC->GetWirePitchW());

            // Fill in the 3D hit metrics now.
            // Use both the MC and standard fits if possible, otherwise just the standard.
            if (pointVector.size() > 10 && pointVectorMC.size() > 10)
            {
                const ThreeDSlidingFitResult slidingFit(&pointVector, 20, layerPitch);
                const ThreeDSlidingFitResult slidingFitMC(&pointVectorMC, 20, layerPitch);

                LArMetricHelper::GetThreeDMetrics(&pointVector, &slidingFit, metricStruct, &slidingFitMC);
            }
            else if (pointVector.size() > 3)
            {
                const ThreeDSlidingFitResult slidingFit(&pointVector, 20, layerPitch);

                LArMetricHelper::GetThreeDMetrics(&pointVector, &slidingFit, metricStruct, NULL);
            }

            // Even if there is stuff missing, we still want to plot the results out to log that
            // there was missing parts.
            plotMetrics(pInputPfo, metricStruct);
        }

#endif

        if (NULL == pOutputPfo)
            continue;

        if (pOutputPfo->GetVertexList().empty())
            throw StatusCodeException(STATUS_CODE_FAILURE);

        // Transfer clusters and hierarchy information to new pfo, and delete old pfo and vertex
        ClusterList clusterList(pInputPfo->GetClusterList().begin(), pInputPfo->GetClusterList().end());
        PfoList parentList(pInputPfo->GetParentPfoList().begin(), pInputPfo->GetParentPfoList().end());
        PfoList daughterList(pInputPfo->GetDaughterPfoList().begin(), pInputPfo->GetDaughterPfoList().end());

        PANDORA_THROW_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::Delete<Pfo>(*this, pInputPfo, m_pfoListName));
        PANDORA_THROW_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::Delete<Vertex>(*this, pInputVertex, m_vertexListName));

        for (ClusterList::const_iterator cIter = clusterList.begin(), cIterEnd = clusterList.end(); cIter != cIterEnd; ++cIter)
        {
            PANDORA_THROW_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::AddToPfo<Cluster>(*this, pOutputPfo, *cIter));
        }

        for (PfoList::const_iterator pIter = parentList.begin(), pIterEnd = parentList.end(); pIter != pIterEnd; ++pIter)
        {
            PANDORA_THROW_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::SetPfoParentDaughterRelationship(*this, *pIter, pOutputPfo));
        }

        for (PfoList::const_iterator dIter = daughterList.begin(), dIterEnd = daughterList.end(); dIter != dIterEnd; ++dIter)
        {
            PANDORA_THROW_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::SetPfoParentDaughterRelationship(*this, pOutputPfo, *dIter));
        }
    }

    if (!pTempPfoList->empty())
    {
        PANDORA_THROW_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::SaveList<Pfo>(*this, m_pfoListName));
        PANDORA_THROW_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::ReplaceCurrentList<Pfo>(*this, m_pfoListName));
    }

    if (!pTempVertexList->empty())
    {
        PANDORA_THROW_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::SaveList<Vertex>(*this, m_vertexListName));
        PANDORA_THROW_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::ReplaceCurrentList<Vertex>(*this, m_vertexListName));
    }

    return STATUS_CODE_SUCCESS;
}

//------------------------------------------------------------------------------------------------------------------------------------------

StatusCode CustomParticleCreationAlgorithm::ReadSettings(const TiXmlHandle xmlHandle)
{
    PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, XmlHelper::ReadValue(xmlHandle, "PfoListName", m_pfoListName));
    PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, XmlHelper::ReadValue(xmlHandle, "VertexListName", m_vertexListName));
    /* PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, XmlHelper::ReadValue(xmlHandle, "MCParticleListName", m_mcParticleListName)); */
    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle, "RunMetrics", m_runMetrics));
    m_mcParticleListName = "Input";

    return STATUS_CODE_SUCCESS;
}

} // namespace lar_content
