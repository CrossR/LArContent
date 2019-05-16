/**
 *  @file   larpandoracontent/LArCustomParticles/CustomParticleCreationAlgorithm.cc
 *
 *  @brief  Implementation of the 3D particle creation algorithm class.
 *
 *  $Log: $
 */

#include "Pandora/AlgorithmHeaders.h"

#include "larpandoracontent/LArHelpers/LArPfoHelper.h"
#include "larpandoracontent/LArHelpers/LArMCParticleHelper.h"

#include "larpandoracontent/LArObjects/LArMCParticle.h"

#include "larpandoracontent/LArCustomParticles/CustomParticleCreationAlgorithm.h"

#include <fstream>
#include <sys/stat.h>

#include "TTree.h"
#include "TFile.h"
#include "TBranch.h"

using namespace pandora;

namespace lar_content
{

void initStructForNoReco(threeDMetric &metricStruct) {
    // Set everything to -999, so we know it failed.
    metricStruct.acosDotProductAverage = -999;
    metricStruct.trackDisplacementAverageMC = -999;
    metricStruct.distanceToFitAverage = -999;
    metricStruct.numberOf3DHits = -999;
    metricStruct.lengthOfTrack = -999;
    metricStruct.numberOfErrors = -999;
}


void plotMetrics(
        const ParticleFlowObject *const pInputPfo,
        threeDMetric &metricStruct
) {
    std::cout << "*************************************************** Did not bail out! Metrics will run." << std::endl;
    // Log out result to ROOT file for plotting.

    // Setup an output tree by just picking a file name
    // until an unused one is found.
    int fileNum = 0;
    std::string fileName = "";

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

    // Make an output folder if needed and a file in it.
    mkdir("/home/scratch/threeDMetricOutput", 0775);
    TFile* f = new TFile(fileName.c_str(), "RECREATE");
    TTree* tree = new TTree("threeDTrackTree", "threeDTrackTree", 0);

    // If we haven't set the values for some reason, set the values
    // to some sensible defaults for "No reconstruction occurred."
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
    if (metricStruct.distanceToFitAverage == -999) {
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
    tree->Branch("acosDotProductAverage", &metricStruct.acosDotProductAverage, 0);
    tree->Branch("trackDisplacementAverageMC", &metricStruct.trackDisplacementAverageMC, 0);
    tree->Branch("distanceToFitAverage", &metricStruct.distanceToFitAverage, 0);

    tree->Branch("numberOf3DHits", &metricStruct.numberOf3DHits, 0);
    tree->Branch("numberOf2DHits", &totalNumberOf2DHits, 0);
    tree->Branch("ratioOf3Dto2D", &convertedRatio, 0);
    tree->Branch("numberOfErrors", &metricStruct.numberOfErrors, 0);
    tree->Branch("lengthOfTrack", &metricStruct.lengthOfTrack, 0);
    tree->Branch("trackWasReconstructed", &trackWasReconstructed, 0);
    tree->Branch("reconstructionState", &reconstructionState, 0);

    tree->Fill();
    f->Write();
    f->Close();
}

StatusCode CustomParticleCreationAlgorithm::Run()
{
    const MCParticleList *pMCParticleList = nullptr;
    PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::GetList(*this, m_mcParticleListName, pMCParticleList));
    std::cout << "Found " << pMCParticleList->size() << " MC Particles." << std::endl;

    // Get input Pfo List
    const PfoList *pPfoList(NULL);

    if (STATUS_CODE_SUCCESS != PandoraContentApi::GetList(*this, m_pfoListName, pPfoList))
    {
        if (PandoraContentApi::GetSettings(*this)->ShouldDisplayAlgorithmInfo())
            std::cout << "CustomParticleCreationAlgorithm: cannot find pfo list " << m_pfoListName << std::endl;

        std::cout << "*************************************************** Bailed out due to missing pfoList" << std::endl;
        return STATUS_CODE_SUCCESS;
    }

    // Get input Vertex List
    const VertexList *pVertexList(NULL);

    if (STATUS_CODE_SUCCESS != PandoraContentApi::GetList(*this, m_vertexListName, pVertexList))
    {
        if (PandoraContentApi::GetSettings(*this)->ShouldDisplayAlgorithmInfo())
            std::cout << "CustomParticleCreationAlgorithm: cannot find vertex list " << m_vertexListName << std::endl;

        std::cout << "*************************************************** Bailed out due to missing vertexList" << std::endl;
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

    std::cout << "p: " << pfoList.size() << std::endl;
    std::cout << "v: " << vertexList.size() << std::endl;
    std::cout << "m: " << mcList.size() << std::endl;

    for (PfoList::const_iterator iter = pfoList.begin(), iterEnd = pfoList.end(); iter != iterEnd; ++iter)
    {
        const ParticleFlowObject *const pInputPfo = *iter;

        threeDMetric metricStruct;
        metricStruct.valuesHaveBeenSet = errorCases::NOT_SET;

        if (pInputPfo->GetVertexList().empty()) {
            // std::cout << "*************************************************** Bailed out due to missing vertexList for current pfo from pfoList" << std::endl;

            // Value wasn't set due to an error.
            metricStruct.valuesHaveBeenSet = errorCases::NO_VERTEX_ERROR;
            plotMetrics(pInputPfo, metricStruct);
            continue;
        }

        const Vertex *const pInputVertex = LArPfoHelper::GetVertex(pInputPfo);
        const MCParticle *const pMCParticle = LArMCParticleHelper::GetMainMCParticle(pInputPfo);

        if (vertexList.end() == std::find(vertexList.begin(), vertexList.end(), pInputVertex)) {
            std::cout << "*************************************************** Bailed out due to missing vertex for current pfo from pfoList" << std::endl;
            throw StatusCodeException(STATUS_CODE_FAILURE);
        }

        if (mcList.end() == std::find(mcList.begin(), mcList.end(), pMCParticle)) {
            std::cout << "*************************************************** Bailed out due to missing MC for current pfo from pfoList" << std::endl;
            throw StatusCodeException(STATUS_CODE_FAILURE);
        }

        // Build a new pfo and vertex from the old pfo
        const ParticleFlowObject *pOutputPfo(NULL);

        // Pass over the input and populate the output, whilst also passing
        // over the MC particle for verifying the 3D positions.
        this->CreatePfo(pInputPfo, pOutputPfo, metricStruct, pMCParticle);
        plotMetrics(pInputPfo, metricStruct);

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
    std::cout << "Loading CustomParticleCreationAlgorithm settings..." << std::endl;
    PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, XmlHelper::ReadValue(xmlHandle, "PfoListName", m_pfoListName));
    PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, XmlHelper::ReadValue(xmlHandle, "VertexListName", m_vertexListName));
    /* PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, XmlHelper::ReadValue(xmlHandle, "MCParticleListName", m_mcParticleListName)); */
    m_mcParticleListName = "Input";

    return STATUS_CODE_SUCCESS;
}

} // namespace lar_content
