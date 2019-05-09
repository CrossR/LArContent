/**
 *  @file   larpandoracontent/LArCustomParticles/TrackParticleBuildingAlgorithm.cc
 *
 *  @brief  Implementation of the 3D track building algorithm class.
 *
 *  $Log: $
 */

#include "Pandora/AlgorithmHeaders.h"

#include "Managers/GeometryManager.h"

#include "larpandoracontent/LArHelpers/LArClusterHelper.h"
#include "larpandoracontent/LArHelpers/LArPfoHelper.h"

#include "larpandoracontent/LArObjects/LArTrackPfo.h"

#include "larpandoracontent/LArCustomParticles/TrackParticleBuildingAlgorithm.h"

#include <fstream>
#include <sys/stat.h>

#include "TTree.h"
#include "TFile.h"
#include "TBranch.h"

using namespace pandora;

namespace lar_content
{

TrackParticleBuildingAlgorithm::TrackParticleBuildingAlgorithm() :
    m_slidingFitHalfWindow(20)
{

}

//------------------------------------------------------------------------------------------------------------------------------------------

void TrackParticleBuildingAlgorithm::CreatePfo(
        const ParticleFlowObject *const pInputPfo,
        const ParticleFlowObject*& pOutputPfo,
        const MCParticle *const pMCParticle
) const
{
    try
    {
        // Need an input vertex to provide a track propagation direction
        const Vertex *const pInputVertex = LArPfoHelper::GetVertex(pInputPfo);

        // In cosmic mode, build tracks from all parent pfos, otherwise require that pfo is track-like
        if (LArPfoHelper::IsNeutrinoFinalState(pInputPfo))
        {
            if (!LArPfoHelper::IsTrack(pInputPfo))
                return;
        }
        else
        {
            if (!LArPfoHelper::IsFinalState(pInputPfo))
                return;

            if (LArPfoHelper::IsNeutrino(pInputPfo))
                return;
        }

        // ATTN If wire w pitches vary between TPCs, exception will be raised in initialisation of lar pseudolayer plugin
        const LArTPC *const pFirstLArTPC(this->GetPandora().GetGeometry()->GetLArTPCMap().begin()->second);
        const float layerPitch(pFirstLArTPC->GetWirePitchW());

        // Calculate sliding fit trajectory
        LArTrackStateVector trackStateVector;
        threeDMetric metricStruct;

        LArPfoHelper::GetSlidingFitTrajectory(
                pInputPfo,
                pInputVertex,
                m_slidingFitHalfWindow,
                layerPitch,
                trackStateVector,
                metricStruct,
                pMCParticle
        );

        if (trackStateVector.empty())
            return;

        // Log out result to ROOT file for plotting.

        // Setup an output tree by just picking a file name
        // until an unused one is found.
        int fileNum = 0;
        std::string fileName = "";

        while (true) {

            fileName = "output/threeDTrackEff_" +
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
        mkdir("output", 0775);
        TFile* f = new TFile(fileName.c_str(), "RECREATE");
        TTree* tree = new TTree("threeDTrackTree", "threeDTrackTree", 0);

        // Calculate the ratio of 2D hits that are converted to 3D hits;
        double convertedRatio = 0.0;
        double totalNumberOf2DHits = 0.0;

        // Get the 2D clusters for this pfo.
        ClusterList clusterList;
        LArPfoHelper::GetTwoDClusterList(pInputPfo, clusterList);

        for (auto cluster : clusterList) {
            totalNumberOf2DHits += cluster->GetNCaloHits();
        }

        convertedRatio = 1 - (totalNumberOf2DHits - metricStruct.numberOf3DHits) / totalNumberOf2DHits;

        std::cout << "Number of 2D Hits: " << totalNumberOf2DHits << std::endl;
        std::cout << "Number of 3D Hits: " << metricStruct.numberOf3DHits << std::endl;
        std::cout << "Ratio: " << convertedRatio << std::endl;

        // Setup the branches, fill them, and then finish up the file.
        tree->Branch("acosDotProductAverage", &metricStruct.acosDotProductAverage, 0);
        tree->Branch("trackDisplacementAverageMC", &metricStruct.trackDisplacementAverageMC, 0);
        tree->Branch("distanceToFitAverage", &metricStruct.distanceToFitAverage, 0);

        tree->Branch("numberOf3DHits", &metricStruct.numberOf3DHits, 0);
        tree->Branch("numberOf2DHits", &totalNumberOf2DHits, 0);
        tree->Branch("ratioOf3Dto2D", &convertedRatio, 0);
        tree->Branch("numberOfErrors", &metricStruct.numberOfErrors, 0);
        tree->Branch("lengthOfTrack", &metricStruct.lengthOfTrack, 0);

        tree->Fill();
        f->Write();
        f->Close();

        // Build track-like pfo from track trajectory (TODO Correct these placeholder parameters)
        LArTrackPfoFactory trackFactory;
        LArTrackPfoParameters pfoParameters;
        pfoParameters.m_particleId = (LArPfoHelper::IsTrack(pInputPfo) ? pInputPfo->GetParticleId() : MU_MINUS);
        pfoParameters.m_charge = PdgTable::GetParticleCharge(pfoParameters.m_particleId.Get());
        pfoParameters.m_mass = PdgTable::GetParticleMass(pfoParameters.m_particleId.Get());
        pfoParameters.m_energy = 0.f;
        pfoParameters.m_momentum = pInputPfo->GetMomentum();
        pfoParameters.m_propertiesToAdd = pInputPfo->GetPropertiesMap();
        pfoParameters.m_trackStateVector = trackStateVector;

        PANDORA_THROW_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::ParticleFlowObject::Create(*this, pfoParameters, pOutputPfo,
            trackFactory));

        const LArTrackPfo *const pLArPfo = dynamic_cast<const LArTrackPfo*>(pOutputPfo);
        if (NULL == pLArPfo)
            throw StatusCodeException(STATUS_CODE_FAILURE);

        // Now update vertex and direction
        PandoraContentApi::ParticleFlowObject::Metadata pfodata;
        pfodata.m_momentum = pLArPfo->GetVertexDirection();
        PANDORA_THROW_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::ParticleFlowObject::AlterMetadata(*this, pOutputPfo, pfodata));

        const Vertex *pOutputVertex(NULL);

        PandoraContentApi::Vertex::Parameters vtxParameters;
        vtxParameters.m_position = pLArPfo->GetVertexPosition();
        vtxParameters.m_vertexLabel = pInputVertex->GetVertexLabel();
        vtxParameters.m_vertexType = pInputVertex->GetVertexType();

        PANDORA_THROW_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::Vertex::Create(*this, vtxParameters, pOutputVertex));
        PANDORA_THROW_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::AddToPfo(*this, pOutputPfo, pOutputVertex));
    }
    catch (StatusCodeException &statusCodeException)
    {
        if (STATUS_CODE_FAILURE == statusCodeException.GetStatusCode())
            throw statusCodeException;
    }
}

//------------------------------------------------------------------------------------------------------------------------------------------

StatusCode TrackParticleBuildingAlgorithm::ReadSettings(const TiXmlHandle xmlHandle)
{
    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle,
        "SlidingFitHalfWindow", m_slidingFitHalfWindow));

    return CustomParticleCreationAlgorithm::ReadSettings(xmlHandle);
}

} // namespace lar_content
