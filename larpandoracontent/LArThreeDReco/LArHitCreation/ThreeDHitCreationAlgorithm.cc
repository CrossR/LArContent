/**
 *  @file   larpandoracontent/LArThreeDReco/LArHitCreation/ThreeDHitCreationAlgorithm.cc
 *
 *  @brief  Implementation of the three dimensional hit creation algorithm class.
 *
 *  $Log: $
 */

#include "Pandora/AlgorithmHeaders.h"

#include "larpandoracontent/LArHelpers/LArClusterHelper.h"
#include "larpandoracontent/LArHelpers/LArGeometryHelper.h"
#include "larpandoracontent/LArHelpers/LArPfoHelper.h"
#include "larpandoracontent/LArHelpers/LArObjectHelper.h"
#include "larpandoracontent/LArHelpers/LArMCParticleHelper.h"

#include "larpandoracontent/LArObjects/LArMCParticle.h"
#include "larpandoracontent/LArObjects/LArThreeDSlidingFitResult.h"

#include "larpandoracontent/LArThreeDReco/LArHitCreation/HitCreationBaseTool.h"
#include "larpandoracontent/LArThreeDReco/LArHitCreation/ThreeDHitCreationAlgorithm.h"

#include <algorithm>

using namespace pandora;

namespace lar_content
{

ThreeDHitCreationAlgorithm::ThreeDHitCreationAlgorithm() :
    m_iterateTrackHits(true),
    m_iterateShowerHits(false),
    m_useInterpolation(false),
    m_slidingFitHalfWindow(10),
    m_nHitRefinementIterations(10),
    m_sigma3DFitMultiplier(0.2),
    m_iterationMaxChi2Ratio(1.),
    m_interpolationCutOff(10.)
{
}

//------------------------------------------------------------------------------------------------------------------------------------------

void ThreeDHitCreationAlgorithm::FilterCaloHitsByType(const CaloHitVector &inputCaloHitVector, const HitType hitType, CaloHitVector &outputCaloHitVector) const
{
    for (const CaloHit *const pCaloHit : inputCaloHitVector)
    {
        if (hitType == pCaloHit->GetHitType())
            outputCaloHitVector.push_back(pCaloHit);
    }
}

//------------------------------------------------------------------------------------------------------------------------------------------

StatusCode ThreeDHitCreationAlgorithm::Run()
{
    const PfoList *pPfoList(nullptr);
    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_INITIALIZED, !=, PandoraContentApi::GetList(*this, m_inputPfoListName, pPfoList));

    if (!pPfoList || pPfoList->empty())
    {
        if (PandoraContentApi::GetSettings(*this)->ShouldDisplayAlgorithmInfo())
            std::cout << "ThreeDHitCreationAlgorithm: unable to find pfo list " << m_inputPfoListName << std::endl;

        return STATUS_CODE_SUCCESS;
    }

    CaloHitList allNewThreeDHits;
    ProtoHitVectorMap allProtoHitVectors;

    PfoVector pfoVector(pPfoList->begin(), pPfoList->end());
    std::sort(pfoVector.begin(), pfoVector.end(), LArPfoHelper::SortByNHits);

    for (const ParticleFlowObject *const pPfo : pfoVector)
    {
        ProtoHitVector protoHitVector;
        int toolNum = 0;

        for (HitCreationBaseTool *const pHitCreationTool : m_algorithmToolVector)
        {
            std::cout << "## Running Tool " << toolNum << std::endl;
            ++toolNum;

            CaloHitVector remainingTwoDHits;

            this->SeparateTwoDHits(pPfo, protoHitVector, remainingTwoDHits);

            if (remainingTwoDHits.empty())
                break;

            pHitCreationTool->Run(this, pPfo, remainingTwoDHits, protoHitVector);

            // If we are using the interpolation, we don't want to separate the
            // 2D hits.  We want every tool to have the full set of 2D hits to
            // get their best approximation of what the 3D reconstruction is.
            // To do that, we should clear the protoHitVector so that no hits
            // are removed for the next algorithm.
            if (m_useInterpolation)
            {

                const MCParticle *const pMCParticle = LArMCParticleHelper::GetMainMCParticle(pPfo);
                const LArMCParticle *const pLArMCParticle(dynamic_cast<const LArMCParticle *>(pMCParticle));

                CartesianPointVector pointVectorBefore;
                CartesianPointVector pointVectorMC;

                for (const auto &nextPoint : protoHitVector)
                    pointVectorBefore.push_back(nextPoint.GetPosition3D());

                for (const auto &nextMCHit : pLArMCParticle->GetMCStepPositions())
                    pointVectorMC.push_back(LArObjectHelper::TypeAdaptor::GetPosition(nextMCHit));


                // TODO: I think we need more of these guards in other parts of the code.
                if (pointVectorBefore.size() <= 3)
                    continue;

                if (pointVectorMC.size() <= 3)
                    continue;

                const float layerPitch(LArGeometryHelper::GetWireZPitch(this->GetPandora()));
                const ThreeDSlidingFitResult slidingFit(&pointVectorBefore, m_slidingFitHalfWindow, layerPitch);
                const ThreeDSlidingFitResult slidingFitMC(&pointVectorMC, 20, layerPitch);
                threeDMetric metricsBefore;

                // Populate the metrics.
                LArMetricHelper::GetThreeDMetrics(&pointVectorBefore, &slidingFit, metricsBefore, &slidingFitMC);

                for (unsigned int i = 0; i < 10; ++i)
                {

                        std::cout << "## Interpolation Stage " << i << std::endl;

                        int sizeBefore = protoHitVector.size();
                        this->InterpolationMethod(pPfo, protoHitVector);
                        int sizeAfter = protoHitVector.size();

                        if (sizeBefore == sizeAfter)
                        {
                            std::cout << "## Size was equal after iteration " << i << std::endl;
                        break;
                    }
                }

                CartesianPointVector pointVectorAfter;

                for (const auto &nextPoint : protoHitVector)
                    pointVectorAfter.push_back(nextPoint.GetPosition3D());

                // TODO: I think we need more of these guards in other parts of the code.
                if (pointVectorAfter.size() <= 1)
                    continue;

                const ThreeDSlidingFitResult slidingFitAfter(&pointVectorAfter, m_slidingFitHalfWindow, layerPitch);
                threeDMetric metricsAfter;

                // Populate the metrics.
                LArMetricHelper::GetThreeDMetrics(&pointVectorAfter, &slidingFitAfter, metricsAfter, &slidingFitMC);

                auto metricDiffMC =  std::fabs(metricsAfter.trackDisplacementAverageMC - metricsBefore.trackDisplacementAverageMC);
                auto metricDiffDist =  std::fabs(metricsAfter.distanceToFitAverage - metricsBefore.distanceToFitAverage);
                std::cout << "######################## metricDiffMC = " << metricDiffMC << ", "
                                                                        << metricsBefore.trackDisplacementAverageMC << ", "
                                                                        << metricsAfter.trackDisplacementAverageMC << ", "
                                                                        << std::endl;

                std::cout << "######################## metricDiffDist = " << metricDiffDist << ", "
                                                                        << metricsBefore.distanceToFitAverage << ", "
                                                                        << metricsAfter.distanceToFitAverage << ", "
                                                                        << std::endl;

                // TODO: Should we be running the IterativeTreatment here as well?

                allProtoHitVectors.insert(ProtoHitVectorMap::value_type(pHitCreationTool->GetInstanceName(), protoHitVector));
                protoHitVector.clear();
            }
        }

        // Only do the iterative treatment if it has been asked for,
        // is the correct particle type, and the consolidated method hasn't
        // been enabled, since that takes priority.
        bool shouldUseIterativeTreatment = (
                (m_iterateTrackHits && LArPfoHelper::IsTrack(pPfo)) ||
                (m_iterateShowerHits && LArPfoHelper::IsShower(pPfo))
        );

        if (shouldUseIterativeTreatment && !m_useInterpolation)
        {
            this->IterativeTreatment(protoHitVector);
            std::cout << "At the end of the IterativeTreatment, the protoHitVector was of size: " << protoHitVector.size() << std::endl;
        }

        if (m_useInterpolation)
        {
            this->ConsolidatedMethod(pPfo, allProtoHitVectors, protoHitVector);
            allProtoHitVectors.clear();
        }

        if (protoHitVector.empty())
            continue;

        CaloHitList newThreeDHits;
        this->CreateThreeDHits(protoHitVector, newThreeDHits);
        this->AddThreeDHitsToPfo(pPfo, newThreeDHits);

        allNewThreeDHits.insert(allNewThreeDHits.end(), newThreeDHits.begin(), newThreeDHits.end());
    }

    if (!allNewThreeDHits.empty())
        PANDORA_THROW_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::SaveList(*this, allNewThreeDHits, m_outputCaloHitListName));

    return STATUS_CODE_SUCCESS;
}

//------------------------------------------------------------------------------------------------------------------------------------------

void ThreeDHitCreationAlgorithm::SeparateTwoDHits(const ParticleFlowObject *const pPfo, const ProtoHitVector &protoHitVector, CaloHitVector &remainingHitVector) const
{
    ClusterList twoDClusterList;
    LArPfoHelper::GetTwoDClusterList(pPfo, twoDClusterList);
    CaloHitList remainingHitList;

    for (const Cluster *const pCluster : twoDClusterList)
    {
        if (TPC_3D == LArClusterHelper::GetClusterHitType(pCluster))
            throw StatusCodeException(STATUS_CODE_FAILURE);

        pCluster->GetOrderedCaloHitList().FillCaloHitList(remainingHitList);
    }

    CaloHitSet remainingHitSet(remainingHitList.begin(), remainingHitList.end());

    for (const ProtoHit &protoHit : protoHitVector)
    {
        CaloHitSet::iterator eraseIter = remainingHitSet.find(protoHit.GetParentCaloHit2D());

        if (remainingHitSet.end() == eraseIter)
            throw StatusCodeException(STATUS_CODE_FAILURE);

        remainingHitSet.erase(eraseIter);
    }

    remainingHitVector.insert(remainingHitVector.end(), remainingHitSet.begin(), remainingHitSet.end());
    std::sort(remainingHitVector.begin(), remainingHitVector.end(), LArClusterHelper::SortHitsByPosition);
}

//------------------------------------------------------------------------------------------------------------------------------------------

void ThreeDHitCreationAlgorithm::IterativeTreatment(ProtoHitVector &protoHitVector) const
{
    const float layerPitch(LArGeometryHelper::GetWireZPitch(this->GetPandora()));
    const unsigned int layerWindow(m_slidingFitHalfWindow);

    double originalChi2(0.);
    CartesianPointVector currentPoints3D;
    this->ExtractResults(protoHitVector, originalChi2, currentPoints3D);

    try
    {
        const ThreeDSlidingFitResult originalSlidingFitResult(&currentPoints3D, layerWindow, layerPitch);
        const double originalChi2WrtFit(this->GetChi2WrtFit(originalSlidingFitResult, protoHitVector));
        double currentChi2(originalChi2 + originalChi2WrtFit);

        unsigned int nIterations(0);

        while (nIterations++ < m_nHitRefinementIterations)
        {
            ProtoHitVector newProtoHitVector(protoHitVector);
            const ThreeDSlidingFitResult newSlidingFitResult(&currentPoints3D, layerWindow, layerPitch);
            this->RefineHitPositions(newSlidingFitResult, newProtoHitVector);

            double newChi2(0.);
            CartesianPointVector newPoints3D;
            this->ExtractResults(newProtoHitVector, newChi2, newPoints3D);

            if (newChi2 > m_iterationMaxChi2Ratio * currentChi2)
                break;

            currentChi2 = newChi2;
            currentPoints3D = newPoints3D;
            protoHitVector = newProtoHitVector;
        }
    }
    catch (const StatusCodeException &)
    {
    }
}

//------------------------------------------------------------------------------------------------------------------------------------------

bool sortByTwoDX(const lar_content::ThreeDHitCreationAlgorithm::ProtoHit &a, const lar_content::ThreeDHitCreationAlgorithm::ProtoHit &b) {
    return a.GetParentCaloHit2D()->GetPositionVector().GetX() < b.GetParentCaloHit2D()->GetPositionVector().GetX();
}

void ThreeDHitCreationAlgorithm::ConsolidatedMethod(const ParticleFlowObject *const pPfo, ProtoHitVectorMap &protoHitVectorMap,
        ProtoHitVector &protoHitVector) const
{
    std::cout << "Starting consolidated method..." << std::endl;

    // Get the 2D clusters for this pfo.
    ClusterList clusterList;
    LArPfoHelper::GetTwoDClusterList(pPfo, clusterList);
    int totalNumberOf2DHits = 0;

    for (auto cluster : clusterList) {
        totalNumberOf2DHits += cluster->GetNCaloHits();
    }

    std::vector<std::pair<float, std::string>> scores;

    // For now, lets just do the same thing that the default algorithm set does.
    // Go over every algorithm in turn and just pull out every hit that we haven't
    // already got matches to.
    //
    // Should be noted that right now, this is dumb and just assumes the insert
    // order is the order we want to pull hits from. In reality, we should be
    // using the actual tool name or something to get the tool we want, not
    // just the insert order.
    for (ProtoHitVectorMap::value_type protoHitVectorPair : protoHitVectorMap)
    {
        // Setup for metric generation.
        CartesianPointVector pointVector;

        for (const auto &nextPoint : protoHitVectorPair.second)
            pointVector.push_back(nextPoint.GetPosition3D());

        // TODO: I think we need more of these guards in other parts of the code.
        if (pointVector.size() <= 1)
            continue;

        const float layerPitch(LArGeometryHelper::GetWireZPitch(this->GetPandora()));
        const ThreeDSlidingFitResult slidingFit(&pointVector, m_slidingFitHalfWindow, layerPitch);
        threeDMetric metrics;

        // Populate the metrics.
        LArMetricHelper::GetThreeDMetrics(&pointVector, &slidingFit, metrics, NULL);

        // Now we need to calculate a score for this algorithm.
        // We've got a few things we can use here:
        //  - Track Wiggle
        //  - Track displacement from the 2D hits
        //  - Completeness
        // This score can then be saved, and we can use the algorithm with the
        // best score to populate the hits.
        float score = 0.0;

        float ratioOf3Dto2D = 1 - (metrics.numberOf3DHits / totalNumberOf2DHits);

        score += (metrics.acosDotProductAverage * metrics.acosDotProductAverage);
        score += (metrics.distanceToFitAverage * metrics.distanceToFitAverage);
        score += (ratioOf3Dto2D * ratioOf3Dto2D);

        std::cout << "Algorithm " << protoHitVectorPair.first << " metrics were:" << std::endl;
        std::cout << "    Wiggle: " << metrics.acosDotProductAverage << std::endl;
        std::cout << "    Displacement: " << metrics.distanceToFitAverage << std::endl;
        std::cout << "    numberOfHits: " << metrics.numberOf3DHits << "/" << totalNumberOf2DHits << std::endl;
        std::cout << "    Conversion: " << ratioOf3Dto2D << std::endl;
        std::cout << "    Final score: " << score << std::endl;

        scores.push_back(std::make_pair(score, protoHitVectorPair.first));
    }

    std::sort(scores.begin(), scores.end());

    if (scores.size() == 0)
        return;

    // Get the hits from the best algorithm.
    protoHitVector = protoHitVectorMap.at(scores.begin()->second);

    for (auto currentAlgorithm : scores)
    {
        ProtoHitVector currentHits = protoHitVectorMap.at(currentAlgorithm.second);
        unsigned int currentHitCount = 0;

        if (currentHits.size() == 0)
            continue;

        for (ProtoHit protoHit : currentHits)
        {
            auto it = std::find_if(
                    protoHitVector.begin(),
                    protoHitVector.end(),
                    [&protoHit](const ProtoHit& obj) {
                        return obj.GetParentCaloHit2D() == protoHit.GetParentCaloHit2D();
                    });

            // This means we couldn't find a 3D hit that is based on the current 2D hit,
            // so we should add the current hit.
            if (it == protoHitVector.end())
                ++currentHitCount;
        }

        std::cout << "Checking algorithm " << currentAlgorithm.second
                  << ", which scored " << currentAlgorithm.first
                  << " and would have contributed " << (currentHitCount)
                  << " more hits."
                  << std::endl;
    }

    // Now apply the iterative treatment since we've got all our hits.
    this->IterativeTreatment(protoHitVector);

    std::cout << "At the end of consolidation, the protoHitVector was of size: " << protoHitVector.size() << std::endl;

    // Get the final hits and plot them out.
    CartesianPointVector markersForNonInterpolatedHits;

    for (auto hit : protoHitVector)
        markersForNonInterpolatedHits.push_back(hit.GetPosition3D());

    for (auto hit : markersForNonInterpolatedHits)
        PANDORA_MONITORING_API(AddMarkerToVisualization( this->GetPandora(), &hit, "NonInterpolated3DHits", GREEN, 1));
}

//------------------------------------------------------------------------------------------------------------------------------------------

void ThreeDHitCreationAlgorithm::InterpolationMethod(const ParticleFlowObject *const pfo, ProtoHitVector &protoHitVector) const
{
    // If there is no hits at all....we can't do any interpolation.
    if (protoHitVector.empty())
    {
        std::cout << "#### Nothing to interpolate from!" << std::endl;
        return;
    }

    // Get the list of remaining hits for the current PFO.
    // That is, the 2D hits that do not have an associated 3D hit.
    CaloHitVector remainingTwoDHits;
    this->SeparateTwoDHits(pfo, protoHitVector, remainingTwoDHits);

    // If there is no remaining hits, then we don't need to interpolate anything.
    if (remainingTwoDHits.empty())
    {
        std::cout << "#### No need to interpolate!" << std::endl;
        return;
    }

    std::cout << "#### Starting to interpolate hits!" << std::endl;
    std::cout << "#### Initial size was: " << protoHitVector.size() << std::endl;
    std::cout << "#### Remaining hits was: " << remainingTwoDHits.size() << std::endl;

    // Get the current sliding linear fit, such that we can produce a point
    // that fits on to that fit.
    const float layerPitch(LArGeometryHelper::GetWireZPitch(this->GetPandora()));
    const unsigned int layerWindow(m_slidingFitHalfWindow); // TODO: Check if this should be the same or different.

    // Lets store the chi2 so that we can check against it later.
    double originalChi2(0.);
    CartesianPointVector currentPoints3D;
    this->ExtractResults(protoHitVector, originalChi2, currentPoints3D);

    std::cout << "#### Fit params: " << currentPoints3D.size() << ", "
              << layerWindow << ", "
              << layerPitch << std::endl;

    if (currentPoints3D.size() <= 1)
        return;

    const ThreeDSlidingFitResult slidingFitResult(&currentPoints3D, layerWindow, layerPitch);

    int failedToInterpolate = 0;
    int managedToSet = 0;

    CartesianPointVector calosForInterpolatedHits;
    CartesianPointVector markersForInterpolatedHits;

    // We can then look over all these remaining hits and interpolate them.
    // For each hit, we want to compare it to the sliding linear fit, get the
    // points near by to this one and then interpolate the 3D hit from there,
    // using the linked 3D hit from the close by 2D hits that do have a
    // produced 3D hit.
    for (const pandora::CaloHit* currentCaloHit : remainingTwoDHits)
    {
        const CartesianVector pointPosition = LArObjectHelper::TypeAdaptor::GetPosition(currentCaloHit);

        // Get the position relative to the fit for the point.
        const float rL(slidingFitResult.GetLongitudinalDisplacement(pointPosition));

        // Attempt to interpolate the 2D hit.
        CartesianVector projectedPosition(0.f, 0.f, 0.f);
        const StatusCode positionStatusCode(slidingFitResult.GetGlobalFitPosition(rL, projectedPosition));

        if (positionStatusCode != STATUS_CODE_SUCCESS)
        {
            // throw StatusCodeException(positionStatusCode);
            ++failedToInterpolate;
            continue;
        }

        // TODO: At some point, add a cut off so the hits are only interpolated
        // if they are of a certain quality.

        // Now we have a position for the interpolated hit, we need to make a
        // protoHit out of it.  This includes the calculation of a chi2 value,
        // for how good the interpolation is.
        ProtoHit interpolatedHit(currentCaloHit);

        // Project the hit into 2D and get the distance between the projected
        // interpolated hit, and the original 2D hit.
        CartesianVector projectedHit = LArGeometryHelper::ProjectPosition(
                this->GetPandora(),
                projectedPosition,
                currentCaloHit->GetHitType()
        );
        double distanceBetweenHitsSqrd = (
                (currentCaloHit->GetPositionVector() - projectedHit).GetMagnitudeSquared()
        );

        // Using this distance, calculate a chi2 value for the interpolated hit.
        const double sigmaUVW(LArGeometryHelper::GetSigmaUVW(this->GetPandora()));
        const double sigma3DFit(sigmaUVW * m_sigma3DFitMultiplier);
        double interpolatedChi2 = (distanceBetweenHitsSqrd) / (sigma3DFit * sigma3DFit);

        // Set the interpolated hit to have the calculated 3D position, with the chi2.
        interpolatedHit.SetPosition3D(projectedPosition, interpolatedChi2, true);
        interpolatedHit.AddTrajectorySample(
                TrajectorySample(projectedPosition, currentCaloHit->GetHitType(), sigmaUVW)
        );

        // Add the interpolated hit to the protoHitVector.
        protoHitVector.push_back(interpolatedHit);

        CartesianVector caloHit3D(
                currentCaloHit->GetPositionVector().GetX(),
                projectedPosition.GetY(),
                currentCaloHit->GetPositionVector().GetZ()
        );
        calosForInterpolatedHits.push_back(caloHit3D);
        markersForInterpolatedHits.push_back(projectedPosition);
        ++managedToSet;
    }

    std::cout << "#### Done interpolating hits!" << std::endl;
    std::cout << "#### Final size was: " << protoHitVector.size() << std::endl;
    std::cout << "#### Interpolated: " << managedToSet << std::endl;
    std::cout << "#### Failed: " << failedToInterpolate << std::endl;

    if (calosForInterpolatedHits.size() > 0)
    {

        std::cout << "#### Starting to visualise " << calosForInterpolatedHits.size() << " hits." << std::endl;

        for (auto calo : calosForInterpolatedHits)
            PANDORA_MONITORING_API(AddMarkerToVisualization(this->GetPandora(), &calo, "CaloHit", BLUE, 1));

        for (auto marker : markersForInterpolatedHits)
            PANDORA_MONITORING_API(AddMarkerToVisualization( this->GetPandora(), &marker, "Interpolated3DHits", RED, 1));
    }
}

//------------------------------------------------------------------------------------------------------------------------------------------

void ThreeDHitCreationAlgorithm::ExtractResults(const ProtoHitVector &protoHitVector, double &chi2, CartesianPointVector &pointVector) const
{
    chi2 = 0.;
    pointVector.clear();

    for (const ProtoHit &protoHit : protoHitVector)
    {
        chi2 += protoHit.GetChi2();
        pointVector.push_back(protoHit.GetPosition3D());
    }
}

//------------------------------------------------------------------------------------------------------------------------------------------

double ThreeDHitCreationAlgorithm::GetChi2WrtFit(const ThreeDSlidingFitResult &slidingFitResult, const ProtoHitVector &protoHitVector) const
{
    const double sigmaUVW(LArGeometryHelper::GetSigmaUVW(this->GetPandora()));
    const double sigma3DFit(sigmaUVW * m_sigma3DFitMultiplier);

    double chi2WrtFit(0.);

    for (const ProtoHit &protoHit : protoHitVector)
    {
        CartesianVector pointOnFit(0.f, 0.f, 0.f);
        const double rL(slidingFitResult.GetLongitudinalDisplacement(protoHit.GetPosition3D()));

        if (STATUS_CODE_SUCCESS != slidingFitResult.GetGlobalFitPosition(rL, pointOnFit))
            continue;

        const double uFit(PandoraContentApi::GetPlugins(*this)->GetLArTransformationPlugin()->YZtoU(pointOnFit.GetY(), pointOnFit.GetZ()));
        const double vFit(PandoraContentApi::GetPlugins(*this)->GetLArTransformationPlugin()->YZtoV(pointOnFit.GetY(), pointOnFit.GetZ()));
        const double wFit(PandoraContentApi::GetPlugins(*this)->GetLArTransformationPlugin()->YZtoW(pointOnFit.GetY(), pointOnFit.GetZ()));

        const double outputU(PandoraContentApi::GetPlugins(*this)->GetLArTransformationPlugin()->YZtoU(protoHit.GetPosition3D().GetY(), protoHit.GetPosition3D().GetZ()));
        const double outputV(PandoraContentApi::GetPlugins(*this)->GetLArTransformationPlugin()->YZtoV(protoHit.GetPosition3D().GetY(), protoHit.GetPosition3D().GetZ()));
        const double outputW(PandoraContentApi::GetPlugins(*this)->GetLArTransformationPlugin()->YZtoW(protoHit.GetPosition3D().GetY(), protoHit.GetPosition3D().GetZ()));

        const double deltaUFit(uFit - outputU), deltaVFit(vFit - outputV), deltaWFit(wFit - outputW);
        chi2WrtFit += ((deltaUFit * deltaUFit) / (sigma3DFit * sigma3DFit)) + ((deltaVFit * deltaVFit) / (sigma3DFit * sigma3DFit)) + ((deltaWFit * deltaWFit) / (sigma3DFit * sigma3DFit));
    }

    return chi2WrtFit;
}

//------------------------------------------------------------------------------------------------------------------------------------------

double ThreeDHitCreationAlgorithm::GetHitMovementChi2(const ProtoHitVector &protoHitVector) const
{
    const double sigmaUVW(LArGeometryHelper::GetSigmaUVW(this->GetPandora()));
    double hitMovementChi2(0.);

    for (const ProtoHit &protoHit : protoHitVector)
    {
        const CaloHit *const pCaloHit2D(protoHit.GetParentCaloHit2D());
        const HitType hitType(pCaloHit2D->GetHitType());

        const CartesianVector projectedPosition(LArGeometryHelper::ProjectPosition(this->GetPandora(), protoHit.GetPosition3D(), hitType));
        const double delta(static_cast<double>(pCaloHit2D->GetPositionVector().GetZ() - projectedPosition.GetZ()));

        hitMovementChi2 += (delta * delta) / (sigmaUVW * sigmaUVW);
    }

    return hitMovementChi2;
}

//------------------------------------------------------------------------------------------------------------------------------------------

void ThreeDHitCreationAlgorithm::RefineHitPositions(const ThreeDSlidingFitResult &slidingFitResult, ProtoHitVector &protoHitVector) const
{
    const double sigmaUVW(LArGeometryHelper::GetSigmaUVW(this->GetPandora()));
    const double sigmaFit(sigmaUVW); // ATTN sigmaFit and sigmaHit here should agree with treatment in HitCreation tools
    const double sigmaHit(sigmaUVW);
    const double sigma3DFit(sigmaUVW * m_sigma3DFitMultiplier);

    for (ProtoHit &protoHit : protoHitVector)
    {
        CartesianVector pointOnFit(0.f, 0.f, 0.f);
        const double rL(slidingFitResult.GetLongitudinalDisplacement(protoHit.GetPosition3D()));

        if (STATUS_CODE_SUCCESS != slidingFitResult.GetGlobalFitPosition(rL, pointOnFit))
            continue;

        const CaloHit *const pCaloHit2D(protoHit.GetParentCaloHit2D());
        const HitType hitType(pCaloHit2D->GetHitType());

        const double uFit(PandoraContentApi::GetPlugins(*this)->GetLArTransformationPlugin()->YZtoU(pointOnFit.GetY(), pointOnFit.GetZ()));
        const double vFit(PandoraContentApi::GetPlugins(*this)->GetLArTransformationPlugin()->YZtoV(pointOnFit.GetY(), pointOnFit.GetZ()));
        const double wFit(PandoraContentApi::GetPlugins(*this)->GetLArTransformationPlugin()->YZtoW(pointOnFit.GetY(), pointOnFit.GetZ()));

        const double sigmaU((TPC_VIEW_U == hitType) ? sigmaHit : sigmaFit);
        const double sigmaV((TPC_VIEW_V == hitType) ? sigmaHit : sigmaFit);
        const double sigmaW((TPC_VIEW_W == hitType) ? sigmaHit : sigmaFit);

        CartesianVector position3D(0.f, 0.f, 0.f);
        double chi2(std::numeric_limits<double>::max());
        double u(std::numeric_limits<double>::max()), v(std::numeric_limits<double>::max()), w(std::numeric_limits<double>::max());

        if (protoHit.GetNTrajectorySamples() == 2)
        {
            u = (TPC_VIEW_U == hitType) ? pCaloHit2D->GetPositionVector().GetZ() : (TPC_VIEW_U == protoHit.GetFirstTrajectorySample().GetHitType()) ? protoHit.GetFirstTrajectorySample().GetPosition().GetZ() : protoHit.GetLastTrajectorySample().GetPosition().GetZ();
            v = (TPC_VIEW_V == hitType) ? pCaloHit2D->GetPositionVector().GetZ() : (TPC_VIEW_V == protoHit.GetFirstTrajectorySample().GetHitType()) ? protoHit.GetFirstTrajectorySample().GetPosition().GetZ() : protoHit.GetLastTrajectorySample().GetPosition().GetZ();
            w = (TPC_VIEW_W == hitType) ? pCaloHit2D->GetPositionVector().GetZ() : (TPC_VIEW_W == protoHit.GetFirstTrajectorySample().GetHitType()) ? protoHit.GetFirstTrajectorySample().GetPosition().GetZ() : protoHit.GetLastTrajectorySample().GetPosition().GetZ();
        }
        else if (protoHit.GetNTrajectorySamples() == 1)
        {
            u = PandoraContentApi::GetPlugins(*this)->GetLArTransformationPlugin()->YZtoU(protoHit.GetPosition3D().GetY(), protoHit.GetPosition3D().GetZ());
            v = PandoraContentApi::GetPlugins(*this)->GetLArTransformationPlugin()->YZtoV(protoHit.GetPosition3D().GetY(), protoHit.GetPosition3D().GetZ());
            w = PandoraContentApi::GetPlugins(*this)->GetLArTransformationPlugin()->YZtoW(protoHit.GetPosition3D().GetY(), protoHit.GetPosition3D().GetZ());
        }
        else
        {
            std::cout << "ThreeDHitCreationAlgorithm::IterativeTreatment - Unexpected number of trajectory samples" << std::endl;
            throw StatusCodeException(STATUS_CODE_FAILURE);
        }

        double bestY(std::numeric_limits<double>::max()), bestZ(std::numeric_limits<double>::max());
        PandoraContentApi::GetPlugins(*this)->GetLArTransformationPlugin()->GetMinChiSquaredYZ(u, v, w, sigmaU, sigmaV, sigmaW, uFit, vFit, wFit, sigma3DFit, bestY, bestZ, chi2);
        position3D.SetValues(protoHit.GetPosition3D().GetX(), static_cast<float>(bestY), static_cast<float>(bestZ));

        protoHit.SetPosition3D(position3D, chi2);
    }
}

//------------------------------------------------------------------------------------------------------------------------------------------

void ThreeDHitCreationAlgorithm::CreateThreeDHits(const ProtoHitVector &protoHitVector, CaloHitList &newThreeDHits) const
{
    for (const ProtoHit &protoHit : protoHitVector)
    {
        const CaloHit *pCaloHit3D(nullptr);
        this->CreateThreeDHit(protoHit, pCaloHit3D);

        if (!pCaloHit3D)
            throw StatusCodeException(STATUS_CODE_FAILURE);

        newThreeDHits.push_back(pCaloHit3D);
    }
}

//------------------------------------------------------------------------------------------------------------------------------------------

void ThreeDHitCreationAlgorithm::CreateThreeDHit(const ProtoHit &protoHit, const CaloHit *&pCaloHit3D) const
{
    if (!this->CheckThreeDHit(protoHit))
        throw StatusCodeException(STATUS_CODE_INVALID_PARAMETER);

    PandoraContentApi::CaloHit::Parameters parameters;
    parameters.m_positionVector = protoHit.GetPosition3D();
    parameters.m_hitType = TPC_3D;

    const CaloHit *const pCaloHit2D(protoHit.GetParentCaloHit2D());
    parameters.m_pParentAddress = static_cast<const void*>(pCaloHit2D);

    // TODO Check these parameters, especially new cell dimensions
    parameters.m_cellThickness = pCaloHit2D->GetCellThickness();
    parameters.m_cellGeometry = RECTANGULAR;
    parameters.m_cellSize0 = pCaloHit2D->GetCellLengthScale();
    parameters.m_cellSize1 = pCaloHit2D->GetCellLengthScale();
    parameters.m_cellNormalVector = pCaloHit2D->GetCellNormalVector();
    parameters.m_expectedDirection = pCaloHit2D->GetExpectedDirection();
    parameters.m_nCellRadiationLengths = pCaloHit2D->GetNCellRadiationLengths();
    parameters.m_nCellInteractionLengths = pCaloHit2D->GetNCellInteractionLengths();
    parameters.m_time = pCaloHit2D->GetTime();
    parameters.m_inputEnergy = pCaloHit2D->GetInputEnergy();
    parameters.m_mipEquivalentEnergy = pCaloHit2D->GetMipEquivalentEnergy();
    parameters.m_electromagneticEnergy = pCaloHit2D->GetElectromagneticEnergy();
    parameters.m_hadronicEnergy = pCaloHit2D->GetHadronicEnergy();
    parameters.m_isDigital = pCaloHit2D->IsDigital();
    parameters.m_hitRegion = pCaloHit2D->GetHitRegion();
    parameters.m_layer = pCaloHit2D->GetLayer();
    parameters.m_isInOuterSamplingLayer = pCaloHit2D->IsInOuterSamplingLayer();
    PANDORA_THROW_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::CaloHit::Create(*this, parameters, pCaloHit3D));
}

//------------------------------------------------------------------------------------------------------------------------------------------

bool ThreeDHitCreationAlgorithm::CheckThreeDHit(const ProtoHit &protoHit) const
{
    try
    {
        // Check that corresponding pseudo layer is within range - TODO use full LArTPC geometry here
        (void) PandoraContentApi::GetPlugins(*this)->GetPseudoLayerPlugin()->GetPseudoLayer(protoHit.GetPosition3D());
    }
    catch (StatusCodeException &)
    {
        return false;
    }

    // TODO Check against detector geometry
    return true;
}

//------------------------------------------------------------------------------------------------------------------------------------------

void ThreeDHitCreationAlgorithm::AddThreeDHitsToPfo(const ParticleFlowObject *const pPfo, const CaloHitList &caloHitList) const
{
    if (caloHitList.empty())
        throw StatusCodeException(STATUS_CODE_NOT_INITIALIZED);

    ClusterList threeDClusterList;
    LArPfoHelper::GetThreeDClusterList(pPfo, threeDClusterList);

    if (!threeDClusterList.empty())
        throw StatusCodeException(STATUS_CODE_FAILURE);

    const ClusterList *pClusterList(nullptr); std::string clusterListName;
    PANDORA_THROW_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::CreateTemporaryListAndSetCurrent(*this, pClusterList, clusterListName));

    PandoraContentApi::Cluster::Parameters parameters;
    parameters.m_caloHitList.insert(parameters.m_caloHitList.end(), caloHitList.begin(), caloHitList.end());

    const Cluster *pCluster3D(nullptr);
    PANDORA_THROW_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::Cluster::Create(*this, parameters, pCluster3D));

    if (!pCluster3D || !pClusterList || pClusterList->empty())
        throw StatusCodeException(STATUS_CODE_FAILURE);

    PANDORA_THROW_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::SaveList<Cluster>(*this, m_outputClusterListName));
    PANDORA_THROW_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::AddToPfo(*this, pPfo, pCluster3D));
}

//------------------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------------------

const CartesianVector &ThreeDHitCreationAlgorithm::ProtoHit::GetPosition3D() const
{
    if (!m_isPositionSet)
        throw StatusCodeException(STATUS_CODE_NOT_INITIALIZED);

    return m_position3D;
}

//------------------------------------------------------------------------------------------------------------------------------------------

double ThreeDHitCreationAlgorithm::ProtoHit::GetChi2() const
{
    if (!m_isPositionSet)
        throw StatusCodeException(STATUS_CODE_NOT_INITIALIZED);

    return m_chi2;
}

//------------------------------------------------------------------------------------------------------------------------------------------

const ThreeDHitCreationAlgorithm::TrajectorySample &ThreeDHitCreationAlgorithm::ProtoHit::GetFirstTrajectorySample() const
{
    if (m_trajectorySampleVector.empty())
        throw StatusCodeException(STATUS_CODE_NOT_INITIALIZED);

    return m_trajectorySampleVector.front();
}

//------------------------------------------------------------------------------------------------------------------------------------------

const ThreeDHitCreationAlgorithm::TrajectorySample &ThreeDHitCreationAlgorithm::ProtoHit::GetLastTrajectorySample() const
{
    if (m_trajectorySampleVector.size() < 2)
        throw StatusCodeException(STATUS_CODE_NOT_INITIALIZED);

    return m_trajectorySampleVector.back();
}

//------------------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------------------

StatusCode ThreeDHitCreationAlgorithm::ReadSettings(const TiXmlHandle xmlHandle)
{
    AlgorithmToolVector algorithmToolVector;
    PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, XmlHelper::ProcessAlgorithmToolList(*this, xmlHandle,
        "HitCreationTools", algorithmToolVector));

    for (AlgorithmToolVector::const_iterator iter = algorithmToolVector.begin(), iterEnd = algorithmToolVector.end(); iter != iterEnd; ++iter)
    {
        HitCreationBaseTool *const pHitCreationTool(dynamic_cast<HitCreationBaseTool*>(*iter));

        if (!pHitCreationTool)
            return STATUS_CODE_INVALID_PARAMETER;

        m_algorithmToolVector.push_back(pHitCreationTool);
    }

    PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, XmlHelper::ReadValue(xmlHandle, "InputPfoListName", m_inputPfoListName));
    PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, XmlHelper::ReadValue(xmlHandle, "OutputCaloHitListName", m_outputCaloHitListName));
    PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, XmlHelper::ReadValue(xmlHandle, "OutputClusterListName", m_outputClusterListName));

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle,
        "IterateTrackHits", m_iterateTrackHits));

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle,
        "IterateShowerHits", m_iterateShowerHits));

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle,
        "UseInterpolation", m_useInterpolation));

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle,
        "InterpolationCut", m_interpolationCutOff));

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle,
        "SlidingFitHalfWindow", m_slidingFitHalfWindow));

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle,
        "NHitRefinementIterations", m_nHitRefinementIterations));

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle,
        "Sigma3DFitMultiplier", m_sigma3DFitMultiplier));

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle,
        "IterationMaxChi2Ratio", m_iterationMaxChi2Ratio));

    return STATUS_CODE_SUCCESS;
}

} // namespace lar_content
