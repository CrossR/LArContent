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

#include "larpandoracontent/LArObjects/LArThreeDSlidingFitResult.h"

#include "larpandoracontent/LArThreeDReco/LArHitCreation/HitCreationBaseTool.h"
#include "larpandoracontent/LArThreeDReco/LArHitCreation/ThreeDHitCreationAlgorithm.h"
#include "larpandoracontent/LArThreeDReco/LArHitCreation/RANSACMethod.h"

#include <algorithm>

using namespace pandora;

namespace lar_content
{

ThreeDHitCreationAlgorithm::ThreeDHitCreationAlgorithm() :
    m_iterateTrackHits(true),
    m_iterateShowerHits(false),
    m_useRANSACMethod(false),
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
        unsigned int numberOfFailedAlgorithms = 0;

        for (HitCreationBaseTool *const pHitCreationTool : m_algorithmToolVector)
        {
            CaloHitVector remainingTwoDHits;
            this->SeparateTwoDHits(pPfo, protoHitVector, remainingTwoDHits);

            if (remainingTwoDHits.empty())
                break;

            pHitCreationTool->Run(this, pPfo, remainingTwoDHits, protoHitVector);

            if (m_useRANSACMethod && LArPfoHelper::IsTrack(pPfo))
            {
                // TODO: Replace 10 with a configuration controlled number.
                for (unsigned int i = 0; i < 10; ++i)
                {
                    const unsigned int sizeBefore = protoHitVector.size();
                    this->InterpolationMethod(pPfo, protoHitVector);
                    const unsigned int sizeAfter = protoHitVector.size();

                    if (sizeBefore == sizeAfter)
                        break;
                }

                this->IterativeTreatment(protoHitVector);
                allProtoHitVectors.insert(ProtoHitVectorMap::value_type(pHitCreationTool->GetInstanceName(), protoHitVector));
                protoHitVector.clear();
            }
        }

        if (numberOfFailedAlgorithms == m_algorithmToolVector.size())
            continue;

        const bool shouldUseIterativeTreatment = (
                (m_iterateTrackHits && LArPfoHelper::IsTrack(pPfo)) ||
                (m_iterateShowerHits && LArPfoHelper::IsShower(pPfo))
        );

        // ATTN: Skip for RANSAC, since it will be done later.
        if (shouldUseIterativeTreatment && !m_useRANSACMethod)
            this->IterativeTreatment(protoHitVector);

        if (m_useRANSACMethod && LArPfoHelper::IsTrack(pPfo))
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

void ThreeDHitCreationAlgorithm::Project3DHit(const ProtoHit &hit, const HitType view, ProtoHit &projectedHit)
{
    projectedHit.SetPosition3D(LArGeometryHelper::ProjectPosition(this->GetPandora(), hit.GetPosition3D(), view), hit.GetChi2());
}

//------------------------------------------------------------------------------------------------------------------------------------------

void ThreeDHitCreationAlgorithm::GetSetIntersection(RANSACHitVector &first, RANSACHitVector &second, RANSACHitVector &result)
{
    const auto compareFunction = [] (const RANSACHit &a, const RANSACHit &b) -> bool {
        return a.GetProtoHit().GetPosition3D().GetX() < b.GetProtoHit().GetPosition3D().GetX();
    };

    std::sort(first.begin(), first.end(), compareFunction);
    std::sort(second.begin(), second.end(), compareFunction);

    std::set_intersection(first.begin(), first.end(), second.begin(), second.end(), std::inserter(result, result.begin()), compareFunction);
}

//------------------------------------------------------------------------------------------------------------------------------------------

void ThreeDHitCreationAlgorithm::ConsolidatedMethod(const ParticleFlowObject *const pPfo, ProtoHitVectorMap &allProtoHitVectors, ProtoHitVector &protoHitVector)
{
    if (allProtoHitVectors.size() == 0)
        return;

    const float DISTANCE_THRESHOLD(0.05); // TODO: Move to config option.
    const std::vector<HitType> views = {TPC_VIEW_U, TPC_VIEW_V, TPC_VIEW_W};
    const std::vector<std::string> toolsToAvoid = {"Tool0039", "Tool0043"}; // TODO: Config option?
    std::map<HitType, RANSACHitVector> goodHits;

    for (auto toolVectorPair : allProtoHitVectors)
    {
        if (toolVectorPair.second.size() == 0)
            continue;

        const auto avoidedIt = std::find(toolsToAvoid.begin(), toolsToAvoid.end(), toolVectorPair.first);
        const bool goodTool = avoidedIt == toolsToAvoid.end();

        // INFO: Project every 3D hit into all 2D views, so how well they match
        // can be compared. This is only really a concern for the tools which
        // can explode, however, they do provide useful inputs so need to
        // still be considered.
        for (const auto &hit : toolVectorPair.second)
        {
            const CaloHit* twoDHit = hit.GetParentCaloHit2D();

            if (twoDHit == nullptr)
                continue;

            for (HitType view : views)
            {
                ProtoHit hitForView(twoDHit);
                this->Project3DHit(hit, view, hitForView);

                if (goodTool)
                {
                    goodHits[view].push_back(RANSACHit(hit, goodTool));
                    continue;
                }

                const float disp = std::fabs(hitForView.GetPosition3D().GetX() - twoDHit->GetPositionVector().GetX());

                if (disp <= DISTANCE_THRESHOLD)
                    goodHits[view].push_back(RANSACHit(hit, goodTool));
            }
        }
    }

    RANSACHitVector UVconsistentHits;
    this->GetSetIntersection(goodHits[TPC_VIEW_V], goodHits[TPC_VIEW_U], UVconsistentHits);

    RANSACHitVector consistentHits;
    this->GetSetIntersection(goodHits[TPC_VIEW_W], UVconsistentHits, consistentHits);

    RANSACMethodTool ransacMethod;
    ransacMethod.Run(consistentHits, protoHitVector);

    this->InterpolationMethod(pPfo, protoHitVector);
    this->IterativeTreatment(protoHitVector);
}

//------------------------------------------------------------------------------------------------------------------------------------------

void ThreeDHitCreationAlgorithm::InterpolationMethod(const ParticleFlowObject *const pfo, ProtoHitVector &protoHitVector) const
{
    if (protoHitVector.empty())
        return;

    CaloHitVector remainingTwoDHits;
    this->SeparateTwoDHits(pfo, protoHitVector, remainingTwoDHits);

    if (remainingTwoDHits.empty())
        return;

    const float layerPitch(LArGeometryHelper::GetWireZPitch(this->GetPandora()));
    const unsigned int layerWindow(100); // TODO: Check if this should be the same or different.

    double originalChi2(0.);
    CartesianPointVector currentPoints3D;
    this->ExtractResults(protoHitVector, originalChi2, currentPoints3D);

    if (currentPoints3D.size() <= 1)
        return;

    const ThreeDSlidingFitResult slidingFitResult(&currentPoints3D, layerWindow, layerPitch);
    // CartesianVector fitDirection = slidingFitResult.GetGlobalMaxLayerDirection();

    const float sizeBefore = protoHitVector.size();

    for (const pandora::CaloHit* currentCaloHit : remainingTwoDHits)
    {
        const CartesianVector pointPosition = LArObjectHelper::TypeAdaptor::GetPosition(currentCaloHit);

        const float rL(slidingFitResult.GetLongitudinalDisplacement(pointPosition));

        CartesianVector projectedPosition(0.f, 0.f, 0.f);
        CartesianVector projectedDirection(0.f, 0.f, 0.f);
        const StatusCode positionStatusCode(slidingFitResult.GetGlobalFitPosition(rL, projectedPosition));
        const StatusCode directionStatusCode(slidingFitResult.GetGlobalFitDirection(rL, projectedDirection));

        if (positionStatusCode != STATUS_CODE_SUCCESS || directionStatusCode != STATUS_CODE_SUCCESS)
            continue;

        // TODO: Tune cut off.
        // Now we have a position for the interpolated hit, we need to make a
        // protoHit out of it.  This includes the calculation of a chi2 value,
        // for how good the interpolation is.
        // const float displacement = projectedPosition.GetCrossProduct(fitDirection).GetMagnitude();
        // const float otherDisp = projectedPosition.GetCrossProduct(projectedDirection).GetMagnitude();

        // if (otherDisp > 150)
        //     continue;

        ProtoHit interpolatedHit(currentCaloHit);

        const CartesianVector projectedHit(LArGeometryHelper::ProjectPosition(this->GetPandora(), projectedPosition, currentCaloHit->GetHitType()));
        const double distanceBetweenHitsSqrd((currentCaloHit->GetPositionVector() - projectedHit).GetMagnitudeSquared());

        const double sigmaUVW(LArGeometryHelper::GetSigmaUVW(this->GetPandora()));
        const double sigma3DFit(sigmaUVW * m_sigma3DFitMultiplier);
        const double interpolatedChi2 = (distanceBetweenHitsSqrd) / (sigma3DFit * sigma3DFit);

        interpolatedHit.SetPosition3D(projectedPosition, interpolatedChi2);
        interpolatedHit.SetInterpolated(true);
        interpolatedHit.AddTrajectorySample(TrajectorySample(projectedPosition, currentCaloHit->GetHitType(), sigmaUVW));

        protoHitVector.push_back(interpolatedHit);
    }

    // ATTN: If we've interpolated at least 80% of this particle, don't use it.
    //
    // TODO: Swap to option?
    // TODO: This ideally would be earlier on, and wouldn't clear, but just drop the interpolated.
    const float numberOfInterpolatedHits = protoHitVector.size() - sizeBefore;
    if (numberOfInterpolatedHits >= (0.8 * protoHitVector.size()))
        protoHitVector.clear();
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
            // throw StatusCodeException(STATUS_CODE_FAILURE);
            continue; // TODO: Fix this, check why it happens and if it is specific to the changes I've made.
        }

        double bestY(std::numeric_limits<double>::max()), bestZ(std::numeric_limits<double>::max());
        PandoraContentApi::GetPlugins(*this)->GetLArTransformationPlugin()->GetMinChiSquaredYZ(u, v, w, sigmaU, sigmaV, sigmaW, uFit, vFit, wFit, sigma3DFit, bestY, bestZ, chi2);

        if (m_useRANSACMethod)
            position3D.SetValues(protoHit.GetPosition3D().GetX(), static_cast<float>(bestY), static_cast<float>(bestZ));
        else
            position3D.SetValues(pCaloHit2D->GetPositionVector().GetX(), static_cast<float>(bestY), static_cast<float>(bestZ));

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
        "UseRANSAC", m_useRANSACMethod));

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
