/**
 *  @file   larpandoradlcontent/LArHelpers/LArDLHelper.cc
 *
 *  @brief  Implementation of the lar deep learning helper helper class.
 *
 *  $Log: $
 */

#include "larpandoradlcontent/LArHelpers/LArDLHelper.h"

namespace lar_dl_content
{

using namespace pandora;

StatusCode LArDLHelper::LoadModel(const std::string &filename, LArDLHelper::TorchModel &model)
{
    try
    {
        torch::jit::setGraphExecutorOptimize(false);
        torch::jit::getProfilingMode() = false;
        torch::jit::getExecutorMode() = false;
        model = torch::jit::load(filename);
        std::cout << "Loaded the TorchScript model \'" << filename << "\'" << std::endl;
    }
    catch (...)
    {
        std::cout << "Error loading the TorchScript model \'" << filename << "\'" << std::endl;
        return STATUS_CODE_FAILURE;
    }

    return STATUS_CODE_SUCCESS;
}

//------------------------------------------------------------------------------------------------------------------------------------------

void LArDLHelper::InitialiseInput(const at::IntArrayRef dimensions, TorchInput &tensor)
{
    tensor = torch::zeros(dimensions);
}

//------------------------------------------------------------------------------------------------------------------------------------------

void LArDLHelper::InitialiseInput(const at::IntArrayRef dimensions, TorchInput &tensor, const torch::TensorOptions &options)
{
    tensor = torch::zeros(dimensions, options);
}

//------------------------------------------------------------------------------------------------------------------------------------------

void LArDLHelper::Forward(TorchModel &model, const TorchInputVector &input, TorchOutput &output)
{
    // ATTN: Disable gradient calculation, since we aren't training.
    //       Lessens memory consmuption.
    torch::NoGradGuard no_grad;

    output = model.forward(input).toTensor();
}

} // namespace lar_dl_content
